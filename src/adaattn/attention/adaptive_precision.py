"""
Adaptive precision attention implementation.

This module provides attention that dynamically adjusts numerical
precision based on runtime analysis of attention patterns.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from adaattn.attention.base import (
    AttentionConfig,
    BaseAttention,
    PrecisionMode,
    NumericalError,
    Timer,
)


class AdaptivePrecisionAttention(BaseAttention):
    """
    Attention with adaptive precision control.

    Dynamically adjusts precision (FP32/FP16/BF16/FP8) based on:
    - Attention entropy
    - QK score magnitudes
    - Softmax saturation

    Example:
        >>> attn = AdaptivePrecisionAttention(embed_dim=512, num_heads=8)
        >>> q = torch.randn(2, 128, 512)
        >>> k = torch.randn(2, 128, 512)
        >>> v = torch.randn(2, 128, 512)
        >>> output, weights = attn(q, k, v)
    """

    # Thresholds for precision decisions
    ENTROPY_HIGH_THRESHOLD = 0.8  # High entropy -> can use lower precision
    ENTROPY_LOW_THRESHOLD = 0.3   # Low entropy -> may need higher precision
    SCORE_MAGNITUDE_THRESHOLD = 50.0  # Large scores -> need higher precision
    SATURATION_THRESHOLD = 0.95  # Near-saturation -> need higher precision

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.0,
        default_precision: PrecisionMode = PrecisionMode.FP16,
        allow_fp8: bool = False,
        precision_fallback: bool = True,
        causal: bool = False,
        config: Optional[AttentionConfig] = None,
        **kwargs,
    ) -> None:
        """
        Initialize adaptive precision attention.

        Args:
            embed_dim: Total embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            default_precision: Default precision mode
            allow_fp8: Whether to allow FP8 precision
            precision_fallback: Whether to fall back to higher precision on errors
            causal: Whether to apply causal masking
            config: Optional AttentionConfig
            **kwargs: Additional config parameters
        """
        if config is None:
            config = AttentionConfig(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                precision=default_precision,
                allow_precision_fallback=precision_fallback,
                **kwargs,
            )

        super().__init__(config=config)

        self.default_precision = default_precision
        self.allow_fp8 = allow_fp8
        self.precision_fallback = precision_fallback
        self.causal = causal

        # Linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Scaling factor
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Precision tracking
        self.register_buffer("_precision_switches", torch.tensor(0))
        self.register_buffer("_precision_distribution", torch.zeros(4))  # FP32, FP16, BF16, FP8

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize parameters."""
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(proj.weight)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)

    def _analyze_precision_requirements(
        self,
        query: Tensor,
        key: Tensor,
    ) -> PrecisionMode:
        """
        Analyze inputs to determine optimal precision.

        Args:
            query: Query tensor
            key: Key tensor

        Returns:
            Recommended PrecisionMode
        """
        with torch.no_grad():
            # Compute approximate attention scores
            q_sample = query[:, :, :min(64, query.size(2))]
            k_sample = key[:, :, :min(64, key.size(2))]

            scores = torch.matmul(q_sample, k_sample.transpose(-2, -1)) * self.scale

            # Check score magnitudes
            max_score = scores.abs().max().item()
            if max_score > self.SCORE_MAGNITUDE_THRESHOLD:
                return PrecisionMode.FP32

            # Compute softmax and check for saturation
            probs = F.softmax(scores, dim=-1)
            max_prob = probs.max().item()
            if max_prob > self.SATURATION_THRESHOLD:
                # Near-saturated attention - need higher precision
                return PrecisionMode.FP32

            # Compute entropy
            entropy = -(probs * torch.log(probs + self.config.eps)).sum(dim=-1)
            max_entropy = math.log(probs.size(-1))
            normalized_entropy = (entropy / max_entropy).mean().item()

            # Decide based on entropy
            if normalized_entropy > self.ENTROPY_HIGH_THRESHOLD:
                # High entropy (uniform) - can use lower precision
                if self.allow_fp8:
                    return PrecisionMode.FP8
                return PrecisionMode.FP16

            elif normalized_entropy < self.ENTROPY_LOW_THRESHOLD:
                # Low entropy (peaked) - may need higher precision
                return PrecisionMode.BF16

            else:
                # Medium entropy - use default
                return self.default_precision

    def _compute_attention_with_precision(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        precision: PrecisionMode,
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Compute attention with specified precision.

        Args:
            query: Query tensor (batch, heads, seq_q, head_dim)
            key: Key tensor (batch, heads, seq_k, head_dim)
            value: Value tensor (batch, heads, seq_k, head_dim)
            precision: Precision mode to use
            attention_mask: Optional attention mask

        Returns:
            Attention output and weights
        """
        original_dtype = query.dtype
        target_dtype = precision.to_dtype()

        # Cast to target precision
        q = query.to(target_dtype)
        k = key.to(target_dtype)
        v = value.to(target_dtype)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Clamp to prevent overflow
        if precision in (PrecisionMode.FP16, PrecisionMode.FP8):
            scores = torch.clamp(scores, min=-65504, max=65504)

        # Apply causal mask if needed
        if self.causal:
            seq_q, seq_k = scores.size(-2), scores.size(-1)
            causal_mask = torch.triu(
                torch.ones(seq_q, seq_k, dtype=torch.bool, device=scores.device),
                diagonal=1,
            )
            scores = scores.masked_fill(causal_mask, float("-inf"))

        # Apply attention mask if provided
        if attention_mask is not None:
            if attention_mask.ndim == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            elif attention_mask.ndim == 3:
                attention_mask = attention_mask.unsqueeze(1)

            if attention_mask.dtype == torch.bool:
                scores = scores.masked_fill(~attention_mask, float("-inf"))
            else:
                scores = scores + attention_mask.to(target_dtype)

        # Softmax in higher precision for stability
        if precision in (PrecisionMode.FP8, PrecisionMode.FP16):
            scores_fp32 = scores.float()
            attn_weights = F.softmax(scores_fp32, dim=-1).to(target_dtype)
        else:
            attn_weights = F.softmax(scores, dim=-1)

        # Apply dropout
        attn_weights = self.attn_dropout(attn_weights)

        # Compute output
        output = torch.matmul(attn_weights, v)

        # Cast back to original dtype
        output = output.to(original_dtype)

        return output, attn_weights

    def _forward_impl(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Implement adaptive precision attention forward pass.

        Args:
            query: Query tensor (batch, seq_q, embed_dim)
            key: Key tensor (batch, seq_k, embed_dim)
            value: Value tensor (batch, seq_k, embed_dim)
            attention_mask: Optional attention mask
            need_weights: Whether to return attention weights

        Returns:
            Tuple of output tensor and optional attention weights
        """
        batch_size, seq_len_q, _ = query.shape
        _, seq_len_k, _ = key.shape

        # Project Q, K, V
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)

        # Determine optimal precision
        precision = self._analyze_precision_requirements(q, k)

        # Update metrics
        if self._metrics is not None:
            self._metrics.precision_mode = precision

        # Track precision usage
        precision_idx = {
            PrecisionMode.FP32: 0,
            PrecisionMode.FP16: 1,
            PrecisionMode.BF16: 2,
            PrecisionMode.FP8: 3,
        }.get(precision, 0)
        self._precision_distribution[precision_idx] += 1

        # Compute attention with selected precision
        try:
            attn_output, attn_weights = self._compute_attention_with_precision(
                q, k, v, precision, attention_mask
            )
        except (RuntimeError, FloatingPointError) as e:
            # Fallback to higher precision on error
            if self.precision_fallback and precision != PrecisionMode.FP32:
                self._precision_switches += 1
                attn_output, attn_weights = self._compute_attention_with_precision(
                    q, k, v, PrecisionMode.FP32, attention_mask
                )
            else:
                raise NumericalError(f"Attention computation failed: {e}")

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len_q, self.embed_dim)

        # Output projection
        output = self.out_proj(attn_output)
        output = self.resid_dropout(output)

        if need_weights:
            avg_weights = attn_weights.mean(dim=1)
            return output, avg_weights
        else:
            return output, None

    def get_precision_statistics(self) -> dict:
        """Get statistics on precision usage."""
        total = self._precision_distribution.sum().item()
        if total > 0:
            distribution = (self._precision_distribution / total).tolist()
        else:
            distribution = [0.0, 0.0, 0.0, 0.0]

        return {
            "precision_switches": self._precision_switches.item(),
            "distribution": {
                "FP32": distribution[0],
                "FP16": distribution[1],
                "BF16": distribution[2],
                "FP8": distribution[3],
            },
        }

    def extra_repr(self) -> str:
        """Return extra representation."""
        base_repr = super().extra_repr()
        return f"{base_repr}, default_precision={self.default_precision.name}"
