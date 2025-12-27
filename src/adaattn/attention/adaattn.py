"""
AdaAttn: Combined adaptive rank and precision attention.

This module provides the flagship AdaAttn implementation that
combines adaptive rank selection and precision control.
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
    AttentionMetrics,
    Timer,
)
from adaattn.attention.adaptive_rank import AdaptiveRankAttention
from adaattn.attention.adaptive_precision import AdaptivePrecisionAttention


class AdaAttention(BaseAttention):
    """
    Adaptive Precision & Rank Attention (AdaAttn).

    This is the flagship implementation combining:
    - Adaptive rank selection (dense vs low-rank)
    - Adaptive precision control (FP32/FP16/BF16/FP8)
    - GPU-optimized computation paths

    The module dynamically selects the most efficient computation
    strategy based on runtime analysis of attention patterns.

    Example:
        >>> attn = AdaAttention(embed_dim=512, num_heads=8)
        >>> q = torch.randn(2, 128, 512)
        >>> k = torch.randn(2, 128, 512)
        >>> v = torch.randn(2, 128, 512)
        >>> output, weights = attn(q, k, v)
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.0,
        # Rank adaptation settings
        enable_adaptive_rank: bool = True,
        rank_ratio: float = 0.5,
        rank_estimation_method: Literal["entropy", "power", "random"] = "entropy",
        rank_threshold: float = 0.95,
        # Precision settings
        enable_adaptive_precision: bool = True,
        default_precision: PrecisionMode = PrecisionMode.FP16,
        allow_fp8: bool = False,
        # Attention settings
        causal: bool = False,
        bias: bool = True,
        # Config override
        config: Optional[AttentionConfig] = None,
        **kwargs,
    ) -> None:
        """
        Initialize AdaAttention.

        Args:
            embed_dim: Total embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            enable_adaptive_rank: Whether to enable adaptive rank selection
            rank_ratio: Default ratio of rank for low-rank path
            rank_estimation_method: Method for rank estimation
            rank_threshold: Threshold for rank selection decision
            enable_adaptive_precision: Whether to enable adaptive precision
            default_precision: Default precision mode
            allow_fp8: Whether to allow FP8 precision
            causal: Whether to apply causal masking
            bias: Whether to use bias in projections
            config: Optional AttentionConfig override
            **kwargs: Additional config parameters
        """
        if config is None:
            config = AttentionConfig(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                enable_adaptive_rank=enable_adaptive_rank,
                rank_threshold=rank_threshold,
                precision=default_precision,
                **kwargs,
            )

        super().__init__(config=config)

        self.enable_adaptive_rank = enable_adaptive_rank
        self.enable_adaptive_precision = enable_adaptive_precision
        self.rank_ratio = rank_ratio
        self.rank_estimation_method = rank_estimation_method
        self.rank_threshold = rank_threshold
        self.default_precision = default_precision
        self.allow_fp8 = allow_fp8
        self.causal = causal

        # Linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Low-rank projections for adaptive rank
        if enable_adaptive_rank:
            max_rank = max(1, int(self.head_dim * rank_ratio))
            self.rank_proj_down = nn.Linear(self.head_dim, max_rank, bias=False)
            self.rank_proj_up = nn.Linear(max_rank, self.head_dim, bias=False)
        else:
            self.rank_proj_down = None
            self.rank_proj_up = None

        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Scaling factor
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Statistics tracking
        self.register_buffer("_call_count", torch.tensor(0))
        self.register_buffer("_low_rank_calls", torch.tensor(0))
        self.register_buffer("_precision_distribution", torch.zeros(4))

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

        for proj in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)

        if self.rank_proj_down is not None:
            nn.init.orthogonal_(self.rank_proj_down.weight)
        if self.rank_proj_up is not None:
            nn.init.orthogonal_(self.rank_proj_up.weight)

    def _analyze_attention_pattern(
        self,
        query: Tensor,
        key: Tensor,
    ) -> Tuple[bool, PrecisionMode, float]:
        """
        Analyze attention pattern to determine optimal computation strategy.

        Args:
            query: Query tensor (batch, heads, seq_q, head_dim)
            key: Key tensor (batch, heads, seq_k, head_dim)

        Returns:
            Tuple of:
            - use_low_rank: Whether to use low-rank approximation
            - precision: Recommended precision mode
            - effective_rank: Estimated effective rank
        """
        with torch.no_grad():
            # Sample for efficiency
            seq_q, seq_k = query.size(2), key.size(2)
            sample_size = min(64, seq_q, seq_k)

            q_sample = query[:, :, :sample_size]
            k_sample = key[:, :, :sample_size]

            # Compute sample attention scores
            scores = torch.matmul(q_sample, k_sample.transpose(-2, -1)) * self.scale

            # Compute softmax
            probs = F.softmax(scores, dim=-1)

            # Compute entropy
            entropy = -(probs * torch.log(probs + self.config.eps)).sum(dim=-1)
            max_entropy = math.log(sample_size)
            normalized_entropy = (entropy / max_entropy).mean().item()

            # Effective rank estimation
            effective_rank = normalized_entropy * min(seq_q, seq_k)

            # Rank decision
            use_low_rank = self.enable_adaptive_rank and (
                normalized_entropy < self.rank_threshold
            )

            # Precision decision
            max_score = scores.abs().max().item()
            max_prob = probs.max().item()

            if max_score > 50.0 or max_prob > 0.95:
                precision = PrecisionMode.FP32
            elif normalized_entropy > 0.8:
                precision = PrecisionMode.FP16 if not self.allow_fp8 else PrecisionMode.FP8
            elif normalized_entropy < 0.3:
                precision = PrecisionMode.BF16
            else:
                precision = self.default_precision

            if not self.enable_adaptive_precision:
                precision = self.default_precision

            return use_low_rank, precision, effective_rank

    def _compute_dense_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        precision: PrecisionMode,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute standard dense attention."""
        target_dtype = precision.to_dtype()
        original_dtype = query.dtype

        q = query.to(target_dtype)
        k = key.to(target_dtype)
        v = value.to(target_dtype)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if precision in (PrecisionMode.FP16, PrecisionMode.FP8):
            scores = torch.clamp(scores, min=-65504, max=65504)

        if self.causal:
            seq_q, seq_k = scores.size(-2), scores.size(-1)
            causal_mask = torch.triu(
                torch.ones(seq_q, seq_k, dtype=torch.bool, device=scores.device),
                diagonal=1,
            )
            scores = scores.masked_fill(causal_mask, float("-inf"))

        if attention_mask is not None:
            if attention_mask.ndim == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            elif attention_mask.ndim == 3:
                attention_mask = attention_mask.unsqueeze(1)

            if attention_mask.dtype == torch.bool:
                scores = scores.masked_fill(~attention_mask, float("-inf"))
            else:
                scores = scores + attention_mask.to(target_dtype)

        if precision in (PrecisionMode.FP8, PrecisionMode.FP16):
            attn_weights = F.softmax(scores.float(), dim=-1).to(target_dtype)
        else:
            attn_weights = F.softmax(scores, dim=-1)

        attn_weights = self.attn_dropout(attn_weights)
        output = torch.matmul(attn_weights, v)

        return output.to(original_dtype)

    def _compute_low_rank_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        precision: PrecisionMode,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute low-rank approximation of attention."""
        if self.rank_proj_down is None:
            return self._compute_dense_attention(query, key, value, precision, attention_mask)

        target_dtype = precision.to_dtype()
        original_dtype = query.dtype

        q = query.to(target_dtype)
        k = key.to(target_dtype)
        v = value.to(target_dtype)

        # Project to low-rank space
        q_low = self.rank_proj_down(q)
        k_low = self.rank_proj_down(k)

        scores = torch.matmul(q_low, k_low.transpose(-2, -1)) * self.scale

        if self.causal:
            seq_q, seq_k = scores.size(-2), scores.size(-1)
            causal_mask = torch.triu(
                torch.ones(seq_q, seq_k, dtype=torch.bool, device=scores.device),
                diagonal=1,
            )
            scores = scores.masked_fill(causal_mask, float("-inf"))

        if attention_mask is not None:
            if attention_mask.ndim == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            elif attention_mask.ndim == 3:
                attention_mask = attention_mask.unsqueeze(1)

            if attention_mask.dtype == torch.bool:
                scores = scores.masked_fill(~attention_mask, float("-inf"))
            else:
                scores = scores + attention_mask.to(target_dtype)

        if precision in (PrecisionMode.FP8, PrecisionMode.FP16):
            attn_weights = F.softmax(scores.float(), dim=-1).to(target_dtype)
        else:
            attn_weights = F.softmax(scores, dim=-1)

        attn_weights = self.attn_dropout(attn_weights)
        output = torch.matmul(attn_weights, v)

        return output.to(original_dtype)

    def _forward_impl(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Implement AdaAttention forward pass.
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

        # Analyze attention pattern
        use_low_rank, precision, effective_rank = self._analyze_attention_pattern(q, k)

        # Update statistics
        self._call_count += 1
        if use_low_rank:
            self._low_rank_calls += 1

        precision_idx = {
            PrecisionMode.FP32: 0,
            PrecisionMode.FP16: 1,
            PrecisionMode.BF16: 2,
            PrecisionMode.FP8: 3,
        }.get(precision, 0)
        self._precision_distribution[precision_idx] += 1

        # Update metrics
        if self._metrics is not None:
            self._metrics.effective_rank = effective_rank
            self._metrics.precision_mode = precision

        # Compute attention
        if use_low_rank:
            attn_output = self._compute_low_rank_attention(
                q, k, v, precision, attention_mask
            )
        else:
            attn_output = self._compute_dense_attention(
                q, k, v, precision, attention_mask
            )

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len_q, self.embed_dim)

        # Output projection
        output = self.out_proj(attn_output)
        output = self.resid_dropout(output)

        return output, None

    def get_statistics(self) -> dict:
        """Get comprehensive statistics."""
        call_count = self._call_count.item()
        if call_count > 0:
            low_rank_ratio = self._low_rank_calls.item() / call_count
            precision_dist = (self._precision_distribution / call_count).tolist()
        else:
            low_rank_ratio = 0.0
            precision_dist = [0.0, 0.0, 0.0, 0.0]

        return {
            "call_count": call_count,
            "low_rank_ratio": low_rank_ratio,
            "precision_distribution": {
                "FP32": precision_dist[0],
                "FP16": precision_dist[1],
                "BF16": precision_dist[2],
                "FP8": precision_dist[3],
            },
        }

    def extra_repr(self) -> str:
        """Return extra representation."""
        base_repr = super().extra_repr()
        return (
            f"{base_repr}, adaptive_rank={self.enable_adaptive_rank}, "
            f"adaptive_precision={self.enable_adaptive_precision}"
        )
