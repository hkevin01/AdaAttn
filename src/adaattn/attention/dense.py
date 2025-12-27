"""
Dense (standard) scaled dot-product attention implementation.

This module provides the baseline dense attention for comparison
and fallback when adaptive methods are not beneficial.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from adaattn.attention.base import (
    AttentionConfig,
    BaseAttention,
    NumericalError,
    Timer,
)


class DenseAttention(BaseAttention):
    """
    Standard scaled dot-product attention.

    This implements the classic attention formula:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    With optional:
    - Multi-head attention
    - Dropout
    - Attention mask
    - Causal masking

    Example:
        >>> attn = DenseAttention(embed_dim=512, num_heads=8)
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
        bias: bool = True,
        add_bias_kv: bool = False,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        causal: bool = False,
        config: Optional[AttentionConfig] = None,
        **kwargs,
    ) -> None:
        """
        Initialize dense attention.

        Args:
            embed_dim: Total embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            bias: Whether to add bias to projections
            add_bias_kv: Whether to add bias to key/value
            kdim: Key dimension (defaults to embed_dim)
            vdim: Value dimension (defaults to embed_dim)
            causal: Whether to apply causal masking
            config: Optional AttentionConfig to use
            **kwargs: Additional config parameters
        """
        # Create config if not provided
        if config is None:
            config = AttentionConfig(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                **kwargs,
            )

        super().__init__(config=config)

        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.causal = causal

        # Validate dimensions
        if self.kdim % num_heads != 0:
            raise ValueError(
                f"kdim ({self.kdim}) must be divisible by num_heads ({num_heads})"
            )

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Optional bias for key and value
        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.bias_v = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.register_parameter("bias_k", None)
            self.register_parameter("bias_v", None)

        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Scaling factor
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Initialize weights
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize parameters using Xavier uniform."""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

        if self.q_proj.bias is not None:
            nn.init.zeros_(self.q_proj.bias)
        if self.k_proj.bias is not None:
            nn.init.zeros_(self.k_proj.bias)
        if self.v_proj.bias is not None:
            nn.init.zeros_(self.v_proj.bias)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def _reshape_for_attention(
        self, x: Tensor, batch_size: int, seq_len: int
    ) -> Tensor:
        """
        Reshape tensor for multi-head attention.

        Args:
            x: Input tensor of shape (batch, seq, embed_dim)
            batch_size: Batch size
            seq_len: Sequence length

        Returns:
            Reshaped tensor of shape (batch, num_heads, seq, head_dim)
        """
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)  # (batch, heads, seq, head_dim)

    def _compute_attention_scores(
        self,
        query: Tensor,
        key: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute scaled dot-product attention scores.

        Args:
            query: Query tensor (batch, heads, seq_q, head_dim)
            key: Key tensor (batch, heads, seq_k, head_dim)
            attention_mask: Optional mask

        Returns:
            Attention scores (batch, heads, seq_q, seq_k)
        """
        # Compute QK^T
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale

        # Apply temperature scaling if configured
        if self.config.softmax_temperature != 1.0:
            scores = scores / self.config.softmax_temperature

        # Clamp scores to prevent overflow
        if self.config.attention_clamp_value > 0:
            scores = torch.clamp(
                scores,
                min=-self.config.attention_clamp_value,
                max=self.config.attention_clamp_value,
            )

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
            # Handle different mask formats
            if attention_mask.ndim == 2:
                # (batch, seq_k) -> (batch, 1, 1, seq_k)
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            elif attention_mask.ndim == 3:
                # (batch, seq_q, seq_k) -> (batch, 1, seq_q, seq_k)
                attention_mask = attention_mask.unsqueeze(1)

            # Apply mask
            if attention_mask.dtype == torch.bool:
                scores = scores.masked_fill(~attention_mask, float("-inf"))
            else:
                scores = scores + attention_mask

        return scores

    def _forward_impl(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Implement dense attention forward pass.

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

        # Add bias to K, V if configured
        if self.bias_k is not None and self.bias_v is not None:
            k = torch.cat([k, self.bias_k.repeat(batch_size, 1, 1)], dim=1)
            v = torch.cat([v, self.bias_v.repeat(batch_size, 1, 1)], dim=1)
            if attention_mask is not None:
                # Extend mask for bias
                attention_mask = F.pad(attention_mask, (0, 1), value=True)

        # Reshape for multi-head attention
        q = self._reshape_for_attention(q, batch_size, seq_len_q)
        k = self._reshape_for_attention(k, batch_size, k.size(1))
        v = self._reshape_for_attention(v, batch_size, v.size(1))

        # Compute attention scores
        if self.config.enable_profiling and self._metrics is not None:
            with Timer(sync_cuda=True, unit="ms") as timer:
                attn_scores = self._compute_attention_scores(q, k, attention_mask)
            self._metrics.qk_matmul_time_ms = timer.elapsed
        else:
            attn_scores = self._compute_attention_scores(q, k, attention_mask)

        # Apply softmax
        if self.config.enable_profiling and self._metrics is not None:
            with Timer(sync_cuda=True, unit="ms") as timer:
                attn_weights = F.softmax(attn_scores, dim=-1)
            self._metrics.softmax_time_ms = timer.elapsed

            # Compute entropy for metrics
            with torch.no_grad():
                entropy = -(attn_weights * torch.log(attn_weights + self.config.eps))
                self._metrics.attention_entropy = entropy.sum(dim=-1).mean().item()
        else:
            attn_weights = F.softmax(attn_scores, dim=-1)

        # Check for NaN in attention weights
        if self.config.enable_nan_check and torch.isnan(attn_weights).any():
            raise NumericalError(
                "NaN detected in attention weights after softmax. "
                "This may indicate numerical overflow in attention scores."
            )

        # Apply dropout
        attn_weights = self.attn_dropout(attn_weights)

        # Compute attention output
        if self.config.enable_profiling and self._metrics is not None:
            with Timer(sync_cuda=True, unit="ms") as timer:
                attn_output = torch.matmul(attn_weights, v)
            self._metrics.av_matmul_time_ms = timer.elapsed
        else:
            attn_output = torch.matmul(attn_weights, v)

        # Reshape back: (batch, heads, seq, head_dim) -> (batch, seq, embed_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len_q, self.embed_dim)

        # Output projection
        output = self.out_proj(attn_output)
        output = self.resid_dropout(output)

        # Return weights if requested
        if need_weights:
            # Average attention weights across heads
            avg_weights = attn_weights.mean(dim=1)
            return output, avg_weights
        else:
            return output, None

    def extra_repr(self) -> str:
        """Return extra representation."""
        base_repr = super().extra_repr()
        return f"{base_repr}, causal={self.causal}"
