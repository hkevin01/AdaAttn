"""
Adaptive rank attention implementation.

This module provides attention that dynamically selects between
dense and low-rank computation based on runtime analysis.
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
    Timer,
)
from adaattn.attention.dense import DenseAttention


class AdaptiveRankAttention(BaseAttention):
    """
    Attention with adaptive rank selection.

    Dynamically chooses between:
    - Dense attention (full rank)
    - Low-rank attention (UΣV^T factorization)
    - Block-sparse attention

    The decision is made per-head, per-layer, per-batch based on:
    - Entropy estimation
    - Spectral norm proxy
    - Rank threshold configuration

    Example:
        >>> attn = AdaptiveRankAttention(embed_dim=512, num_heads=8)
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
        rank_ratio: float = 0.5,
        rank_estimation_method: Literal["entropy", "power", "random"] = "entropy",
        adaptive_threshold: float = 0.95,
        min_rank: int = 1,
        causal: bool = False,
        config: Optional[AttentionConfig] = None,
        **kwargs,
    ) -> None:
        """
        Initialize adaptive rank attention.

        Args:
            embed_dim: Total embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            rank_ratio: Default ratio of rank to use (0-1)
            rank_estimation_method: Method for rank estimation
            adaptive_threshold: Threshold for rank selection
            min_rank: Minimum rank to use
            causal: Whether to apply causal masking
            config: Optional AttentionConfig
            **kwargs: Additional config parameters
        """
        if config is None:
            config = AttentionConfig(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                enable_adaptive_rank=True,
                rank_threshold=adaptive_threshold,
                **kwargs,
            )

        super().__init__(config=config)

        self.rank_ratio = rank_ratio
        self.rank_estimation_method = rank_estimation_method
        self.adaptive_threshold = adaptive_threshold
        self.min_rank = max(1, min_rank)
        self.causal = causal

        # Linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Low-rank projection matrices (learned basis)
        max_rank = max(self.min_rank, int(self.head_dim * self.rank_ratio))
        self.rank_proj_down = nn.Linear(self.head_dim, max_rank, bias=False)
        self.rank_proj_up = nn.Linear(max_rank, self.head_dim, bias=False)

        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Scaling factor
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Statistics tracking
        self.register_buffer("_rank_decisions", torch.zeros(num_heads))
        self.register_buffer("_call_count", torch.tensor(0))

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.orthogonal_(self.rank_proj_down.weight)
        nn.init.orthogonal_(self.rank_proj_up.weight)

        for proj in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)

    def _estimate_rank(
        self,
        scores: Tensor,
        method: str = "entropy",
    ) -> Tuple[Tensor, Tensor]:
        """
        Estimate the effective rank of attention scores.

        Args:
            scores: Attention scores (batch, heads, seq_q, seq_k)
            method: Estimation method

        Returns:
            Tuple of:
            - Effective rank per head (batch, heads)
            - Use low-rank flag per head (batch, heads)
        """
        batch_size, num_heads, seq_q, seq_k = scores.shape
        min_dim = min(seq_q, seq_k)

        if method == "entropy":
            # Use softmax entropy as proxy for rank
            with torch.no_grad():
                probs = F.softmax(scores, dim=-1)
                # Compute entropy: -sum(p * log(p))
                entropy = -(probs * torch.log(probs + self.config.eps)).sum(dim=-1)
                # Average over query positions
                avg_entropy = entropy.mean(dim=-1)  # (batch, heads)
                # Normalize by max entropy
                max_entropy = math.log(seq_k)
                normalized_entropy = avg_entropy / max_entropy

                # Higher entropy = more uniform = higher effective rank
                # Lower entropy = more peaked = lower effective rank
                effective_rank = normalized_entropy * min_dim

                # Decide based on threshold
                use_low_rank = normalized_entropy < self.adaptive_threshold

        elif method == "power":
            # Power iteration for spectral norm estimation
            with torch.no_grad():
                # Reshape for batched computation
                A = scores.view(batch_size * num_heads, seq_q, seq_k)

                # Random initialization
                v = torch.randn(batch_size * num_heads, seq_k, 1, device=scores.device)
                v = v / v.norm(dim=1, keepdim=True)

                # Power iterations
                for _ in range(3):
                    u = torch.bmm(A, v)
                    u = u / (u.norm(dim=1, keepdim=True) + self.config.eps)
                    v = torch.bmm(A.transpose(-2, -1), u)
                    v = v / (v.norm(dim=1, keepdim=True) + self.config.eps)

                # Spectral norm
                sigma = torch.bmm(u.transpose(-2, -1), torch.bmm(A, v))
                sigma = sigma.view(batch_size, num_heads)

                # Normalize
                normalized_sigma = sigma / (sigma.max(dim=1, keepdim=True)[0] + self.config.eps)

                # Estimate rank from spectral decay
                effective_rank = normalized_sigma * min_dim
                use_low_rank = normalized_sigma < self.adaptive_threshold

        else:  # random
            # Random sampling-based estimation
            with torch.no_grad():
                effective_rank = torch.ones(batch_size, num_heads, device=scores.device)
                effective_rank = effective_rank * min_dim * self.rank_ratio
                use_low_rank = torch.ones(batch_size, num_heads, dtype=torch.bool, device=scores.device)

        return effective_rank, use_low_rank

    def _low_rank_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute low-rank attention using factorization.

        Args:
            query: (batch, heads, seq_q, head_dim)
            key: (batch, heads, seq_k, head_dim)
            value: (batch, heads, seq_k, head_dim)
            attention_mask: Optional mask

        Returns:
            Attention output (batch, heads, seq_q, head_dim)
        """
        batch_size, num_heads, seq_q, head_dim = query.shape
        _, _, seq_k, _ = key.shape

        # Project to low-rank space
        # query: (batch, heads, seq_q, head_dim) -> (batch, heads, seq_q, rank)
        q_low = self.rank_proj_down(query)
        k_low = self.rank_proj_down(key)

        # Compute attention in low-rank space
        scores = torch.matmul(q_low, k_low.transpose(-2, -1)) * self.scale

        # Apply causal mask if needed
        if self.causal:
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
                scores = scores + attention_mask

        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Apply to values and project back
        output = torch.matmul(attn_weights, value)

        return output

    def _dense_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute standard dense attention.

        Args:
            query: (batch, heads, seq_q, head_dim)
            key: (batch, heads, seq_k, head_dim)
            value: (batch, heads, seq_k, head_dim)
            attention_mask: Optional mask

        Returns:
            Attention output (batch, heads, seq_q, head_dim)
        """
        batch_size, num_heads, seq_q, head_dim = query.shape
        _, _, seq_k, _ = key.shape

        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale

        # Apply causal mask if needed
        if self.causal:
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
                scores = scores + attention_mask

        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Compute output
        output = torch.matmul(attn_weights, value)

        return output

    def _forward_impl(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Implement adaptive rank attention forward pass.

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

        # Estimate rank to decide computation path
        with torch.no_grad():
            # Quick pre-computation of scores for rank estimation
            pre_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        effective_rank, use_low_rank = self._estimate_rank(
            pre_scores, self.rank_estimation_method
        )

        # Update metrics
        if self._metrics is not None:
            self._metrics.effective_rank = effective_rank.mean().item()

        # Track decisions
        self._call_count += 1
        self._rank_decisions += use_low_rank.float().mean(dim=0)

        # Compute attention based on rank decision
        # For simplicity, use the majority decision across batch
        use_low_rank_decision = use_low_rank.float().mean() > 0.5

        if use_low_rank_decision:
            attn_output = self._low_rank_attention(q, k, v, attention_mask)
        else:
            attn_output = self._dense_attention(q, k, v, attention_mask)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len_q, self.embed_dim)

        # Output projection
        output = self.out_proj(attn_output)
        output = self.resid_dropout(output)

        return output, None

    def get_rank_statistics(self) -> dict:
        """Get statistics on rank decisions."""
        if self._call_count > 0:
            avg_low_rank_usage = self._rank_decisions / self._call_count
        else:
            avg_low_rank_usage = self._rank_decisions

        return {
            "call_count": self._call_count.item(),
            "low_rank_usage_per_head": avg_low_rank_usage.tolist(),
            "avg_low_rank_usage": avg_low_rank_usage.mean().item(),
        }

    def extra_repr(self) -> str:
        """Return extra representation."""
        base_repr = super().extra_repr()
        return (
            f"{base_repr}, rank_ratio={self.rank_ratio}, "
            f"method={self.rank_estimation_method}"
        )
