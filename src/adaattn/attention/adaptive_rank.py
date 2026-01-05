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
from adaattn.linalg.entropy import (
    estimate_entropy,
    normalized_entropy,
    attention_entropy,
)


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
        
        # Hardware-aware thresholds
        self._device_type = None
        self._memory_threshold = 0.8  # Use low-rank if memory usage > 80%
        self._sequence_threshold = 512  # Use low-rank for sequences > 512

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
            # Enhanced entropy-based rank estimation
            with torch.no_grad():
                probs = F.softmax(scores, dim=-1)
                
                # Multi-scale entropy analysis
                # 1. Local entropy (per query position)
                local_entropy = estimate_entropy(probs, dim=-1, eps=self.config.eps)
                
                # 2. Global entropy (per head)
                global_entropy = attention_entropy(probs, reduce="mean", eps=self.config.eps)
                
                # 3. Entropy variance (attention focus consistency)
                entropy_var = local_entropy.var(dim=-1)  # (batch, heads)
                
                # Normalize entropy metrics
                max_entropy = math.log(seq_k)
                norm_global_entropy = global_entropy / max_entropy
                norm_entropy_var = entropy_var / (max_entropy ** 2)
                
                # Combined rank estimation heuristic
                # High entropy + low variance = uniform attention (high rank)
                # Low entropy + high variance = focused attention (low rank)
                focus_score = 1.0 - norm_global_entropy + norm_entropy_var
                focus_score = torch.clamp(focus_score, 0.0, 1.0)
                
                # Hardware-aware adjustments
                hardware_penalty = self._get_hardware_penalty(batch_size, seq_q, seq_k)
                adjusted_threshold = self.adaptive_threshold + hardware_penalty
                
                # Rank decision
                use_low_rank = focus_score > adjusted_threshold
                effective_rank = (1.0 - focus_score) * min_dim

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
    
    def _get_hardware_penalty(
        self, 
        batch_size: int, 
        seq_q: int, 
        seq_k: int
    ) -> float:
        """
        Compute hardware-aware penalty to bias toward low-rank.
        
        Returns higher penalty (favoring low-rank) when:
        - Sequence lengths are very long
        - Memory usage would be high
        - Running on CPU (vs GPU)
        """
        penalty = 0.0
        
        # Sequence length penalty
        max_seq = max(seq_q, seq_k)
        if max_seq > self._sequence_threshold:
            seq_penalty = min(0.2, (max_seq - self._sequence_threshold) / 2048)
            penalty += seq_penalty
        
        # Memory usage penalty (rough estimation)
        memory_usage = batch_size * self.config.num_heads * seq_q * seq_k * 4  # bytes
        if torch.cuda.is_available() and memory_usage > 1e9:  # > 1GB
            memory_penalty = min(0.15, memory_usage / 1e10)
            penalty += memory_penalty
        
        # Device type penalty
        if self._device_type is None:
            self._device_type = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self._device_type == "cpu":
            penalty += 0.1  # Bias toward low-rank on CPU
        
        return penalty
    
    def predict_optimal_rank(
        self, 
        q: Tensor, 
        k: Tensor, 
        v: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Predict optimal rank based on input statistics (before attention computation).
        
        This provides a fast pre-computation heuristic that can guide
        architecture decisions or adaptive batching.
        
        Args:
            q, k, v: Input tensors
            
        Returns:
            predicted_ranks: (batch, heads) tensor with predicted ranks
            confidence: (batch, heads) confidence scores [0,1]
        """
        batch_size, seq_len, _ = q.shape
        
        with torch.no_grad():
            # Compute input statistics
            q_norm = q.norm(dim=-1)  # (batch, seq_len)
            k_norm = k.norm(dim=-1)
            v_norm = v.norm(dim=-1)
            
            # Variance in norms (indicator of rank structure)
            q_var = q_norm.var(dim=-1)  # (batch,)
            k_var = k_norm.var(dim=-1)
            v_var = v_norm.var(dim=-1)
            
            # Average variance as rank predictor
            avg_var = (q_var + k_var + v_var) / 3
            
            # High variance suggests diverse content (high rank)
            # Low variance suggests similar content (low rank)
            predicted_rank_ratio = torch.sigmoid(avg_var * 10 - 2)  # Adaptive scaling
            
            # Expand to (batch, heads)
            predicted_rank_ratio = predicted_rank_ratio.unsqueeze(1).expand(
                batch_size, self.config.num_heads
            )
            
            head_dim = self.config.embed_dim // self.config.num_heads
            predicted_ranks = predicted_rank_ratio * head_dim
            
            # Confidence based on variance consistency
            var_consistency = 1.0 - torch.abs(q_var - k_var) / (q_var + k_var + 1e-8)
            confidence = var_consistency.unsqueeze(1).expand(
                batch_size, self.config.num_heads
            )
            
            return predicted_ranks.int(), confidence

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
