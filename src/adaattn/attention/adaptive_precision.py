"""
Adaptive precision attention implementation.

This module provides attention that dynamically selects numerical
precision (FP32 -> BF16 -> FP16 -> FP8) based on hardware capabilities,
model quality requirements, and runtime performance analysis.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Literal, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from adaattn.attention.base import (
    AttentionConfig,
    BaseAttention,
    Timer,
    PrecisionMode,
)
from adaattn.linalg.entropy import estimate_entropy, normalized_entropy


class AdaptivePrecisionAttention(BaseAttention):
    """
    Attention with adaptive numerical precision selection.

    Dynamically chooses between:
    - FP32: Highest precision for critical computations
    - BF16: Good balance of precision and performance 
    - FP16: Faster computation, some precision loss
    - FP8: Experimental, maximum speed (if supported)

    The decision is made based on:
    - Hardware capabilities (CUDA compute capability)
    - Attention pattern analysis (entropy, gradients)
    - Quality vs speed trade-off requirements
    - Memory bandwidth constraints

    Example:
        >>> attn = AdaptivePrecisionAttention(embed_dim=512, num_heads=8)
        >>> q = torch.randn(2, 128, 512)
        >>> output, weights = attn(q, q, q)
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.0,
        precision_policy: Literal["quality", "balanced", "speed"] = "balanced",
        min_precision: str = "fp16",
        max_precision: str = "fp32",
        adaptive_threshold: float = 0.1,
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
            precision_policy: Trading off quality vs speed
            min_precision: Lowest precision to use (fp32, bf16, fp16, fp8)
            max_precision: Highest precision to use
            adaptive_threshold: Threshold for precision switching
            causal: Whether to apply causal masking
            config: Optional AttentionConfig
        """
        if config is None:
            config = AttentionConfig(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                precision=PrecisionMode.AUTO,
                **kwargs,
            )

        super().__init__(config=config)

        self.precision_policy = precision_policy
        self.min_precision = min_precision
        self.max_precision = max_precision
        self.adaptive_threshold = adaptive_threshold
        self.causal = causal

        # Precision hierarchy (lower index = higher precision)
        self.precision_levels = ["fp32", "bf16", "fp16", "fp8"]
        self.min_level = self.precision_levels.index(max_precision)
        self.max_level = self.precision_levels.index(min_precision)

        # Hardware capabilities
        self.cuda_capability = self._detect_cuda_capability()
        self.supports_bf16 = self._supports_bfloat16()
        self.supports_fp8 = self._supports_fp8()

        # Linear projections (Q, K, V, and output)
        self.q_proj = nn.Linear(self.config.embed_dim, self.config.embed_dim, bias=False)
        self.k_proj = nn.Linear(self.config.embed_dim, self.config.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.config.embed_dim, self.config.embed_dim, bias=False)
        self.out_proj = nn.Linear(self.config.embed_dim, self.config.embed_dim, bias=False)

        # Dropout layers  
        self.attn_dropout = nn.Dropout(self.config.dropout)
        self.resid_dropout = nn.Dropout(self.config.dropout)

        # Statistics tracking
        self._precision_usage = torch.zeros(len(self.precision_levels))
        self._call_count = 0
        self._quality_loss = 0.0

        # Scaling factor 
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def _detect_cuda_capability(self) -> Tuple[int, int]:
        """Detect CUDA compute capability."""
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            return cap
        return (0, 0)

    def _supports_bfloat16(self) -> bool:
        """Check if hardware supports efficient BF16."""
        if not torch.cuda.is_available():
            return False
        
        # BF16 well supported on Ampere+ (8.x+) and some Turing (7.5+)
        major, minor = self.cuda_capability
        return (major > 7) or (major == 7 and minor >= 5)

    def _supports_fp8(self) -> bool:
        """Check if hardware supports FP8.""" 
        if not torch.cuda.is_available():
            return False
        
        # FP8 requires Hopper (9.x+) or newer
        major, _ = self.cuda_capability
        return major >= 9

    def _select_precision(
        self, 
        q: Tensor, 
        k: Tensor, 
        v: Tensor,
        attention_mask: Optional[Tensor] = None
    ) -> str:
        """
        Select optimal precision based on input analysis and policy.
        
        Args:
            q, k, v: Input tensors
            attention_mask: Optional attention mask
            
        Returns:
            Precision string (fp32, bf16, fp16, fp8)
        """
        batch_size, seq_len, _ = q.shape
        
        # Start with policy-based baseline
        if self.precision_policy == "quality":
            base_level = self.min_level  # Highest precision
        elif self.precision_policy == "speed":
            base_level = self.max_level  # Lowest precision
        else:  # balanced
            base_level = (self.min_level + self.max_level) // 2

        # Adjust based on input characteristics
        with torch.no_grad():
            # Check for extreme values that need high precision
            q_max = q.abs().max()
            k_max = k.abs().max()
            v_max = v.abs().max()
            max_val = max(q_max, k_max, v_max)
            
            # Very large values need higher precision
            if max_val > 10.0:
                base_level = max(0, base_level - 1)
            
            # Check gradient requirements (if in training)
            if self.training:
                # Check if any gradients are very small (need precision)
                if hasattr(q, 'grad') and q.grad is not None:
                    grad_min = q.grad.abs().min()
                    if grad_min < 1e-5:
                        base_level = max(0, base_level - 1)
            
            # Sequence length considerations
            if seq_len > 2048:
                # Long sequences benefit from lower precision for speed
                base_level = min(self.max_level, base_level + 1)
            
            # Memory pressure considerations
            memory_usage = batch_size * seq_len * seq_len * 4  # Rough estimate
            if torch.cuda.is_available():
                available_memory = torch.cuda.get_device_properties(0).total_memory
                if memory_usage > available_memory * 0.7:  # High memory pressure
                    base_level = min(self.max_level, base_level + 1)

        # Hardware capability constraints
        target_precision = self.precision_levels[base_level]
        
        if target_precision == "bf16" and not self.supports_bf16:
            if base_level < self.max_level:
                target_precision = "fp16"
            else:
                target_precision = "fp32"
                
        elif target_precision == "fp8" and not self.supports_fp8:
            if self.supports_bf16:
                target_precision = "bf16" 
            else:
                target_precision = "fp16"

        return target_precision

    def _convert_precision(self, tensor: Tensor, precision: str) -> Tensor:
        """Convert tensor to specified precision."""
        if precision == "fp32":
            return tensor.float()
        elif precision == "bf16":
            return tensor.to(torch.bfloat16) if self.supports_bf16 else tensor.half()
        elif precision == "fp16":
            return tensor.half()
        elif precision == "fp8":
            # FP8 not directly supported, use FP16 as fallback
            return tensor.half()
        else:
            return tensor

    def _forward_impl(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass with adaptive precision.

        Args:
            query: Query tensor (batch_size, seq_len, embed_dim)
            key: Key tensor (batch_size, seq_len, embed_dim)
            value: Value tensor (batch_size, seq_len, embed_dim)
            attention_mask: Optional attention mask
            need_weights: Whether to return attention weights

        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_len_q, embed_dim = query.shape
        seq_len_k = key.size(1)

        # Select optimal precision
        target_precision = self._select_precision(query, key, value, attention_mask)
        
        # Track precision usage
        self._call_count += 1
        precision_idx = self.precision_levels.index(target_precision)
        self._precision_usage[precision_idx] += 1

        # Store original precision for output
        original_dtype = query.dtype

        # Convert inputs to target precision
        with Timer(f"precision_conversion_{target_precision}"):
            q = self._convert_precision(self.q_proj(query), target_precision)
            k = self._convert_precision(self.k_proj(key), target_precision) 
            v = self._convert_precision(self.v_proj(value), target_precision)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len_q, self.config.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len_k, self.config.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len_k, self.config.num_heads, self.head_dim)

        # Transpose for attention computation
        q = q.transpose(1, 2)  # (batch, heads, seq_q, head_dim)
        k = k.transpose(1, 2)  # (batch, heads, seq_k, head_dim)
        v = v.transpose(1, 2)  # (batch, heads, seq_k, head_dim)

        # Compute attention in target precision
        with Timer(f"attention_computation_{target_precision}"):
            attn_output = self._compute_attention(q, k, v, attention_mask)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len_q, embed_dim)

        # Convert back to original precision for output projection
        if target_precision != str(original_dtype).replace('torch.', ''):
            attn_output = attn_output.to(original_dtype)

        # Output projection
        output = self.out_proj(attn_output)
        output = self.resid_dropout(output)

        return output, None

    def _compute_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute attention with current precision.
        
        Args:
            q, k, v: Query, key, value tensors in target precision
            attention_mask: Optional attention mask
            
        Returns:
            Attention output tensor
        """
        batch_size, num_heads, seq_q, head_dim = q.shape
        _, _, seq_k, _ = k.shape

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply causal mask if needed
        if self.causal:
            causal_mask = torch.triu(
                torch.ones(seq_q, seq_k, dtype=torch.bool, device=scores.device),
                diagonal=1,
            )
            scores = scores.masked_fill(causal_mask, float('-inf'))

        # Apply attention mask if provided
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
            
            scores = scores + attention_mask

        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Apply to values
        output = torch.matmul(attn_weights, v)

        return output

    def get_precision_statistics(self) -> dict:
        """Get statistics on precision usage."""
        if self._call_count > 0:
            usage_percentages = (self._precision_usage / self._call_count * 100).tolist()
        else:
            usage_percentages = [0.0] * len(self.precision_levels)

        return {
            "call_count": self._call_count,
            "precision_usage": dict(zip(self.precision_levels, usage_percentages)),
            "hardware_support": {
                "cuda_capability": self.cuda_capability,
                "supports_bf16": self.supports_bf16,
                "supports_fp8": self.supports_fp8,
            },
        }

    def set_precision_policy(self, policy: str) -> None:
        """Update precision policy at runtime."""
        if policy not in ["quality", "balanced", "speed"]:
            raise ValueError(f"Invalid policy: {policy}")
        self.precision_policy = policy

    def extra_repr(self) -> str:
        """Return extra representation.""" 
        base_repr = super().extra_repr()
        return (
            f"{base_repr}, precision_policy={self.precision_policy}, "
            f"range={self.max_precision}-{self.min_precision}"
        )
