"""
Attention modules for AdaAttn.

This package provides various attention implementations:
- BaseAttention: Abstract base class for all attention mechanisms
- DenseAttention: Standard scaled dot-product attention
- AdaptiveRankAttention: Attention with dynamic rank selection
- AdaptivePrecisionAttention: Attention with dynamic precision control
- AdaAttention: Combined adaptive rank and precision attention
"""

from adaattn.attention.base import BaseAttention
from adaattn.attention.dense import DenseAttention
from adaattn.attention.adaptive_rank import AdaptiveRankAttention
from adaattn.attention.adaptive_precision import AdaptivePrecisionAttention
from adaattn.attention.adaattn import AdaAttention

__all__ = [
    "BaseAttention",
    "DenseAttention",
    "AdaptiveRankAttention",
    "AdaptivePrecisionAttention",
    "AdaAttention",
]
