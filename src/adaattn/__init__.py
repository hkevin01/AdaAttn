"""
AdaAttn: Adaptive Precision & Rank Attention for GPU-Efficient Transformers.

AdaAttn is a GPU-native attention mechanism that dynamically adapts both
numerical precision and matrix rank at runtime to reduce memory bandwidth
and computational overhead in large language models.

Example:
    >>> import torch
    >>> from adaattn import AdaAttention
    >>> attn = AdaAttention(embed_dim=512, num_heads=8)
    >>> q = torch.randn(2, 128, 512)
    >>> k = torch.randn(2, 128, 512)
    >>> v = torch.randn(2, 128, 512)
    >>> output = attn(q, k, v)

"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "AdaAttn Contributors"
__license__ = "MIT"

# Lazy imports to avoid loading heavy dependencies at import time
def __getattr__(name: str):
    """Lazy loading of submodules and classes."""
    if name == "AdaAttention":
        from adaattn.attention.adaattn import AdaAttention
        return AdaAttention
    elif name == "DenseAttention":
        from adaattn.attention.dense import DenseAttention
        return DenseAttention
    elif name == "AdaptiveRankAttention":
        from adaattn.attention.adaptive_rank import AdaptiveRankAttention
        return AdaptiveRankAttention
    elif name == "AdaptivePrecisionAttention":
        from adaattn.attention.adaptive_precision import AdaptivePrecisionAttention
        return AdaptivePrecisionAttention
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "__version__",
    "AdaAttention",
    "DenseAttention",
    "AdaptiveRankAttention",
    "AdaptivePrecisionAttention",
]
