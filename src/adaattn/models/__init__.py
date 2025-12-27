"""
Model implementations using AdaAttn.

This module provides transformer models that use adaptive attention
as a drop-in replacement for standard attention.
"""

from adaattn.models.transformer import (
    AdaAttnTransformer,
    AdaAttnTransformerConfig,
    AdaAttnTransformerLayer,
)
from adaattn.models.gpt_wrapper import (
    GPTWithAdaAttn,
    replace_attention_with_adaattn,
)

__all__ = [
    "AdaAttnTransformer",
    "AdaAttnTransformerConfig",
    "AdaAttnTransformerLayer",
    "GPTWithAdaAttn",
    "replace_attention_with_adaattn",
]
