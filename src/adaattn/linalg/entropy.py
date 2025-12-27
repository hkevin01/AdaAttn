"""
Entropy estimation utilities for attention analysis.

This module provides efficient entropy computation for attention
weight matrices to guide adaptive rank and precision decisions.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor


def estimate_entropy(
    probs: Tensor,
    dim: int = -1,
    eps: float = 1e-10,
) -> Tensor:
    """
    Compute Shannon entropy of a probability distribution.

    Args:
        probs: Probability tensor (must sum to 1 along dim)
        dim: Dimension along which to compute entropy
        eps: Small constant for numerical stability

    Returns:
        Entropy tensor with dim reduced
    """
    # Clamp probabilities to avoid log(0)
    probs = probs.clamp(min=eps, max=1.0 - eps)

    # Shannon entropy: H = -sum(p * log(p))
    entropy = -(probs * torch.log(probs)).sum(dim=dim)

    return entropy


def normalized_entropy(
    probs: Tensor,
    dim: int = -1,
    eps: float = 1e-10,
) -> Tensor:
    """
    Compute normalized entropy (entropy / max_entropy).

    Normalized entropy is in range [0, 1]:
    - 0: Completely peaked (single element = 1)
    - 1: Completely uniform (all elements equal)

    Args:
        probs: Probability tensor
        dim: Dimension along which to compute entropy
        eps: Small constant for numerical stability

    Returns:
        Normalized entropy tensor
    """
    entropy = estimate_entropy(probs, dim=dim, eps=eps)

    # Maximum entropy for uniform distribution
    n_elements = probs.size(dim)
    max_entropy = math.log(n_elements)

    if max_entropy > 0:
        return entropy / max_entropy
    else:
        return torch.zeros_like(entropy)


def attention_entropy(
    attention_weights: Tensor,
    reduce: str = "mean",
    eps: float = 1e-10,
) -> Tensor:
    """
    Compute entropy of attention weights.

    Args:
        attention_weights: Attention weights of shape
            (batch, heads, seq_q, seq_k) or (batch, seq_q, seq_k)
        reduce: How to reduce across positions ("mean", "sum", "none")
        eps: Small constant for numerical stability

    Returns:
        Entropy tensor. Shape depends on reduce:
        - "mean"/"sum": (batch, heads) or (batch,)
        - "none": Same shape with seq_k reduced

    Raises:
        ValueError: If reduce is invalid
    """
    if reduce not in ("mean", "sum", "none"):
        raise ValueError(f"Invalid reduce: {reduce}. Must be 'mean', 'sum', or 'none'")

    # Compute entropy along key dimension (last dim)
    entropy = estimate_entropy(attention_weights, dim=-1, eps=eps)

    if reduce == "none":
        return entropy
    elif reduce == "mean":
        # Average entropy across query positions
        return entropy.mean(dim=-1)
    else:  # sum
        return entropy.sum(dim=-1)


def entropy_based_rank_hint(
    attention_weights: Tensor,
    threshold_low: float = 0.3,
    threshold_high: float = 0.8,
) -> Tensor:
    """
    Generate rank hints based on attention entropy.

    Args:
        attention_weights: Attention weights (batch, heads, seq_q, seq_k)
        threshold_low: Below this, suggest low rank
        threshold_high: Above this, suggest full rank

    Returns:
        Rank hints per head (batch, heads):
        - 0.0: Suggest low rank (peaked attention)
        - 0.5: Neutral
        - 1.0: Suggest full rank (uniform attention)
    """
    norm_entropy = normalized_entropy(attention_weights, dim=-1)
    avg_entropy = norm_entropy.mean(dim=-1)  # (batch, heads)

    # Map entropy to rank hint
    rank_hint = torch.where(
        avg_entropy < threshold_low,
        torch.zeros_like(avg_entropy),
        torch.where(
            avg_entropy > threshold_high,
            torch.ones_like(avg_entropy),
            torch.full_like(avg_entropy, 0.5),
        ),
    )

    return rank_hint
