"""
GPU Kernels for AdaAttn.

This module provides CUDA kernels for efficient attention computation.
Currently contains Python fallback implementations, with CUDA kernels
to be compiled via setup.py when available.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

# Try to import compiled CUDA extensions
_CUDA_AVAILABLE = False
try:
    from adaattn.kernels import _adaattn_cuda
    _CUDA_AVAILABLE = True
    logger.info("AdaAttn CUDA kernels loaded successfully")
except ImportError:
    logger.warning(
        "AdaAttn CUDA kernels not available. "
        "Using PyTorch fallback implementations."
    )


def is_cuda_available() -> bool:
    """Check if CUDA kernels are available."""
    return _CUDA_AVAILABLE


def flash_attention_forward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> Tensor:
    """
    FlashAttention-style forward pass.

    Args:
        q: Query tensor (batch, seq_len, num_heads, head_dim)
        k: Key tensor (batch, seq_len, num_heads, head_dim)
        v: Value tensor (batch, seq_len, num_heads, head_dim)
        softmax_scale: Scaling factor for softmax
        causal: Whether to use causal masking

    Returns:
        Output tensor (batch, seq_len, num_heads, head_dim)
    """
    if _CUDA_AVAILABLE and q.is_cuda:
        return _adaattn_cuda.flash_attention_forward(
            q, k, v, softmax_scale, causal
        )

    # PyTorch fallback
    return _flash_attention_forward_pytorch(q, k, v, softmax_scale, causal)


def _flash_attention_forward_pytorch(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> Tensor:
    """PyTorch fallback for flash attention."""
    batch, seq_len, num_heads, head_dim = q.shape

    if softmax_scale is None:
        softmax_scale = head_dim ** -0.5

    # Transpose for batch matmul: (batch, num_heads, seq_len, head_dim)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    # Compute attention scores
    scores = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale

    # Apply causal mask if needed
    if causal:
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool),
            diagonal=1
        )
        scores = scores.masked_fill(mask, float('-inf'))

    # Softmax
    attn_weights = torch.softmax(scores, dim=-1)

    # Apply attention to values
    output = torch.matmul(attn_weights, v)

    # Transpose back: (batch, seq_len, num_heads, head_dim)
    return output.transpose(1, 2)


def adaptive_attention_forward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    target_rank: Optional[int] = None,
    precision: str = "fp16",
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> Tensor:
    """
    Adaptive attention forward pass with rank and precision control.

    Args:
        q: Query tensor (batch, seq_len, num_heads, head_dim)
        k: Key tensor (batch, seq_len, num_heads, head_dim)
        v: Value tensor (batch, seq_len, num_heads, head_dim)
        target_rank: Target rank for low-rank approximation (None = dense)
        precision: Computation precision ("fp32", "fp16", "bf16")
        softmax_scale: Scaling factor for softmax
        causal: Whether to use causal masking

    Returns:
        Output tensor (batch, seq_len, num_heads, head_dim)
    """
    if _CUDA_AVAILABLE and q.is_cuda:
        return _adaattn_cuda.adaptive_attention_forward(
            q, k, v, target_rank, precision, softmax_scale, causal
        )

    # PyTorch fallback
    return _adaptive_attention_forward_pytorch(
        q, k, v, target_rank, precision, softmax_scale, causal
    )


def _adaptive_attention_forward_pytorch(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    target_rank: Optional[int] = None,
    precision: str = "fp16",
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> Tensor:
    """PyTorch fallback for adaptive attention."""
    original_dtype = q.dtype
    batch, seq_len, num_heads, head_dim = q.shape

    # Set precision
    if precision == "fp16":
        compute_dtype = torch.float16
    elif precision == "bf16":
        compute_dtype = torch.bfloat16
    else:
        compute_dtype = torch.float32

    # Cast to compute precision
    q = q.to(compute_dtype)
    k = k.to(compute_dtype)
    v = v.to(compute_dtype)

    if softmax_scale is None:
        softmax_scale = head_dim ** -0.5

    # Transpose for batch matmul
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    # Apply low-rank approximation if requested
    if target_rank is not None and target_rank < seq_len:
        # Low-rank attention via randomized projection
        output = _low_rank_attention(q, k, v, target_rank, softmax_scale, causal)
    else:
        # Dense attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale

        if causal:
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool),
                diagonal=1
            )
            scores = scores.masked_fill(mask, float('-inf'))

        # Use FP32 for softmax stability
        attn_weights = torch.softmax(scores.float(), dim=-1).to(compute_dtype)
        output = torch.matmul(attn_weights, v)

    # Transpose back and cast to original dtype
    return output.transpose(1, 2).to(original_dtype)


def _low_rank_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    rank: int,
    softmax_scale: float,
    causal: bool,
) -> Tensor:
    """Low-rank attention approximation."""
    batch, num_heads, seq_len, head_dim = q.shape

    # Random projection for low-rank approximation
    # Project K to lower dimension
    projection = torch.randn(
        seq_len, rank,
        device=q.device,
        dtype=q.dtype
    ) / (rank ** 0.5)

    # K_proj: (batch, num_heads, rank, head_dim)
    k_proj = torch.matmul(projection.T, k)

    # Compute low-rank scores
    # scores_low: (batch, num_heads, seq_len, rank)
    scores_low = torch.matmul(q, k_proj.transpose(-2, -1)) * softmax_scale

    # Approximate full attention
    # Reconstruct approximate attention weights
    attn_low = torch.softmax(scores_low.float(), dim=-1).to(q.dtype)

    # Project V similarly
    v_proj = torch.matmul(projection.T, v)

    # Output via low-rank factorization
    output = torch.matmul(attn_low, v_proj)

    return output


__all__ = [
    "is_cuda_available",
    "flash_attention_forward",
    "adaptive_attention_forward",
]
