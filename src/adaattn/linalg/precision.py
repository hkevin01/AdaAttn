"""
Precision control utilities.

This module provides utilities for adaptive numerical precision
control in attention computations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Union

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


class PrecisionLevel(Enum):
    """Supported precision levels."""

    FP32 = "fp32"
    TF32 = "tf32"
    BF16 = "bf16"
    FP16 = "fp16"
    FP8 = "fp8"

    @property
    def torch_dtype(self) -> torch.dtype:
        """Convert to PyTorch dtype."""
        mapping = {
            PrecisionLevel.FP32: torch.float32,
            PrecisionLevel.TF32: torch.float32,
            PrecisionLevel.BF16: torch.bfloat16,
            PrecisionLevel.FP16: torch.float16,
            PrecisionLevel.FP8: torch.float16,  # FP8 not natively supported
        }
        return mapping[self]

    @property
    def bits(self) -> int:
        """Number of bits for this precision."""
        mapping = {
            PrecisionLevel.FP32: 32,
            PrecisionLevel.TF32: 19,  # Effective precision
            PrecisionLevel.BF16: 16,
            PrecisionLevel.FP16: 16,
            PrecisionLevel.FP8: 8,
        }
        return mapping[self]

    @property
    def mantissa_bits(self) -> int:
        """Number of mantissa bits."""
        mapping = {
            PrecisionLevel.FP32: 23,
            PrecisionLevel.TF32: 10,
            PrecisionLevel.BF16: 7,
            PrecisionLevel.FP16: 10,
            PrecisionLevel.FP8: 3,  # E4M3 format
        }
        return mapping[self]


@dataclass
class PrecisionConfig:
    """Configuration for adaptive precision control."""

    min_precision: PrecisionLevel = PrecisionLevel.FP16
    max_precision: PrecisionLevel = PrecisionLevel.FP32
    qk_precision: PrecisionLevel = PrecisionLevel.FP16
    softmax_precision: PrecisionLevel = PrecisionLevel.FP32
    av_precision: PrecisionLevel = PrecisionLevel.FP16
    accumulator_precision: PrecisionLevel = PrecisionLevel.FP32
    entropy_threshold_high: float = 0.9
    entropy_threshold_low: float = 0.3
    magnitude_threshold: float = 1e4
    enable_adaptive: bool = True


def compute_precision_requirements(
    qk_scores: Tensor,
    config: PrecisionConfig,
) -> Tuple[PrecisionLevel, PrecisionLevel]:
    """
    Determine precision requirements based on QK score characteristics.

    Args:
        qk_scores: QK^T scores of shape (..., seq_len, seq_len)
        config: Precision configuration

    Returns:
        Tuple of (compute_precision, accumulator_precision)
    """
    if not config.enable_adaptive:
        return config.qk_precision, config.accumulator_precision

    # Check magnitude range
    max_val = qk_scores.abs().max().item()
    min_nonzero = qk_scores[qk_scores != 0].abs().min().item() if (qk_scores != 0).any() else 1.0

    dynamic_range = max_val / (min_nonzero + 1e-10)

    # High dynamic range requires higher precision
    if dynamic_range > config.magnitude_threshold:
        logger.debug(f"High dynamic range detected: {dynamic_range:.2e}")
        return PrecisionLevel.FP32, PrecisionLevel.FP32

    # Check for potential overflow in softmax
    if max_val > 80:  # exp(80) is near FP16 max
        logger.debug(f"Potential softmax overflow detected: max={max_val:.2f}")
        return PrecisionLevel.FP32, PrecisionLevel.FP32

    # Otherwise use configured lower precision
    return config.qk_precision, config.accumulator_precision


def compute_error_bound(
    A: Tensor,
    precision: PrecisionLevel,
) -> float:
    """
    Compute theoretical error bound for matrix operations at given precision.

    Uses the formula: ||error|| ≤ n * eps * ||A|| where eps is machine epsilon.

    Args:
        A: Input matrix
        precision: Target precision level

    Returns:
        Upper bound on relative error
    """
    # Machine epsilon for different precisions
    eps_values = {
        PrecisionLevel.FP32: 1.19e-7,
        PrecisionLevel.TF32: 9.77e-4,
        PrecisionLevel.BF16: 3.91e-3,
        PrecisionLevel.FP16: 9.77e-4,
        PrecisionLevel.FP8: 6.25e-2,
    }

    eps = eps_values[precision]
    n = max(A.shape[-2:])

    # Frobenius norm
    norm_A = torch.linalg.norm(A.float()).item()

    return n * eps * norm_A


def safe_cast(
    tensor: Tensor,
    target_precision: PrecisionLevel,
    check_overflow: bool = True,
) -> Tuple[Tensor, bool]:
    """
    Safely cast tensor to target precision with overflow checking.

    Args:
        tensor: Input tensor
        target_precision: Target precision level
        check_overflow: Whether to check for potential overflow

    Returns:
        Tuple of (casted_tensor, overflow_detected)
    """
    target_dtype = target_precision.torch_dtype
    overflow_detected = False

    if check_overflow and target_precision in [PrecisionLevel.FP16, PrecisionLevel.BF16]:
        # Check if values exceed representable range
        max_val = tensor.abs().max().item()

        if target_precision == PrecisionLevel.FP16:
            max_representable = 65504.0
        else:  # BF16
            max_representable = 3.4e38

        if max_val > max_representable:
            overflow_detected = True
            logger.warning(
                f"Overflow detected: max={max_val:.2e}, "
                f"representable={max_representable:.2e}"
            )
            # Scale down to prevent overflow
            scale = max_representable / (max_val + 1e-10)
            tensor = tensor * scale

    return tensor.to(target_dtype), overflow_detected


def mixed_precision_matmul(
    A: Tensor,
    B: Tensor,
    compute_precision: PrecisionLevel = PrecisionLevel.FP16,
    accumulator_precision: PrecisionLevel = PrecisionLevel.FP32,
) -> Tensor:
    """
    Perform matrix multiplication with mixed precision.

    Args:
        A: First matrix
        B: Second matrix
        compute_precision: Precision for the computation
        accumulator_precision: Precision for accumulation

    Returns:
        Result matrix in accumulator precision
    """
    # Cast inputs to compute precision
    A_cast, _ = safe_cast(A, compute_precision, check_overflow=True)
    B_cast, _ = safe_cast(B, compute_precision, check_overflow=True)

    # Perform matmul
    if compute_precision in [PrecisionLevel.FP16, PrecisionLevel.BF16]:
        # Use autocast for potential hardware acceleration
        with torch.amp.autocast(device_type='cuda', dtype=compute_precision.torch_dtype):
            result = torch.matmul(A_cast, B_cast)
    else:
        result = torch.matmul(A_cast, B_cast)

    # Cast result to accumulator precision
    return result.to(accumulator_precision.torch_dtype)


def analyze_numerical_stability(
    scores: Tensor,
    attention_weights: Tensor,
) -> dict:
    """
    Analyze numerical stability of attention computation.

    Args:
        scores: Raw QK^T scores
        attention_weights: Softmax output

    Returns:
        Dictionary with stability metrics
    """
    metrics = {}

    # Score statistics
    metrics["score_max"] = scores.max().item()
    metrics["score_min"] = scores.min().item()
    metrics["score_range"] = metrics["score_max"] - metrics["score_min"]
    metrics["score_std"] = scores.std().item()

    # Attention weight statistics
    metrics["attn_max"] = attention_weights.max().item()
    metrics["attn_min"] = attention_weights.min().item()
    metrics["attn_entropy"] = -(
        attention_weights * torch.log(attention_weights + 1e-10)
    ).sum(dim=-1).mean().item()

    # Check for numerical issues
    metrics["has_nan"] = bool(torch.isnan(attention_weights).any())
    metrics["has_inf"] = bool(torch.isinf(scores).any())
    metrics["near_zero_count"] = int((attention_weights < 1e-10).sum())
    metrics["saturated_count"] = int((attention_weights > 0.99).sum())

    # Stability score (0-1, higher is better)
    stability = 1.0
    if metrics["has_nan"] or metrics["has_inf"]:
        stability = 0.0
    elif metrics["score_range"] > 100:
        stability -= 0.3
    if metrics["saturated_count"] / attention_weights.numel() > 0.1:
        stability -= 0.2
    metrics["stability_score"] = max(0.0, stability)

    return metrics
