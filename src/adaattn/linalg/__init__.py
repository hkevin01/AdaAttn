"""
Linear algebra utilities for AdaAttn.

This module provides efficient linear algebra operations
optimized for attention computation.
"""

from adaattn.linalg.entropy import (
    attention_entropy,
    entropy_based_rank,
    normalized_entropy,
    softmax_entropy,
)
from adaattn.linalg.low_rank import (
    estimate_effective_rank,
    low_rank_approx,
    power_iteration,
    randomized_svd,
    spectral_norm,
)
from adaattn.linalg.precision import (
    PrecisionConfig,
    PrecisionLevel,
    analyze_numerical_stability,
    compute_error_bound,
    compute_precision_requirements,
    mixed_precision_matmul,
    safe_cast,
)
from adaattn.linalg.utils import (
    batch_frobenius_norm,
    batch_trace,
    causal_mask,
    condition_number_estimate,
    log_softmax_stable,
    masked_fill_inf,
    sliding_window_mask,
    stable_softmax,
    tile_matrix,
    untile_matrix,
)

__all__ = [
    # Entropy
    "attention_entropy",
    "entropy_based_rank",
    "normalized_entropy",
    "softmax_entropy",
    # Low-rank
    "estimate_effective_rank",
    "low_rank_approx",
    "power_iteration",
    "randomized_svd",
    "spectral_norm",
    # Precision
    "PrecisionConfig",
    "PrecisionLevel",
    "analyze_numerical_stability",
    "compute_error_bound",
    "compute_precision_requirements",
    "mixed_precision_matmul",
    "safe_cast",
    # Utils
    "batch_frobenius_norm",
    "batch_trace",
    "causal_mask",
    "condition_number_estimate",
    "log_softmax_stable",
    "masked_fill_inf",
    "sliding_window_mask",
    "stable_softmax",
    "tile_matrix",
    "untile_matrix",
]
