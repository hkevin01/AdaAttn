"""
Utility modules for AdaAttn.

Provides profiling, logging, and metrics utilities.
"""

from adaattn.utils.profiling import (
    CUDAProfiler,
    MemoryTracker,
    profile_attention,
    benchmark_attention,
)
from adaattn.utils.logging import (
    setup_logging,
    get_logger,
    AttentionLogger,
)
from adaattn.utils.metrics import (
    compute_throughput,
    compute_memory_efficiency,
    compute_accuracy_metrics,
    AttentionMetrics,
)

__all__ = [
    # Profiling
    "CUDAProfiler",
    "MemoryTracker",
    "profile_attention",
    "benchmark_attention",
    # Logging
    "setup_logging",
    "get_logger",
    "AttentionLogger",
    # Metrics
    "compute_throughput",
    "compute_memory_efficiency",
    "compute_accuracy_metrics",
    "AttentionMetrics",
]
