"""
Metrics utilities for AdaAttn.

Provides metrics computation for performance evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor


@dataclass
class AttentionMetrics:
    """Container for attention evaluation metrics."""

    # Performance metrics
    throughput_tokens_per_sec: float = 0.0
    latency_ms: float = 0.0
    gpu_utilization: float = 0.0

    # Memory metrics
    peak_memory_mb: float = 0.0
    average_memory_mb: float = 0.0
    memory_bandwidth_gb_s: float = 0.0

    # Quality metrics
    relative_error: float = 0.0
    cosine_similarity: float = 0.0
    max_absolute_error: float = 0.0

    # Adaptive metrics
    avg_rank_used: float = 0.0
    rank_savings_percent: float = 0.0
    precision_distribution: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "throughput_tokens_per_sec": self.throughput_tokens_per_sec,
            "latency_ms": self.latency_ms,
            "gpu_utilization": self.gpu_utilization,
            "peak_memory_mb": self.peak_memory_mb,
            "average_memory_mb": self.average_memory_mb,
            "memory_bandwidth_gb_s": self.memory_bandwidth_gb_s,
            "relative_error": self.relative_error,
            "cosine_similarity": self.cosine_similarity,
            "max_absolute_error": self.max_absolute_error,
            "avg_rank_used": self.avg_rank_used,
            "rank_savings_percent": self.rank_savings_percent,
            "precision_distribution": self.precision_distribution,
        }

    def __str__(self) -> str:
        lines = [
            "AttentionMetrics:",
            f"  Throughput: {self.throughput_tokens_per_sec:.0f} tokens/sec",
            f"  Latency: {self.latency_ms:.3f} ms",
            f"  Peak Memory: {self.peak_memory_mb:.1f} MB",
            f"  Relative Error: {self.relative_error:.6f}",
            f"  Cosine Similarity: {self.cosine_similarity:.6f}",
            f"  Avg Rank: {self.avg_rank_used:.1f}",
            f"  Rank Savings: {self.rank_savings_percent:.1f}%",
        ]
        return "\n".join(lines)


def compute_throughput(
    batch_size: int,
    seq_len: int,
    elapsed_time_ms: float,
    num_iterations: int = 1,
) -> float:
    """
    Compute token throughput.

    Args:
        batch_size: Batch size
        seq_len: Sequence length
        elapsed_time_ms: Total elapsed time in milliseconds
        num_iterations: Number of iterations

    Returns:
        Tokens processed per second
    """
    total_tokens = batch_size * seq_len * num_iterations
    elapsed_seconds = elapsed_time_ms / 1000.0
    return total_tokens / elapsed_seconds


def compute_memory_efficiency(
    seq_len: int,
    num_heads: int,
    batch_size: int,
    actual_memory_mb: float,
    dense_memory_mb: Optional[float] = None,
) -> Dict[str, float]:
    """
    Compute memory efficiency metrics.

    Args:
        seq_len: Sequence length
        num_heads: Number of attention heads
        batch_size: Batch size
        actual_memory_mb: Actual memory used
        dense_memory_mb: Memory for dense attention (computed if not provided)

    Returns:
        Dictionary with memory efficiency metrics
    """
    # Theoretical dense attention memory (attention matrix only)
    # Size: batch * num_heads * seq_len * seq_len * sizeof(float16)
    if dense_memory_mb is None:
        dense_memory_bytes = (
            batch_size * num_heads * seq_len * seq_len * 2  # FP16
        )
        dense_memory_mb = dense_memory_bytes / 1e6

    savings = 1.0 - (actual_memory_mb / (dense_memory_mb + 1e-10))

    return {
        "dense_memory_mb": dense_memory_mb,
        "actual_memory_mb": actual_memory_mb,
        "savings_mb": dense_memory_mb - actual_memory_mb,
        "savings_percent": savings * 100,
        "efficiency_ratio": dense_memory_mb / (actual_memory_mb + 1e-10),
    }


def compute_accuracy_metrics(
    output: Tensor,
    reference: Tensor,
    eps: float = 1e-10,
) -> Dict[str, float]:
    """
    Compute accuracy metrics between adaptive and reference outputs.

    Args:
        output: Adaptive attention output
        reference: Reference (dense) attention output
        eps: Small constant for numerical stability

    Returns:
        Dictionary with accuracy metrics
    """
    # Ensure same dtype for comparison
    output = output.float()
    reference = reference.float()

    # Flatten for easier computation
    output_flat = output.flatten()
    reference_flat = reference.flatten()

    # Relative error
    diff = output - reference
    relative_error = torch.norm(diff) / (torch.norm(reference) + eps)

    # Cosine similarity
    cosine_sim = torch.nn.functional.cosine_similarity(
        output_flat.unsqueeze(0),
        reference_flat.unsqueeze(0),
    ).item()

    # Max absolute error
    max_abs_error = diff.abs().max().item()

    # Mean absolute error
    mean_abs_error = diff.abs().mean().item()

    # Mean squared error
    mse = (diff ** 2).mean().item()

    # RMSE
    rmse = mse ** 0.5

    # Per-element statistics
    percentiles = torch.quantile(
        diff.abs().flatten(),
        torch.tensor([0.5, 0.9, 0.99], device=diff.device),
    )

    return {
        "relative_error": relative_error.item(),
        "cosine_similarity": cosine_sim,
        "max_absolute_error": max_abs_error,
        "mean_absolute_error": mean_abs_error,
        "mse": mse,
        "rmse": rmse,
        "median_abs_error": percentiles[0].item(),
        "p90_abs_error": percentiles[1].item(),
        "p99_abs_error": percentiles[2].item(),
    }


def compute_rank_statistics(
    ranks_used: List[int],
    max_rank: int,
) -> Dict[str, float]:
    """
    Compute statistics about rank usage.

    Args:
        ranks_used: List of ranks used across forward passes
        max_rank: Maximum possible rank (sequence length)

    Returns:
        Dictionary with rank statistics
    """
    if not ranks_used:
        return {}

    ranks = torch.tensor(ranks_used, dtype=torch.float32)

    return {
        "mean_rank": ranks.mean().item(),
        "std_rank": ranks.std().item(),
        "min_rank": ranks.min().item(),
        "max_rank": ranks.max().item(),
        "median_rank": ranks.median().item(),
        "rank_savings_percent": (1.0 - ranks.mean().item() / max_rank) * 100,
    }


def compute_flops(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    rank: Optional[int] = None,
) -> Dict[str, int]:
    """
    Compute FLOPs for attention computation.

    Args:
        batch_size: Batch size
        seq_len: Sequence length
        num_heads: Number of attention heads
        head_dim: Head dimension
        rank: Rank for low-rank attention (None for dense)

    Returns:
        Dictionary with FLOP counts
    """
    # Dense attention FLOPs:
    # QK^T: 2 * batch * heads * seq * seq * dim
    # Softmax: ~ 5 * batch * heads * seq * seq
    # AV: 2 * batch * heads * seq * seq * dim
    dense_qk = 2 * batch_size * num_heads * seq_len * seq_len * head_dim
    dense_softmax = 5 * batch_size * num_heads * seq_len * seq_len
    dense_av = 2 * batch_size * num_heads * seq_len * seq_len * head_dim
    dense_total = dense_qk + dense_softmax + dense_av

    if rank is None or rank >= seq_len:
        return {
            "qk_flops": dense_qk,
            "softmax_flops": dense_softmax,
            "av_flops": dense_av,
            "total_flops": dense_total,
            "is_dense": True,
        }

    # Low-rank attention FLOPs (approximate):
    # K projection: 2 * batch * heads * seq * rank * dim
    # V projection: 2 * batch * heads * seq * rank * dim
    # Low-rank QK^T: 2 * batch * heads * seq * rank * dim
    # Softmax: 5 * batch * heads * seq * rank
    # Low-rank AV: 2 * batch * heads * seq * rank * dim
    lr_proj = 4 * batch_size * num_heads * seq_len * rank * head_dim
    lr_qk = 2 * batch_size * num_heads * seq_len * rank * head_dim
    lr_softmax = 5 * batch_size * num_heads * seq_len * rank
    lr_av = 2 * batch_size * num_heads * seq_len * rank * head_dim
    lr_total = lr_proj + lr_qk + lr_softmax + lr_av

    return {
        "projection_flops": lr_proj,
        "qk_flops": lr_qk,
        "softmax_flops": lr_softmax,
        "av_flops": lr_av,
        "total_flops": lr_total,
        "dense_flops": dense_total,
        "flop_savings_percent": (1.0 - lr_total / dense_total) * 100,
        "is_dense": False,
    }


def compute_perplexity(
    logits: Tensor,
    labels: Tensor,
    ignore_index: int = -100,
) -> float:
    """
    Compute perplexity from model logits.

    Args:
        logits: Model logits (batch, seq_len, vocab_size)
        labels: Target labels (batch, seq_len)
        ignore_index: Index to ignore in loss computation

    Returns:
        Perplexity value
    """
    # Flatten
    logits_flat = logits.view(-1, logits.size(-1))
    labels_flat = labels.view(-1)

    # Compute cross entropy
    loss = torch.nn.functional.cross_entropy(
        logits_flat,
        labels_flat,
        ignore_index=ignore_index,
        reduction="mean",
    )

    return torch.exp(loss).item()
