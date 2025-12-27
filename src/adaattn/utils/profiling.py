"""
Profiling utilities for AdaAttn.

Provides tools for measuring performance and memory usage.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch
from torch import Tensor


@dataclass
class ProfileResult:
    """Result of a profiling run."""

    name: str
    elapsed_time_ms: float
    memory_allocated_mb: float
    memory_reserved_mb: float
    peak_memory_mb: float
    iterations: int = 1

    @property
    def time_per_iter_ms(self) -> float:
        return self.elapsed_time_ms / self.iterations

    def __repr__(self) -> str:
        return (
            f"ProfileResult({self.name}: "
            f"{self.time_per_iter_ms:.3f}ms/iter, "
            f"peak_mem={self.peak_memory_mb:.1f}MB)"
        )


class CUDAProfiler:
    """CUDA profiler for measuring GPU performance."""

    def __init__(self, warmup_iters: int = 3, profile_iters: int = 10):
        self.warmup_iters = warmup_iters
        self.profile_iters = profile_iters
        self.results: Dict[str, ProfileResult] = {}

    def profile(
        self,
        func: Callable,
        name: str = "unnamed",
        *args,
        **kwargs,
    ) -> ProfileResult:
        """
        Profile a function on CUDA.

        Args:
            func: Function to profile
            name: Name for the profiled operation
            *args, **kwargs: Arguments to pass to func

        Returns:
            ProfileResult with timing and memory statistics
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for profiling")

        device = torch.device("cuda")

        # Warmup
        for _ in range(self.warmup_iters):
            func(*args, **kwargs)
            torch.cuda.synchronize()

        # Reset memory stats
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

        # Profile
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(self.profile_iters):
            func(*args, **kwargs)
        end_event.record()

        torch.cuda.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)

        # Memory stats
        memory_allocated = torch.cuda.memory_allocated(device) / 1e6
        memory_reserved = torch.cuda.memory_reserved(device) / 1e6
        peak_memory = torch.cuda.max_memory_allocated(device) / 1e6

        result = ProfileResult(
            name=name,
            elapsed_time_ms=elapsed_ms,
            memory_allocated_mb=memory_allocated,
            memory_reserved_mb=memory_reserved,
            peak_memory_mb=peak_memory,
            iterations=self.profile_iters,
        )

        self.results[name] = result
        return result

    def compare(self, *names: str) -> str:
        """Compare profiled results by name."""
        if not names:
            names = tuple(self.results.keys())

        lines = ["Profiling Comparison:", "-" * 60]

        for name in names:
            if name in self.results:
                r = self.results[name]
                lines.append(
                    f"{name:30s} | {r.time_per_iter_ms:8.3f} ms | "
                    f"{r.peak_memory_mb:8.1f} MB"
                )

        return "\n".join(lines)


class MemoryTracker:
    """Track GPU memory usage over time."""

    def __init__(self):
        self.snapshots: List[Dict] = []
        self._tracking = False

    def start(self):
        """Start tracking memory."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self._tracking = True
            self.snapshot("start")

    def stop(self) -> Dict:
        """Stop tracking and return summary."""
        self._tracking = False
        self.snapshot("stop")
        return self.get_summary()

    def snapshot(self, label: str = ""):
        """Take a memory snapshot."""
        if not torch.cuda.is_available():
            return

        self.snapshots.append({
            "label": label,
            "timestamp": time.time(),
            "allocated_mb": torch.cuda.memory_allocated() / 1e6,
            "reserved_mb": torch.cuda.memory_reserved() / 1e6,
            "peak_mb": torch.cuda.max_memory_allocated() / 1e6,
        })

    def get_summary(self) -> Dict:
        """Get memory usage summary."""
        if not self.snapshots:
            return {}

        return {
            "num_snapshots": len(self.snapshots),
            "peak_memory_mb": max(s["peak_mb"] for s in self.snapshots),
            "final_allocated_mb": self.snapshots[-1]["allocated_mb"],
            "snapshots": self.snapshots,
        }

    def clear(self):
        """Clear all snapshots."""
        self.snapshots = []


@contextmanager
def profile_attention(name: str = "attention"):
    """Context manager for profiling attention operations."""
    if not torch.cuda.is_available():
        yield {}
        return

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    try:
        yield {}
    finally:
        end.record()
        torch.cuda.synchronize()

        elapsed_ms = start.elapsed_time(end)
        peak_memory = torch.cuda.max_memory_allocated() / 1e6

        print(f"[{name}] Time: {elapsed_ms:.3f}ms, Peak Memory: {peak_memory:.1f}MB")


def benchmark_attention(
    attention_fn: Callable,
    batch_size: int = 4,
    seq_len: int = 512,
    num_heads: int = 8,
    head_dim: int = 64,
    warmup: int = 3,
    iterations: int = 10,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
) -> ProfileResult:
    """
    Benchmark an attention function.

    Args:
        attention_fn: Attention function to benchmark
        batch_size: Batch size
        seq_len: Sequence length
        num_heads: Number of attention heads
        head_dim: Head dimension
        warmup: Number of warmup iterations
        iterations: Number of benchmark iterations
        dtype: Data type
        device: Device to use

    Returns:
        ProfileResult with benchmark results
    """
    device = torch.device(device)

    # Create input tensors
    q = torch.randn(
        batch_size, seq_len, num_heads, head_dim,
        dtype=dtype, device=device
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    profiler = CUDAProfiler(warmup_iters=warmup, profile_iters=iterations)

    def run_attention():
        return attention_fn(q, k, v)

    result = profiler.profile(
        run_attention,
        name=f"attention_b{batch_size}_s{seq_len}_h{num_heads}_d{head_dim}",
    )

    # Add throughput metrics
    total_tokens = batch_size * seq_len * iterations
    tokens_per_second = total_tokens / (result.elapsed_time_ms / 1000)
    result.tokens_per_second = tokens_per_second

    return result
