"""
Benchmark script for comparing attention implementations.

This script benchmarks:
- PyTorch baseline attention
- FlashAttention v2 (if available)
- Adaptive precision attention
- Adaptive rank attention
- Full AdaAttn implementation

Measures:
- Forward pass time
- Backward pass time
- Memory usage
- Throughput (tokens/sec)
- Numerical accuracy
"""

import argparse
import time
from typing import Dict, List, Tuple, Optional
import warnings

import torch
import torch.nn.functional as F
import numpy as np

from benchmarks.pytorch_attention.baseline import PyTorchAttentionBaseline
from benchmarks.flashattention.flash_attention import FlashAttentionBaseline
from adaattn.attention.adaptive_precision import AdaptivePrecisionAttention
from adaattn.attention.adaptive_rank import AdaptiveRankAttention
from adaattn.attention.adaattn import AdaAttention
from adaattn.attention.dense import DenseAttention


def get_memory_mb() -> float:
    """Get current memory allocated in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


def benchmark_forward(
    model: torch.nn.Module,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_iterations: int = 100,
    warmup: int = 10,
) -> Tuple[float, float]:
    """
    Benchmark forward pass.

    Returns:
        Tuple of (mean_time_ms, std_time_ms)
    """
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(q, k, v)

    # Synchronize if using CUDA
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start = time.perf_counter()
            _ = model(q, k, v)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

    times_tensor = torch.tensor(times)
    return times_tensor.mean().item(), times_tensor.std().item()


def benchmark_backward(
    model: torch.nn.Module,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_iterations: int = 100,
    warmup: int = 10,
) -> Tuple[float, float]:
    """
    Benchmark forward + backward pass.

    Returns:
        Tuple of (mean_time_ms, std_time_ms)
    """
    model.train()

    # Warmup
    for _ in range(warmup):
        q_grad = q.detach().clone().requires_grad_(True)
        k_grad = k.detach().clone().requires_grad_(True)
        v_grad = v.detach().clone().requires_grad_(True)

        output, _ = model(q_grad, k_grad, v_grad)
        loss = output.sum()
        loss.backward()

    # Synchronize if using CUDA
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(num_iterations):
        q_grad = q.detach().clone().requires_grad_(True)
        k_grad = k.detach().clone().requires_grad_(True)
        v_grad = v.detach().clone().requires_grad_(True)

        start = time.perf_counter()

        output, _ = model(q_grad, k_grad, v_grad)
        loss = output.sum()
        loss.backward()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    times_tensor = torch.tensor(times)
    return times_tensor.mean().item(), times_tensor.std().item()


def benchmark_memory(
    model: torch.nn.Module,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> Dict[str, float]:
    """
    Benchmark memory usage.

    Returns:
        Dictionary with memory statistics in MB
    """
    if not torch.cuda.is_available():
        return {"peak_mb": 0.0, "allocated_mb": 0.0}

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    model.eval()
    with torch.no_grad():
        _ = model(q, k, v)

    peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    allocated_mb = torch.cuda.memory_allocated() / 1024 / 1024

    return {
        "peak_mb": peak_mb,
        "allocated_mb": allocated_mb,
    }


def run_benchmark_suite(
    embed_dim: int = 512,
    num_heads: int = 8,
    batch_size: int = 8,
    seq_len: int = 512,
    device: str = "cpu",
    num_iterations: int = 100,
) -> Dict[str, Dict[str, float]]:
    """Run complete benchmark suite."""

    print(f"\n{'='*70}")
    print(f"Benchmark Configuration:")
    print(f"  embed_dim={embed_dim}, num_heads={num_heads}")
    print(f"  batch_size={batch_size}, seq_len={seq_len}")
    print(f"  device={device}, iterations={num_iterations}")
    print(f"{'='*70}\n")

    # Create input tensors
    q = torch.randn(batch_size, seq_len, embed_dim, device=device)
    k = torch.randn(batch_size, seq_len, embed_dim, device=device)
    v = torch.randn(batch_size, seq_len, embed_dim, device=device)

    # Models to benchmark
    models = {
        "Dense": DenseAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=0.0),
        "AdaptivePrecision": AdaptivePrecisionAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=0.0
        ),
        "AdaptiveRank": AdaptiveRankAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=0.0, rank_ratio=0.5
        ),
    }

    # Move models to device
    for model in models.values():
        model.to(device)
        model.eval()

    results = {}

    for name, model in models.items():
        print(f"Benchmarking {name}...")

        # Forward pass
        fwd_mean, fwd_std = benchmark_forward(
            model, q, k, v, num_iterations=num_iterations
        )

        # Backward pass
        bwd_mean, bwd_std = benchmark_backward(
            model, q, k, v, num_iterations=num_iterations
        )

        # Memory
        mem_stats = benchmark_memory(model, q, k, v)

        # Calculate throughput
        tokens_per_batch = batch_size * seq_len
        fwd_throughput = tokens_per_batch / (fwd_mean / 1000)  # tokens/sec
        bwd_throughput = tokens_per_batch / (bwd_mean / 1000)  # tokens/sec

        results[name] = {
            "forward_ms": fwd_mean,
            "forward_std": fwd_std,
            "backward_ms": bwd_mean,
            "backward_std": bwd_std,
            "total_ms": fwd_mean + bwd_mean,
            "fwd_throughput": fwd_throughput,
            "bwd_throughput": bwd_throughput,
            **mem_stats,
        }

        print(f"  Forward:  {fwd_mean:.3f} ± {fwd_std:.3f} ms")
        print(f"  Backward: {bwd_mean:.3f} ± {bwd_std:.3f} ms")
        print(f"  Total:    {fwd_mean + bwd_mean:.3f} ms")
        print(f"  Throughput (fwd): {fwd_throughput:.0f} tokens/sec")
        if device == "cuda":
            print(f"  Peak memory: {mem_stats['peak_mb']:.1f} MB")
        print()

    return results


def print_comparison_table(results: Dict[str, Dict[str, float]]):
    """Print formatted comparison table."""
    print(f"\n{'='*70}")
    print("Performance Comparison (relative to Dense baseline)")
    print(f"{'='*70}")

    baseline = results["Dense"]

    print(
        f"\n{'Model':<20} {'Forward':>12} {'Backward':>12} {'Total':>12} {'Memory':>12}"
    )
    print(f"{'-'*70}")

    for name, result in results.items():
        fwd_rel = result["forward_ms"] / baseline["forward_ms"]
        bwd_rel = result["backward_ms"] / baseline["backward_ms"]
        total_rel = result["total_ms"] / baseline["total_ms"]

        fwd_str = f"{fwd_rel:.2f}x"
        bwd_str = f"{bwd_rel:.2f}x"
        total_str = f"{total_rel:.2f}x"

        if name == "Dense":
            if result.get("peak_mb", 0) > 0:
                mem_str = f"{result.get('peak_mb', 0):.1f} MB"
            else:
                mem_str = "N/A"
        else:
            baseline_mem = baseline.get("peak_mb", 0)
            if baseline_mem > 0:
                mem_rel = result.get("peak_mb", 0) / baseline_mem
                mem_str = f"{mem_rel:.2f}x"
            else:
                mem_str = "N/A"

        print(f"{name:<20} {fwd_str:>12} {bwd_str:>12} {total_str:>12} {mem_str:>12}")

    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Benchmark attention implementations")
    parser.add_argument(
        "--embed-dim", type=int, default=512, help="Embedding dimension"
    )
    parser.add_argument(
        "--num-heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument(
        "--iterations", type=int, default=100, help="Number of benchmark iterations"
    )

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    results = run_benchmark_suite(
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        device=args.device,
        num_iterations=args.iterations,
    )

    print_comparison_table(results)


if __name__ == "__main__":
    main()
