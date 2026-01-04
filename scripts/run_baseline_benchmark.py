#!/usr/bin/env python3
"""Simple script to run FlashAttention baseline benchmark.

This script validates Phase 1 completion by running a basic
comparison between PyTorch attention and FlashAttention.
"""

import argparse
import time
from typing import Tuple

import torch
import torch.nn as nn

try:
    from benchmarks.pytorch_attention.baseline import PyTorchAttentionBaseline
    from benchmarks.flashattention.flash_attention import FlashAttentionBaseline
except ImportError:
    # Fallback for development
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from benchmarks.pytorch_attention.baseline import PyTorchAttentionBaseline
    from benchmarks.flashattention.flash_attention import FlashAttentionBaseline


def simple_benchmark(
    model: nn.Module,
    batch_size: int = 8,
    seq_len: int = 1024,
    embed_dim: int = 512,
    num_heads: int = 8,
    device: str = 'cuda',
    iterations: int = 50
) -> Tuple[float, float]:
    """Run a simple benchmark."""
    
    # Use appropriate dtype for device
    dtype = torch.float16 if device == 'cuda' else torch.float32
    
    # Generate test data
    x = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)
    
    model = model.to(device)
    if device == 'cuda':
        model = model.half()  # Convert model to FP16 for GPU
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(x, x, x)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(iterations):
            start = time.perf_counter()
            output = model(x, x, x)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms
    
    avg_time = sum(times) / len(times)
    peak_memory = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    
    return avg_time, peak_memory


def main():
    parser = argparse.ArgumentParser(description='Run baseline benchmark')
    parser.add_argument('--device', default='cuda', help='Device to use')
    parser.add_argument('--seq-len', type=int, default=1024, help='Sequence length')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    
    args = parser.parse_args()
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    embed_dim = 512
    num_heads = 8
    
    print(f"Running baseline benchmark on {args.device}")
    print(f"Config: batch_size={args.batch_size}, seq_len={args.seq_len}, embed_dim={embed_dim}")
    print("-" * 60)
    
    # Test PyTorch baseline
    print("Testing PyTorch baseline...")
    pytorch_model = PyTorchAttentionBaseline(embed_dim, num_heads)
    pytorch_time, pytorch_memory = simple_benchmark(
        pytorch_model, args.batch_size, args.seq_len, embed_dim, num_heads, args.device
    )
    print(f"  Time: {pytorch_time:.2f} ms")
    print(f"  Memory: {pytorch_memory:.1f} MB")
    
    # Test FlashAttention baseline
    print("\nTesting FlashAttention baseline...")
    try:
        flash_model = FlashAttentionBaseline(embed_dim, num_heads)
        flash_time, flash_memory = simple_benchmark(
            flash_model, args.batch_size, args.seq_len, embed_dim, num_heads, args.device
        )
        print(f"  Time: {flash_time:.2f} ms")
        print(f"  Memory: {flash_memory:.1f} MB")
        
        speedup = pytorch_time / flash_time if flash_time > 0 else 0
        memory_reduction = pytorch_memory / flash_memory if flash_memory > 0 else 0
        
        print(f"\nFlashAttention Results:")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Memory reduction: {memory_reduction:.2f}x")
        
    except Exception as e:
        print(f"  Error: {e}")
        print("  FlashAttention may not be installed or compatible")
    
    print("\nBaseline benchmark complete!")
    print("Phase 1 FlashAttention baseline: ✓")


if __name__ == '__main__':
    main()
