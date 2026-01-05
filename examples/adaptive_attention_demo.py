"""
Demonstration of AdaAttn's adaptive attention mechanisms.

This example shows how to use:
1. Adaptive rank attention
2. Adaptive precision attention
3. Combined adaptive strategies
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from adaattn.attention.adaptive_rank import AdaptiveRankAttention
from adaattn.attention.adaptive_precision import AdaptivePrecisionAttention


def demo_adaptive_rank():
    """Demonstrate adaptive rank attention."""
    print("\n" + "="*60)
    print("ADAPTIVE RANK ATTENTION DEMO")
    print("="*60)
    
    # Create model with entropy-based rank selection
    model = AdaptiveRankAttention(
        embed_dim=256,
        num_heads=4,
        rank_ratio=0.5,
        rank_estimation_method="entropy",
        adaptive_threshold=0.8
    )
    
    # Generate sample data
    batch_size, seq_len = 2, 128
    q = torch.randn(batch_size, seq_len, 256)
    k = torch.randn(batch_size, seq_len, 256)
    v = torch.randn(batch_size, seq_len, 256)
    
    print(f"\nInput: batch_size={batch_size}, seq_len={seq_len}, embed_dim=256")
    print(f"Configuration: rank_ratio=0.5, method=entropy, threshold=0.8")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output, _ = model(q, k, v)
    
    print(f"Output shape: {output.shape}")
    
    # Get statistics
    stats = model.get_rank_statistics()
    print(f"\nRank Statistics:")
    print(f"  Total calls: {stats['call_count']}")
    print(f"  Low-rank usage: {stats['avg_low_rank_usage']:.1%}")
    print(f"  Per-head usage: {[f'{x:.1%}' for x in stats['low_rank_usage_per_head']]}")
    
    # Demonstrate rank prediction
    predicted_ranks, confidence = model.predict_optimal_rank(q, k, v)
    print(f"\nPredicted ranks (pre-computation):")
    print(f"  Ranks per head: {predicted_ranks[0].tolist()}")
    print(f"  Confidence: {[f'{x:.2f}' for x in confidence[0].tolist()]}")


def demo_adaptive_precision():
    """Demonstrate adaptive precision attention."""
    print("\n" + "="*60)
    print("ADAPTIVE PRECISION ATTENTION DEMO")
    print("="*60)
    
    # Create model with balanced precision policy
    model = AdaptivePrecisionAttention(
        embed_dim=256,
        num_heads=4,
        precision_policy="balanced",
        min_precision="fp16",
        max_precision="fp32"
    )
    
    # Generate sample data
    batch_size, seq_len = 2, 128
    q = torch.randn(batch_size, seq_len, 256)
    k = torch.randn(batch_size, seq_len, 256)
    v = torch.randn(batch_size, seq_len, 256)
    
    print(f"\nInput: batch_size={batch_size}, seq_len={seq_len}, embed_dim=256")
    print(f"Configuration: policy=balanced, range=fp32-fp16")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output, _ = model(q, k, v)
    
    print(f"Output shape: {output.shape}")
    
    # Get statistics
    stats = model.get_precision_statistics()
    print(f"\nPrecision Statistics:")
    print(f"  Total calls: {stats['call_count']}")
    print(f"  Precision usage:")
    for precision, percentage in stats['precision_usage'].items():
        if percentage > 0:
            print(f"    {precision}: {percentage:.1f}%")
    
    print(f"\nHardware Support:")
    hw = stats['hardware_support']
    print(f"  CUDA capability: {hw['cuda_capability']}")
    print(f"  BF16 support: {hw['supports_bf16']}")
    print(f"  FP8 support: {hw['supports_fp8']}")
    
    # Test different policies
    print(f"\nTesting different policies:")
    for policy in ["quality", "balanced", "speed"]:
        model.set_precision_policy(policy)
        with torch.no_grad():
            _ = model(q, k, v)
        stats = model.get_precision_statistics()
        used_precision = max(stats['precision_usage'].items(), key=lambda x: x[1])[0]
        print(f"  {policy:8s} -> {used_precision}")


def demo_comparison():
    """Compare different attention mechanisms."""
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    
    import time
    
    # Setup
    batch_size, seq_len, embed_dim, num_heads = 4, 256, 256, 4
    q = torch.randn(batch_size, seq_len, embed_dim)
    k = torch.randn(batch_size, seq_len, embed_dim)
    v = torch.randn(batch_size, seq_len, embed_dim)
    
    models = {
        "Adaptive Rank": AdaptiveRankAttention(embed_dim, num_heads),
        "Adaptive Precision": AdaptivePrecisionAttention(embed_dim, num_heads),
    }
    
    print(f"\nConfiguration: B={batch_size}, L={seq_len}, D={embed_dim}, H={num_heads}")
    print(f"\nRunning 50 iterations per model...\n")
    
    for name, model in models.items():
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(q, k, v)
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(50):
                start = time.perf_counter()
                output, _ = model(q, k, v)
                end = time.perf_counter()
                times.append((end - start) * 1000)
        
        avg_time = sum(times) / len(times)
        std_time = (sum((t - avg_time)**2 for t in times) / len(times)) ** 0.5
        
        print(f"{name:20s}: {avg_time:6.2f}ms (±{std_time:.2f}ms)")


def main():
    """Run all demonstrations."""
    print("\n" + "="*60)
    print("AdaAttn: Adaptive Attention Mechanisms Demo")
    print("="*60)
    
    demo_adaptive_rank()
    demo_adaptive_precision()
    demo_comparison()
    
    print("\n" + "="*60)
    print("Demo complete!")
    print("="*60)


if __name__ == "__main__":
    main()
