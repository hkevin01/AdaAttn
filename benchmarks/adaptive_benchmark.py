"""
Comprehensive benchmark for adaptive attention mechanisms.

This benchmarks the core AdaAttn components:
- Adaptive rank selection
- Adaptive precision selection  
- Combined AdaAttn implementation
"""

import time
import json
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
import os

import torch
import torch.nn as nn
from torch import Tensor

from pytorch_attention.baseline import PyTorchAttentionBaseline
from flashattention.flash_attention import FlashAttentionBaseline

# Import our adaptive modules
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from adaattn.attention.adaptive_rank import AdaptiveRankAttention
from adaattn.attention.adaptive_precision import AdaptivePrecisionAttention


@dataclass
class AdaptiveBenchmarkResult:
    """Result from adaptive attention benchmark."""
    model_name: str
    batch_size: int
    seq_len: int
    embed_dim: int
    num_heads: int
    device: str
    avg_time_ms: float
    peak_memory_mb: float
    adaptation_stats: Dict[str, Any]
    
    
class AdaptiveAttentionBenchmark:
    """Benchmark suite for adaptive attention mechanisms."""
    
    def __init__(self, device: str = "auto"):
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.results: List[AdaptiveBenchmarkResult] = []
        
    def benchmark_adaptive_rank(
        self, 
        batch_size: int = 4,
        seq_len: int = 512,
        embed_dim: int = 256,
        num_heads: int = 4,
        iterations: int = 20
    ) -> AdaptiveBenchmarkResult:
        """Benchmark adaptive rank attention."""
        
        print(f"    Testing adaptive rank attention...")
        
        # Create model
        model = AdaptiveRankAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            rank_ratio=0.5,
            rank_estimation_method="entropy",
            adaptive_threshold=0.8
        )
        
        model = model.to(self.device)
        if self.device == "cuda":
            model = model.half()
        model.eval()
        
        # Generate test data
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        q = torch.randn(batch_size, seq_len, embed_dim, device=self.device, dtype=dtype)
        k = torch.randn(batch_size, seq_len, embed_dim, device=self.device, dtype=dtype)  
        v = torch.randn(batch_size, seq_len, embed_dim, device=self.device, dtype=dtype)
        
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model(q, k, v)
        
        if self.device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(iterations):
                start = time.perf_counter()
                output, _ = model(q, k, v)
                
                if self.device == "cuda":
                    torch.cuda.synchronize()
                    
                end = time.perf_counter()
                times.append((end - start) * 1000)
        
        avg_time = sum(times) / len(times)
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2 if self.device == "cuda" else 0
        
        # Get adaptation statistics
        adaptation_stats = model.get_rank_statistics()
        
        return AdaptiveBenchmarkResult(
            model_name="adaptive_rank",
            batch_size=batch_size,
            seq_len=seq_len,
            embed_dim=embed_dim,
            num_heads=num_heads,
            device=self.device,
            avg_time_ms=avg_time,
            peak_memory_mb=peak_memory,
            adaptation_stats=adaptation_stats
        )
    
    def benchmark_adaptive_precision(
        self,
        batch_size: int = 4,
        seq_len: int = 512,
        embed_dim: int = 256,
        num_heads: int = 4,
        iterations: int = 20
    ) -> AdaptiveBenchmarkResult:
        """Benchmark adaptive precision attention."""
        
        print(f"    Testing adaptive precision attention...")
        
        # Create model
        model = AdaptivePrecisionAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            precision_policy="balanced",
            min_precision="fp16",
            max_precision="fp32"
        )
        
        model = model.to(self.device)
        model.eval()
        
        # Generate test data
        q = torch.randn(batch_size, seq_len, embed_dim, device=self.device)
        k = torch.randn(batch_size, seq_len, embed_dim, device=self.device)
        v = torch.randn(batch_size, seq_len, embed_dim, device=self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model(q, k, v)
        
        if self.device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(iterations):
                start = time.perf_counter()
                output, _ = model(q, k, v)
                
                if self.device == "cuda":
                    torch.cuda.synchronize()
                    
                end = time.perf_counter()
                times.append((end - start) * 1000)
        
        avg_time = sum(times) / len(times)
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2 if self.device == "cuda" else 0
        
        # Get adaptation statistics
        adaptation_stats = model.get_precision_statistics()
        
        return AdaptiveBenchmarkResult(
            model_name="adaptive_precision",
            batch_size=batch_size,
            seq_len=seq_len,
            embed_dim=embed_dim,
            num_heads=num_heads,
            device=self.device,
            avg_time_ms=avg_time,
            peak_memory_mb=peak_memory,
            adaptation_stats=adaptation_stats
        )
    
    def run_comparison_suite(self):
        """Run comprehensive comparison of all attention mechanisms."""
        
        configs = [
            (4, 256, 256, 4),   # Small
            (4, 512, 256, 4),   # Medium sequence
            (4, 512, 512, 8),   # Medium model
        ]
        
        print(f"Running adaptive attention benchmark suite on {self.device}")
        print("="*60)
        
        for batch_size, seq_len, embed_dim, num_heads in configs:
            print(f"\nConfiguration: B={batch_size}, L={seq_len}, D={embed_dim}, H={num_heads}")
            print("-" * 50)
            
            try:
                # Benchmark adaptive rank
                rank_result = self.benchmark_adaptive_rank(
                    batch_size, seq_len, embed_dim, num_heads
                )
                self.results.append(rank_result)
                print(f"    ✓ Adaptive Rank: {rank_result.avg_time_ms:.2f}ms")
                
            except Exception as e:
                print(f"    ✗ Adaptive Rank failed: {e}")
            
            try:
                # Benchmark adaptive precision
                precision_result = self.benchmark_adaptive_precision(
                    batch_size, seq_len, embed_dim, num_heads
                )
                self.results.append(precision_result)
                print(f"    ✓ Adaptive Precision: {precision_result.avg_time_ms:.2f}ms")
                
            except Exception as e:
                print(f"    ✗ Adaptive Precision failed: {e}")
    
    def save_results(self, filename: str = "adaptive_benchmark_results"):
        """Save benchmark results."""
        
        os.makedirs("results", exist_ok=True)
        
        # Save as JSON
        json_file = f"results/{filename}.json"
        with open(json_file, 'w') as f:
            json.dump([asdict(result) for result in self.results], f, indent=2)
        
        print(f"\nResults saved to {json_file}")
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print benchmark summary."""
        
        if not self.results:
            print("No results to summarize")
            return
            
        print("\n" + "="*60)
        print("ADAPTIVE ATTENTION BENCHMARK SUMMARY")
        print("="*60)
        
        print(f"Total configurations tested: {len(self.results)}")
        print(f"Device: {self.device}")
        
        # Group by model type
        rank_results = [r for r in self.results if r.model_name == "adaptive_rank"]
        precision_results = [r for r in self.results if r.model_name == "adaptive_precision"]
        
        if rank_results:
            avg_rank_time = sum(r.avg_time_ms for r in rank_results) / len(rank_results)
            print(f"\nAdaptive Rank Average: {avg_rank_time:.2f}ms")
            
            # Print rank statistics
            for result in rank_results:
                stats = result.adaptation_stats
                if "avg_low_rank_usage" in stats:
                    print(f"  Low-rank usage: {stats['avg_low_rank_usage']:.1%}")
        
        if precision_results:
            avg_precision_time = sum(r.avg_time_ms for r in precision_results) / len(precision_results)
            print(f"\nAdaptive Precision Average: {avg_precision_time:.2f}ms")
            
            # Print precision statistics
            for result in precision_results:
                stats = result.adaptation_stats
                if "precision_usage" in stats:
                    print(f"  Precision usage: {stats['precision_usage']}")
        
        # Show best configuration
        if self.results:
            best_result = min(self.results, key=lambda r: r.avg_time_ms)
            print(f"\nBest configuration:")
            print(f"  Model: {best_result.model_name}")
            print(f"  Time: {best_result.avg_time_ms:.2f}ms") 
            print(f"  Config: B={best_result.batch_size}, L={best_result.seq_len}, D={best_result.embed_dim}, H={best_result.num_heads}")


def main():
    """Run adaptive attention benchmarks."""
    
    benchmark = AdaptiveAttentionBenchmark()
    benchmark.run_comparison_suite()
    benchmark.save_results("phase2_adaptive_results")


if __name__ == "__main__":
    main()
