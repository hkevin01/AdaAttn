"""
Comprehensive attention mechanism benchmarking suite.
"""

import time
import csv
import math
from typing import Dict, List, Tuple, Optional, Any
import json
from dataclasses import dataclass, asdict
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashattention.flash_attention import FlashAttentionBaseline
from pytorch_attention.baseline import PyTorchAttentionBaseline

@dataclass
class BenchmarkConfig:
    """Configuration for attention benchmarking."""
    batch_sizes: List[int]
    sequence_lengths: List[int]
    embedding_dims: List[int]
    num_heads: List[int]
    devices: List[str]
    iterations: int = 50
    warmup_iterations: int = 10
    output_dir: str = "results"
    
@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    model_name: str
    batch_size: int
    seq_len: int
    embed_dim: int
    num_heads: int
    device: str
    avg_time_ms: float
    std_time_ms: float
    peak_memory_mb: float
    throughput_tokens_per_sec: float
    
class AttentionBenchmark:
    """Comprehensive attention benchmarking suite."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results: List[BenchmarkResult] = []
    
    def _get_model(self, model_name: str, embed_dim: int, num_heads: int) -> nn.Module:
        """Get attention model by name."""
        if model_name == "pytorch":
            return PyTorchAttentionBaseline(embed_dim, num_heads)
        elif model_name == "flash":
            return FlashAttentionBaseline(embed_dim, num_heads)
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def _benchmark_single(
        self,
        model: nn.Module,
        batch_size: int,
        seq_len: int,
        embed_dim: int,
        num_heads: int,
        device: str,
        model_name: str
    ) -> BenchmarkResult:
        """Benchmark a single configuration."""
        
        # Use appropriate dtype
        dtype = torch.float16 if device == 'cuda' else torch.float32
        
        # Generate test data
        x = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)
        
        # Prepare model
        model = model.to(device)
        if device == 'cuda':
            model = model.half()
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(self.config.warmup_iterations):
                _ = model(x, x, x)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(self.config.iterations):
                start = time.perf_counter()
                output = model(x, x, x)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                end = time.perf_counter()
                times.append((end - start) * 1000)  # ms
        
        # Calculate metrics
        avg_time = sum(times) / len(times)
        std_time = math.sqrt(sum((t - avg_time) ** 2 for t in times) / len(times))
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        
        # Throughput: tokens/sec
        total_tokens = batch_size * seq_len
        throughput = (total_tokens / avg_time) * 1000  # tokens per second
        
        return BenchmarkResult(
            model_name=model_name,
            batch_size=batch_size,
            seq_len=seq_len,
            embed_dim=embed_dim,
            num_heads=num_heads,
            device=device,
            avg_time_ms=avg_time,
            std_time_ms=std_time,
            peak_memory_mb=peak_memory,
            throughput_tokens_per_sec=throughput
        )
    
    def run_benchmarks(self, models: List[str] = None) -> List[BenchmarkResult]:
        """Run comprehensive benchmarks."""
        
        if models is None:
            models = ["pytorch", "flash"]
            
        print("Starting comprehensive attention benchmarking...")
        print(f"Models: {models}")
        print(f"Configurations: {len(self.config.batch_sizes) * len(self.config.sequence_lengths) * len(self.config.embedding_dims) * len(self.config.num_heads) * len(self.config.devices)}")
        
        total_runs = 0
        current_run = 0
        
        # Calculate total runs for progress tracking
        for device in self.config.devices:
            for batch_size in self.config.batch_sizes:
                for seq_len in self.config.sequence_lengths:
                    for embed_dim in self.config.embedding_dims:
                        for num_heads in self.config.num_heads:
                            total_runs += len(models)
        
        for device in self.config.devices:
            print(f"\nBenchmarking on {device}...")
            
            # Skip CUDA if not available
            if device == 'cuda' and not torch.cuda.is_available():
                print(f"CUDA not available, skipping {device}")
                continue
                
            for batch_size in self.config.batch_sizes:
                for seq_len in self.config.sequence_lengths:
                    for embed_dim in self.config.embedding_dims:
                        for num_heads in self.config.num_heads:
                            # Check if configuration is valid
                            if embed_dim % num_heads != 0:
                                continue
                            
                            print(f"\n  Config: B={batch_size}, L={seq_len}, D={embed_dim}, H={num_heads}")
                            
                            for model_name in models:
                                current_run += 1
                                progress = (current_run / total_runs) * 100
                                print(f"    [{progress:5.1f}%] {model_name}...", end=" ")
                                
                                try:
                                    model = self._get_model(model_name, embed_dim, num_heads)
                                    result = self._benchmark_single(
                                        model, batch_size, seq_len, embed_dim, num_heads, device, model_name
                                    )
                                    self.results.append(result)
                                    print(f"✓ {result.avg_time_ms:.2f}ms")
                                    
                                except Exception as e:
                                    print(f"✗ Error: {e}")
                                    continue
        
        print(f"\nBenchmarking complete! {len(self.results)} results collected.")
        return self.results
    
    def save_results(self, filename: str = None) -> str:
        """Save benchmark results to CSV and JSON."""
        
        if filename is None:
            filename = f"attention_benchmark_results"
        
        # Save as CSV
        csv_file = f"{self.config.output_dir}/{filename}.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            if self.results:
                writer.writerow(asdict(self.results[0]).keys())
                
                # Data
                for result in self.results:
                    writer.writerow(asdict(result).values())
        
        # Save as JSON
        json_file = f"{self.config.output_dir}/{filename}.json"
        with open(json_file, 'w') as f:
            json.dump([asdict(result) for result in self.results], f, indent=2)
        
        print(f"Results saved to {csv_file} and {json_file}")
        return csv_file
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze benchmark results."""
        
        if not self.results:
            return {}
        
        analysis = {
            "summary": {
                "total_runs": len(self.results),
                "models": list(set(r.model_name for r in self.results)),
                "devices": list(set(r.device for r in self.results))
            },
            "speedups": {},
            "memory_comparison": {}
        }
        
        # Group by configuration for comparison
        config_groups = {}
        for result in self.results:
            key = (result.batch_size, result.seq_len, result.embed_dim, result.num_heads, result.device)
            if key not in config_groups:
                config_groups[key] = {}
            config_groups[key][result.model_name] = result
        
        # Calculate speedups (vs pytorch baseline)
        speedups = []
        memory_ratios = []
        
        for config, models in config_groups.items():
            if "pytorch" in models and "flash" in models:
                pytorch_time = models["pytorch"].avg_time_ms
                flash_time = models["flash"].avg_time_ms
                speedup = pytorch_time / flash_time
                speedups.append(speedup)
                
                pytorch_mem = models["pytorch"].peak_memory_mb
                flash_mem = models["flash"].peak_memory_mb
                if pytorch_mem > 0:
                    memory_ratio = pytorch_mem / flash_mem
                    memory_ratios.append(memory_ratio)
        
        if speedups:
            analysis["speedups"] = {
                "mean": sum(speedups) / len(speedups),
                "min": min(speedups),
                "max": max(speedups),
                "values": speedups
            }
        
        if memory_ratios:
            analysis["memory_comparison"] = {
                "mean": sum(memory_ratios) / len(memory_ratios),
                "min": min(memory_ratios),
                "max": max(memory_ratios),
                "values": memory_ratios
            }
        
        return analysis
    
    def print_summary(self):
        """Print benchmark summary."""
        
        if not self.results:
            print("No results to summarize")
            return
        
        analysis = self.analyze_results()
        
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        
        print(f"Total runs: {analysis['summary']['total_runs']}")
        print(f"Models tested: {', '.join(analysis['summary']['models'])}")
        print(f"Devices: {', '.join(analysis['summary']['devices'])}")
        
        if "speedups" in analysis:
            speedups = analysis["speedups"]
            print(f"\nFlashAttention Speedup vs PyTorch:")
            print(f"  Mean: {speedups['mean']:.2f}x")
            print(f"  Range: {speedups['min']:.2f}x - {speedups['max']:.2f}x")
        
        if "memory_comparison" in analysis and analysis["memory_comparison"]:
            memory = analysis["memory_comparison"]
            print(f"\nMemory Usage Comparison:")
            print(f"  PyTorch/FlashAttention ratio: {memory['mean']:.2f}x")
        
        # Show best and worst performers
        best_result = min(self.results, key=lambda r: r.avg_time_ms)
        worst_result = max(self.results, key=lambda r: r.avg_time_ms)
        
        print(f"\nFastest configuration:")
        print(f"  {best_result.model_name}: {best_result.avg_time_ms:.2f}ms")
        print(f"  Config: B={best_result.batch_size}, L={best_result.seq_len}, D={best_result.embed_dim}, H={best_result.num_heads}")
        
        print(f"\nSlowest configuration:")
        print(f"  {worst_result.model_name}: {worst_result.avg_time_ms:.2f}ms")
        print(f"  Config: B={worst_result.batch_size}, L={worst_result.seq_len}, D={worst_result.embed_dim}, H={worst_result.num_heads}")


# Default benchmark configuration
DEFAULT_CONFIG = BenchmarkConfig(
    batch_sizes=[4, 8],
    sequence_lengths=[512, 1024],
    embedding_dims=[256, 512],
    num_heads=[4, 8],
    devices=['cpu'] + (['cuda'] if torch.cuda.is_available() else []),
    iterations=20,
    warmup_iterations=5
)


def main():
    """Run comprehensive attention benchmarks."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Attention Benchmark Suite")
    parser.add_argument("--config", type=str, help="JSON config file")
    parser.add_argument("--models", nargs="+", default=["pytorch", "flash"], help="Models to benchmark")
    parser.add_argument("--output", type=str, default="attention_benchmark", help="Output filename")
    parser.add_argument("--quick", action="store_true", help="Quick benchmark with reduced configurations")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = BenchmarkConfig(**config_dict)
    else:
        config = DEFAULT_CONFIG
        
        # Quick benchmark mode
        if args.quick:
            config.batch_sizes = [4]
            config.sequence_lengths = [512]
            config.embedding_dims = [256]
            config.num_heads = [4]
            config.iterations = 10
    
    # Ensure output directory exists
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Run benchmarks
    benchmark = AttentionBenchmark(config)
    results = benchmark.run_benchmarks(args.models)
    
    # Save and analyze results
    benchmark.save_results(args.output)
    benchmark.print_summary()


if __name__ == "__main__":
    main()
