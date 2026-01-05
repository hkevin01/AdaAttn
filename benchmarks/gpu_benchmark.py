"""
GPU optimization benchmarks for AdaAttn.
Tests FlashAttention, CUDA kernels, and memory efficiency.
"""

import time
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional
import logging

from adaattn.attention.adaattn import AdaAttention
from adaattn.attention.base import PrecisionMode

logger = logging.getLogger(__name__)


class GPUBenchmark:
    """Comprehensive GPU benchmarking for AdaAttn optimizations."""
    
    def __init__(self, device: str = "auto"):
        self.device = self._setup_device(device)
        self.results = []
        
    def _setup_device(self, device: str) -> torch.device:
        """Setup compute device."""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
                logger.warning("CUDA not available, using CPU")
        
        return torch.device(device)
    
    def benchmark_attention_implementations(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        seq_lens: List[int] = None,
        batch_sizes: List[int] = None,
        num_runs: int = 10,
        warmup_runs: int = 3,
    ) -> Dict[str, Any]:
        """Benchmark different attention implementations."""
        
        if seq_lens is None:
            seq_lens = [128, 512, 1024, 2048]
        if batch_sizes is None:
            batch_sizes = [1, 4, 8]
        
        print("=" * 80)
        print("GPU Attention Implementation Benchmark")
        print("=" * 80)
        print(f"Device: {self.device}")
        print(f"Embed dim: {embed_dim}, Num heads: {num_heads}")
        print(f"Sequence lengths: {seq_lens}")
        print(f"Batch sizes: {batch_sizes}")
        print()
        
        configurations = [
            ("AdaAttn (CPU)", {"enable_gpu_optimization": False}),
            ("AdaAttn (GPU)", {"enable_gpu_optimization": True}),
            ("PyTorch SDPA", None),  # Reference implementation
        ]
        
        results = {}
        
        for config_name, config_kwargs in configurations:
            print(f"Testing: {config_name}")
            config_results = {}
            
            for batch_size in batch_sizes:
                for seq_len in seq_lens:
                    key = f"B{batch_size}_S{seq_len}"
                    
                    # Create model
                    if config_name == "PyTorch SDPA":
                        model = self._create_pytorch_reference(embed_dim, num_heads)
                    else:
                        model = AdaAttention(
                            embed_dim=embed_dim,
                            num_heads=num_heads,
                            **config_kwargs
                        )
                    
                    model = model.to(self.device).eval()
                    
                    # Create inputs
                    inputs = self._create_inputs(batch_size, seq_len, embed_dim)
                    
                    try:
                        # Benchmark
                        times = self._benchmark_forward_pass(
                            model, inputs, num_runs, warmup_runs
                        )
                        
                        config_results[key] = {
                            "mean_time_ms": times["mean"] * 1000,
                            "std_time_ms": times["std"] * 1000,
                            "memory_mb": self._get_peak_memory() / 1024**2,
                        }
                        
                        print(f"  {key}: {times['mean']*1000:.2f}±{times['std']*1000:.2f}ms")
                        
                    except Exception as e:
                        print(f"  {key}: Failed - {e}")
                        config_results[key] = {"error": str(e)}
                    
                    # Cleanup
                    del model
                    torch.cuda.empty_cache() if self.device.type == "cuda" else None
            
            results[config_name] = config_results
            print()
        
        return results
    
    def benchmark_precision_modes(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        seq_len: int = 1024,
        batch_size: int = 4,
        num_runs: int = 10,
    ) -> Dict[str, Any]:
        """Benchmark different precision modes."""
        
        if not torch.cuda.is_available():
            print("CUDA not available, skipping precision benchmark")
            return {}
        
        print("=" * 80)
        print("Precision Mode Benchmark")
        print("=" * 80)
        
        precision_modes = [
            PrecisionMode.FP32,
            PrecisionMode.FP16,
        ]
        
        # Add BF16 if supported
        if torch.cuda.is_bf16_supported():
            precision_modes.append(PrecisionMode.BF16)
        
        results = {}
        inputs = self._create_inputs(batch_size, seq_len, embed_dim)
        
        for precision in precision_modes:
            print(f"Testing precision: {precision.name}")
            
            # Create model with specific precision
            model = AdaAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                default_precision=precision,
            ).to(self.device).eval()
            
            try:
                # Convert inputs to target precision
                target_dtype = precision.to_dtype()
                test_inputs = {k: v.to(target_dtype) for k, v in inputs.items()}
                
                times = self._benchmark_forward_pass(model, test_inputs, num_runs, 3)
                peak_memory = self._get_peak_memory() / 1024**2
                
                # Test accuracy (compare to FP32)
                if precision != PrecisionMode.FP32:
                    accuracy = self._test_precision_accuracy(model, inputs, precision)
                else:
                    accuracy = {"mse": 0.0, "max_diff": 0.0}
                
                results[precision.name] = {
                    "mean_time_ms": times["mean"] * 1000,
                    "peak_memory_mb": peak_memory,
                    "accuracy": accuracy,
                }
                
                print(f"  Time: {times['mean']*1000:.2f}ms")
                print(f"  Memory: {peak_memory:.1f}MB")
                print(f"  MSE vs FP32: {accuracy['mse']:.2e}")
                
            except Exception as e:
                print(f"  Failed: {e}")
                results[precision.name] = {"error": str(e)}
            
            del model
            torch.cuda.empty_cache()
            print()
        
        return results
    
    def benchmark_memory_efficiency(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        max_seq_len: int = 8192,
        batch_size: int = 1,
    ) -> Dict[str, Any]:
        """Benchmark memory usage for long sequences."""
        
        print("=" * 80)
        print("Memory Efficiency Benchmark")
        print("=" * 80)
        
        seq_lens = []
        current = 256
        while current <= max_seq_len:
            seq_lens.append(current)
            current *= 2
        
        results = {}
        
        for seq_len in seq_lens:
            print(f"Testing sequence length: {seq_len}")
            
            try:
                # Test standard AdaAttn
                model = AdaAttention(embed_dim=embed_dim, num_heads=num_heads)
                model = model.to(self.device).eval()
                
                inputs = self._create_inputs(batch_size, seq_len, embed_dim)
                
                # Measure memory before
                torch.cuda.reset_peak_memory_stats() if self.device.type == "cuda" else None
                
                with torch.no_grad():
                    output, _ = model(**inputs)
                
                peak_memory = self._get_peak_memory() / 1024**2
                
                results[f"seq_len_{seq_len}"] = {
                    "peak_memory_mb": peak_memory,
                    "output_shape": list(output.shape),
                }
                
                print(f"  Peak memory: {peak_memory:.1f}MB")
                
                del model, output
                torch.cuda.empty_cache() if self.device.type == "cuda" else None
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  OOM at sequence length {seq_len}")
                    break
                else:
                    raise e
        
        return results
    
    def _create_pytorch_reference(self, embed_dim: int, num_heads: int) -> torch.nn.Module:
        """Create PyTorch reference implementation."""
        class PyTorchAttention(torch.nn.Module):
            def __init__(self, embed_dim: int, num_heads: int):
                super().__init__()
                self.attention = torch.nn.MultiheadAttention(
                    embed_dim, num_heads, batch_first=True
                )
            
            def forward(self, query, key=None, value=None, **kwargs):
                if key is None:
                    key = query
                if value is None:
                    value = key
                output, _ = self.attention(query, key, value)
                return output, None
        
        return PyTorchAttention(embed_dim, num_heads)
    
    def _create_inputs(
        self, 
        batch_size: int, 
        seq_len: int, 
        embed_dim: int
    ) -> Dict[str, torch.Tensor]:
        """Create test inputs."""
        return {
            "query": torch.randn(batch_size, seq_len, embed_dim, device=self.device),
            "key": torch.randn(batch_size, seq_len, embed_dim, device=self.device),
            "value": torch.randn(batch_size, seq_len, embed_dim, device=self.device),
        }
    
    def _benchmark_forward_pass(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        num_runs: int,
        warmup_runs: int,
    ) -> Dict[str, float]:
        """Benchmark forward pass timing."""
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(**inputs)
        
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.perf_counter()
                _ = model(**inputs)
                
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                
                end = time.perf_counter()
                times.append(end - start)
        
        return {
            "mean": sum(times) / len(times),
            "std": (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5,
            "min": min(times),
            "max": max(times),
        }
    
    def _get_peak_memory(self) -> int:
        """Get peak memory usage in bytes."""
        if self.device.type == "cuda":
            return torch.cuda.max_memory_allocated()
        else:
            # For CPU, we can't easily get peak memory, return 0
            return 0
    
    def _test_precision_accuracy(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        precision: PrecisionMode,
    ) -> Dict[str, float]:
        """Test numerical accuracy compared to FP32."""
        
        # Get FP32 reference
        model_fp32 = AdaAttention(
            embed_dim=model.config.embed_dim,
            num_heads=model.config.num_heads,
            default_precision=PrecisionMode.FP32,
        ).to(self.device).eval()
        
        with torch.no_grad():
            output_fp32, _ = model_fp32(**inputs)
            output_test, _ = model(**inputs)
            
            # Convert to same dtype for comparison
            output_test_fp32 = output_test.float()
            
            mse = F.mse_loss(output_test_fp32, output_fp32).item()
            max_diff = (output_test_fp32 - output_fp32).abs().max().item()
        
        del model_fp32
        torch.cuda.empty_cache() if self.device.type == "cuda" else None
        
        return {"mse": mse, "max_diff": max_diff}


def run_gpu_benchmarks():
    """Run comprehensive GPU benchmarks."""
    print("AdaAttn GPU Optimization Benchmarks")
    print("=" * 80)
    
    benchmark = GPUBenchmark()
    
    # Test different implementations
    impl_results = benchmark.benchmark_attention_implementations(
        embed_dim=512,
        num_heads=8,
        seq_lens=[128, 512, 1024],
        batch_sizes=[1, 4],
        num_runs=10,
    )
    
    # Test precision modes (CUDA only)
    precision_results = benchmark.benchmark_precision_modes()
    
    # Test memory efficiency
    memory_results = benchmark.benchmark_memory_efficiency()
    
    # Summary
    print("=" * 80)
    print("Benchmark Summary")
    print("=" * 80)
    
    print("\n1. Implementation Comparison:")
    for impl, results in impl_results.items():
        print(f"  {impl}:")
        for config, metrics in results.items():
            if "error" not in metrics:
                print(f"    {config}: {metrics['mean_time_ms']:.2f}ms")
    
    if precision_results:
        print("\n2. Precision Mode Performance:")
        for precision, metrics in precision_results.items():
            if "error" not in metrics:
                print(f"  {precision}: {metrics['mean_time_ms']:.2f}ms")
    
    print("\n3. Memory Efficiency:")
    for config, metrics in memory_results.items():
        print(f"  {config}: {metrics['peak_memory_mb']:.1f}MB")


if __name__ == "__main__":
    run_gpu_benchmarks()
