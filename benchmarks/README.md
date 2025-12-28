# AdaAttn Benchmarks

This directory contains benchmarking scripts for evaluating the performance of different attention implementations.

## Quick Start

Run basic benchmark:
```bash
python benchmarks/benchmark_attention.py
```

Run with custom parameters:
```bash
python benchmarks/benchmark_attention.py \
    --batch-size 8 \
    --seq-len 512 \
    --embed-dim 512 \
    --num-heads 8 \
    --iterations 100 \
    --device cpu
```

For GPU benchmarks (requires CUDA):
```bash
python benchmarks/benchmark_attention.py --device cuda
```

## Benchmark Scripts

### benchmark_attention.py

Compares different attention implementations:
- **Dense Attention**: Standard baseline implementation
- **Adaptive Precision**: Dynamic precision selection
- **Adaptive Rank**: Dynamic rank approximation

**Metrics measured:**
- Forward pass latency (mean ± std)
- Backward pass latency (mean ± std)  
- Total latency
- Throughput (tokens/second)
- Memory usage (GPU only)

**Example output:**
```
Benchmark Configuration:
  embed_dim=512, num_heads=8
  batch_size=2, seq_len=128
  device=cpu, iterations=20

Benchmarking Dense...
  Forward:  45.374 ± 0.874 ms
  Backward: 51.294 ± 0.622 ms
  Total:    96.668 ms
  Throughput (fwd): 5642 tokens/sec

Performance Comparison (relative to Dense baseline)

Model                     Forward     Backward        Total       Memory
----------------------------------------------------------------------
Dense                       1.00x        1.00x        1.00x          N/A
AdaptivePrecision           1.49x        1.76x        1.64x          N/A
AdaptiveRank                1.01x        1.02x        1.01x          N/A
```

## Parameters

- `--embed-dim`: Embedding dimension (default: 512)
- `--num-heads`: Number of attention heads (default: 8)
- `--batch-size`: Batch size (default: 8)
- `--seq-len`: Sequence length (default: 512)
- `--device`: Device to use, 'cpu' or 'cuda' (default: cpu)
- `--iterations`: Number of benchmark iterations (default: 100)

## Interpreting Results

### Performance Ratios
- **< 1.0x**: Faster than baseline (better)
- **= 1.0x**: Same speed as baseline
- **> 1.0x**: Slower than baseline (worse)

### When to Use Each Implementation

**Dense Attention:**
- Baseline reference
- Best when you need maximum accuracy
- Simple and well-tested

**Adaptive Precision:**
- GPU workloads with mixed precision support
- When memory bandwidth is the bottleneck
- Can sacrifice some numerical precision

**Adaptive Rank:**
- When attention patterns are structured/low-rank
- Large sequence lengths
- Minimal overhead on CPU (~1-2%)

## Performance Tips

1. **Warmup iterations**: First few runs are slower due to PyTorch JIT compilation
2. **Batch size**: Larger batches improve GPU utilization
3. **Sequence length**: Performance characteristics change with length
4. **Device**: GPU benchmarks require CUDA-capable hardware
5. **Iterations**: More iterations give more stable measurements (but take longer)

## Comparing with FlashAttention

To compare with FlashAttention (requires separate installation):

```bash
pip install flash-attn --no-build-isolation
```

Then benchmark FlashAttention separately or add it to the benchmark suite.

## Next Steps

- See `experiments/` for ablation studies
- Check `results/` for detailed benchmark logs
- Read the main README for architecture details
