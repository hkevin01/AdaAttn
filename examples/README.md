# AdaAttn Usage Examples

This directory contains examples demonstrating how to use the AdaAttn attention modules.

## Examples

### basic_usage.py

Demonstrates the core attention modules:

1. **Dense Attention** - Standard scaled dot-product attention
2. **Adaptive Precision Attention** - Attention with dynamic precision selection
3. **Adaptive Rank Attention** - Attention with dynamic rank approximation
4. **Cross-Attention** - Using different query and key/value sequence lengths
5. **Attention with Masking** - Both causal and custom masking

Run the examples:

```bash
python examples/basic_usage.py
```

## Key Concepts

### Dense Attention

Standard multi-head attention with optional causal masking:

```python
from adaattn.attention.dense import DenseAttention

attention = DenseAttention(
    embed_dim=512,
    num_heads=8,
    dropout=0.1,
    causal=False,  # Set True for autoregressive models
)

output, weights = attention(q, k, v, need_weights=True)
```

### Adaptive Precision Attention

Automatically selects numerical precision based on input characteristics:

```python
from adaattn.attention.adaptive_precision import AdaptivePrecisionAttention

attention = AdaptivePrecisionAttention(
    embed_dim=512,
    num_heads=8,
    default_precision="adaptive",  # Options: fp32, fp16, bf16, adaptive
)

output, _ = attention(q, k, v)
```

### Adaptive Rank Attention

Dynamically chooses between dense and low-rank computation:

```python
from adaattn.attention.adaptive_rank import AdaptiveRankAttention

attention = AdaptiveRankAttention(
    embed_dim=512,
    num_heads=8,
    rank_ratio=0.5,  # Use 50% of full rank
    rank_estimation_method="entropy",  # Options: entropy, power, random
    adaptive_threshold=0.95,
)

output, _ = attention(q, k, v)

# Get statistics on rank decisions
stats = attention.get_rank_statistics()
print(stats)
```

### Masking

Both causal and custom masking are supported:

```python
# Causal masking (for autoregressive models)
attention = DenseAttention(embed_dim=512, num_heads=8, causal=True)
output, _ = attention(q, k, v)

# Custom mask
mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
mask[:, seq_len // 2:] = False  # Attend only to first half
output, _ = attention(q, k, v, attention_mask=mask)
```

## Input/Output Shapes

All attention modules expect inputs in the format `(batch, sequence, embedding)`:

- **Query (Q)**: `(batch_size, seq_len_q, embed_dim)`
- **Key (K)**: `(batch_size, seq_len_k, embed_dim)`
- **Value (V)**: `(batch_size, seq_len_k, embed_dim)`

Output:
- **Output**: `(batch_size, seq_len_q, embed_dim)`
- **Weights** (if requested): `(batch_size, seq_len_q, seq_len_k)`

Note: Attention weights are averaged across heads when returned.

## Performance Tips

1. **Use `causal=True`** for autoregressive models instead of providing a causal mask
2. **Set `dropout=0.0`** during inference for deterministic behavior
3. **Use `need_weights=False`** (default) if you don't need attention weights
4. **Adaptive precision** can save memory and computation on compatible hardware
5. **Adaptive rank** is beneficial when attention patterns are structured/low-rank

## Next Steps

- See `tests/` for more comprehensive usage examples
- Check `docs/` for API documentation
- Read the README.md in the project root for architecture details
