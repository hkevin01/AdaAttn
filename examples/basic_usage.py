"""
Basic usage examples for AdaAttn attention modules.
"""

import torch

from adaattn.attention.adaptive_precision import AdaptivePrecisionAttention
from adaattn.attention.adaptive_rank import AdaptiveRankAttention
from adaattn.attention.dense import DenseAttention


def example_dense_attention():
    """Example of using standard dense attention."""
    print("=" * 60)
    print("Example 1: Dense Attention")
    print("=" * 60)

    # Create attention module
    attention = DenseAttention(
        embed_dim=512,
        num_heads=8,
        dropout=0.1,
        causal=False,
    )

    # Create input tensors
    batch_size, seq_len = 4, 128
    q = torch.randn(batch_size, seq_len, 512)
    k = torch.randn(batch_size, seq_len, 512)
    v = torch.randn(batch_size, seq_len, 512)

    # Forward pass
    output, weights = attention(q, k, v, need_weights=True)

    print(f"Input shape: {q.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
    print()


def example_adaptive_precision():
    """Example of using adaptive precision attention."""
    print("=" * 60)
    print("Example 2: Adaptive Precision Attention")
    print("=" * 60)

    # Create attention module with adaptive precision
    attention = AdaptivePrecisionAttention(
        embed_dim=512,
        num_heads=8,
        dropout=0.1,
        default_precision="adaptive",  # Can be: fp32, fp16, bf16, adaptive
    )

    # Create input tensors
    batch_size, seq_len = 4, 128
    q = torch.randn(batch_size, seq_len, 512)
    k = torch.randn(batch_size, seq_len, 512)
    v = torch.randn(batch_size, seq_len, 512)

    # Forward pass
    output, weights = attention(q, k, v)

    print(f"Input shape: {q.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Default precision: {attention.default_precision}")
    print()


def example_adaptive_rank():
    """Example of using adaptive rank attention."""
    print("=" * 60)
    print("Example 3: Adaptive Rank Attention")
    print("=" * 60)

    # Create attention module with adaptive rank
    attention = AdaptiveRankAttention(
        embed_dim=512,
        num_heads=8,
        dropout=0.1,
        rank_ratio=0.5,  # Use 50% of full rank
        rank_estimation_method="entropy",  # Can be: entropy, power, random
        adaptive_threshold=0.95,
    )

    # Create input tensors
    batch_size, seq_len = 4, 128
    q = torch.randn(batch_size, seq_len, 512)
    k = torch.randn(batch_size, seq_len, 512)
    v = torch.randn(batch_size, seq_len, 512)

    # Forward pass
    output, weights = attention(q, k, v)

    print(f"Input shape: {q.shape}")
    print(f"Output shape: {output.shape}")

    # Get rank statistics
    stats = attention.get_rank_statistics()
    print(f"Rank statistics: {stats}")
    print()


def example_cross_attention():
    """Example of using cross-attention."""
    print("=" * 60)
    print("Example 4: Cross-Attention")
    print("=" * 60)

    # Create attention module
    attention = DenseAttention(
        embed_dim=512,
        num_heads=8,
        dropout=0.1,
    )

    # Create input tensors with different sequence lengths
    batch_size = 4
    seq_len_q = 64  # Query sequence length
    seq_len_kv = 128  # Key/Value sequence length

    q = torch.randn(batch_size, seq_len_q, 512)
    k = torch.randn(batch_size, seq_len_kv, 512)
    v = torch.randn(batch_size, seq_len_kv, 512)

    # Forward pass
    output, weights = attention(q, k, v, need_weights=True)

    print(f"Query shape: {q.shape}")
    print(f"Key shape: {k.shape}")
    print(f"Value shape: {v.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
    print()


def example_with_masking():
    """Example of using attention with masking."""
    print("=" * 60)
    print("Example 5: Attention with Masking")
    print("=" * 60)

    # Create causal attention module
    attention = DenseAttention(
        embed_dim=512,
        num_heads=8,
        dropout=0.1,
        causal=True,  # Enable causal masking
    )

    # Create input tensors
    batch_size, seq_len = 4, 64
    q = torch.randn(batch_size, seq_len, 512)
    k = torch.randn(batch_size, seq_len, 512)
    v = torch.randn(batch_size, seq_len, 512)

    # Forward pass with causal mask
    output, weights = attention(q, k, v, need_weights=True)

    print(f"Input shape: {q.shape}")
    print(f"Output shape: {output.shape}")
    print("Causal masking: Enabled")

    # Custom attention mask
    attention_no_causal = DenseAttention(embed_dim=512, num_heads=8, dropout=0.1)

    # Create a custom mask (attend only to first half of sequence)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    mask[:, seq_len // 2 :] = False

    output_masked, _ = attention_no_causal(q, k, v, attention_mask=mask)
    print(f"Custom mask shape: {mask.shape}")
    print(f"Masked output shape: {output_masked.shape}")
    print()


if __name__ == "__main__":
    # Run all examples
    example_dense_attention()
    example_adaptive_precision()
    example_adaptive_rank()
    example_cross_attention()
    example_with_masking()

    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
