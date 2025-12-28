"""Tests for dense attention."""

import pytest
import torch
import torch.nn as nn

from adaattn.attention.base import AttentionConfig
from adaattn.attention.dense import DenseAttention


class TestDenseAttention:
    """Test suite for DenseAttention."""

    @pytest.fixture
    def attention(self):
        """Create a dense attention module."""
        return DenseAttention(embed_dim=64, num_heads=4, dropout=0.1)

    def test_initialization(self, attention):
        """Test module initialization."""
        assert attention.embed_dim == 64
        assert attention.num_heads == 4
        assert attention.head_dim == 16
        assert attention.kdim == 64
        assert attention.vdim == 64

    def test_forward_shape(self, attention):
        """Test forward pass output shape."""
        batch_size, seq_len = 2, 32
        q = torch.randn(batch_size, seq_len, 64)
        k = torch.randn(batch_size, seq_len, 64)
        v = torch.randn(batch_size, seq_len, 64)

        output, weights = attention(q, k, v)

        assert output.shape == (batch_size, seq_len, 64)
        assert weights is None

    def test_forward_with_weights(self, attention):
        """Test forward pass with attention weights."""
        batch_size, seq_len = 2, 16
        q = torch.randn(batch_size, seq_len, 64)
        k = torch.randn(batch_size, seq_len, 64)
        v = torch.randn(batch_size, seq_len, 64)

        output, weights = attention(q, k, v, need_weights=True)

        assert output.shape == (batch_size, seq_len, 64)
        assert weights is not None
        # Weights are averaged across heads
        assert weights.shape == (batch_size, seq_len, seq_len)

    def test_causal_masking(self):
        """Test causal masking."""
        attention = DenseAttention(embed_dim=64, num_heads=4, causal=True)

        batch_size, seq_len = 2, 8
        q = torch.randn(batch_size, seq_len, 64)
        k = torch.randn(batch_size, seq_len, 64)
        v = torch.randn(batch_size, seq_len, 64)

        output, _ = attention(q, k, v)

        assert output.shape == (batch_size, seq_len, 64)

    def test_attention_mask(self, attention):
        """Test with attention mask."""
        batch_size, seq_len = 2, 16
        q = torch.randn(batch_size, seq_len, 64)
        k = torch.randn(batch_size, seq_len, 64)
        v = torch.randn(batch_size, seq_len, 64)

        # Create mask
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        mask[:, seq_len // 2 :] = False

        output, _ = attention(q, k, v, attention_mask=mask)

        assert output.shape == (batch_size, seq_len, 64)

    def test_different_kv_dims(self):
        """Test with different key/value dimensions."""
        attention = DenseAttention(
            embed_dim=64,
            num_heads=4,
            kdim=32,
            vdim=32,
        )

        batch_size, seq_len = 2, 16
        q = torch.randn(batch_size, seq_len, 64)
        k = torch.randn(batch_size, seq_len, 32)
        v = torch.randn(batch_size, seq_len, 32)

        output, _ = attention(q, k, v)

        assert output.shape == (batch_size, seq_len, 64)

    def test_gradient_flow(self, attention):
        """Test gradient flow."""
        batch_size, seq_len = 2, 16
        q = torch.randn(batch_size, seq_len, 64, requires_grad=True)
        k = torch.randn(batch_size, seq_len, 64, requires_grad=True)
        v = torch.randn(batch_size, seq_len, 64, requires_grad=True)

        output, _ = attention(q, k, v)
        loss = output.sum()
        loss.backward()

        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None

    def test_dropout(self):
        """Test dropout during training."""
        attention = DenseAttention(embed_dim=64, num_heads=4, dropout=0.5)
        attention.train()

        batch_size, seq_len = 2, 16
        q = torch.randn(batch_size, seq_len, 64)
        k = torch.randn(batch_size, seq_len, 64)
        v = torch.randn(batch_size, seq_len, 64)

        output1, _ = attention(q, k, v)
        output2, _ = attention(q, k, v)

        # Outputs should differ due to dropout
        assert not torch.allclose(output1, output2)

        # Eval mode should be deterministic
        attention.eval()
        output3, _ = attention(q, k, v)
        output4, _ = attention(q, k, v)
        assert torch.allclose(output3, output4)

    def test_no_bias(self):
        """Test without bias."""
        attention = DenseAttention(embed_dim=64, num_heads=4, bias=False)

        assert attention.q_proj.bias is None
        assert attention.k_proj.bias is None
        assert attention.v_proj.bias is None
        assert attention.out_proj.bias is None

    def test_with_bias_kv(self):
        """Test with key/value bias."""
        attention = DenseAttention(
            embed_dim=64,
            num_heads=4,
            add_bias_kv=True,
        )

        assert attention.bias_k is not None
        assert attention.bias_v is not None

        batch_size, seq_len = 2, 16
        q = torch.randn(batch_size, seq_len, 64)
        k = torch.randn(batch_size, seq_len, 64)
        v = torch.randn(batch_size, seq_len, 64)

        output, _ = attention(q, k, v)
        assert output.shape == (batch_size, seq_len, 64)

    def test_cross_attention(self, attention):
        """Test cross-attention with different seq lengths."""
        batch_size = 2
        seq_q, seq_k = 16, 32

        q = torch.randn(batch_size, seq_q, 64)
        k = torch.randn(batch_size, seq_k, 64)
        v = torch.randn(batch_size, seq_k, 64)

        output, _ = attention(q, k, v)

        assert output.shape == (batch_size, seq_q, 64)

    def test_config_override(self):
        """Test with AttentionConfig override."""
        config = AttentionConfig(
            embed_dim=128,
            num_heads=8,
            dropout=0.2,
        )

        attention = DenseAttention(config=config)

        assert attention.embed_dim == 128
        assert attention.num_heads == 8
        assert attention.head_dim == 16

    def test_batch_size_one(self, attention):
        """Test with batch size 1."""
        q = torch.randn(1, 16, 64)
        k = torch.randn(1, 16, 64)
        v = torch.randn(1, 16, 64)

        output, _ = attention(q, k, v)
        assert output.shape == (1, 16, 64)

    def test_large_sequence(self, attention):
        """Test with large sequence length."""
        batch_size, seq_len = 2, 512
        q = torch.randn(batch_size, seq_len, 64)
        k = torch.randn(batch_size, seq_len, 64)
        v = torch.randn(batch_size, seq_len, 64)

        output, _ = attention(q, k, v)
        assert output.shape == (batch_size, seq_len, 64)

    def test_attention_weights_sum_to_one(self):
        """Test that attention weights sum to 1."""
        # Use eval mode and no dropout to ensure exact sum
        attention = DenseAttention(embed_dim=64, num_heads=4, dropout=0.0)
        attention.eval()

        batch_size, seq_len = 2, 8
        q = torch.randn(batch_size, seq_len, 64)
        k = torch.randn(batch_size, seq_len, 64)
        v = torch.randn(batch_size, seq_len, 64)

        output, weights = attention(q, k, v, need_weights=True)

        # Check weights sum to 1 along last dimension
        weights_sum = weights.sum(dim=-1)
        assert torch.allclose(weights_sum, torch.ones_like(weights_sum), atol=1e-5)
