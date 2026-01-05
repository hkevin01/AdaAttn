"""
Tests for unified AdaAttn implementation.
"""

import pytest
import torch
from adaattn.attention.adaattn import AdaAttention
from adaattn.attention.base import PrecisionMode


class TestAdaAttentionUnified:
    """Test suite for unified AdaAttention."""

    def test_initialization(self):
        """Test basic initialization."""
        attn = AdaAttention(embed_dim=256, num_heads=4)
        assert attn.config.embed_dim == 256
        assert attn.config.num_heads == 4
        assert attn.enable_adaptive_rank is True
        assert attn.enable_adaptive_precision is True

    def test_forward_shape(self):
        """Test forward pass output shape."""
        attn = AdaAttention(embed_dim=256, num_heads=4)
        q = torch.randn(2, 64, 256)
        k = torch.randn(2, 64, 256)
        v = torch.randn(2, 64, 256)
        
        output, _ = attn(q, k, v)
        assert output.shape == (2, 64, 256)

    def test_adaptive_rank_enabled(self):
        """Test with adaptive rank enabled."""
        attn = AdaAttention(
            embed_dim=256,
            num_heads=4,
            enable_adaptive_rank=True,
            rank_threshold=0.8
        )
        q = torch.randn(2, 64, 256)
        k = torch.randn(2, 64, 256)
        v = torch.randn(2, 64, 256)
        
        output, _ = attn(q, k, v)
        assert output.shape == q.shape
        
        stats = attn.get_statistics()
        assert 'low_rank_ratio' in stats
        assert 'call_count' in stats

    def test_adaptive_rank_disabled(self):
        """Test with adaptive rank disabled."""
        attn = AdaAttention(
            embed_dim=256,
            num_heads=4,
            enable_adaptive_rank=False
        )
        q = torch.randn(2, 64, 256)
        k = torch.randn(2, 64, 256)
        v = torch.randn(2, 64, 256)
        
        output, _ = attn(q, k, v)
        stats = attn.get_statistics()
        assert stats['low_rank_ratio'] == 0.0

    def test_adaptive_precision_enabled(self):
        """Test with adaptive precision enabled."""
        attn = AdaAttention(
            embed_dim=256,
            num_heads=4,
            enable_adaptive_precision=True
        )
        q = torch.randn(2, 64, 256)
        k = torch.randn(2, 64, 256)
        v = torch.randn(2, 64, 256)
        
        output, _ = attn(q, k, v)
        stats = attn.get_statistics()
        assert 'precision_distribution' in stats

    def test_different_precisions(self):
        """Test with different default precisions."""
        for precision in [PrecisionMode.FP32, PrecisionMode.FP16]:
            attn = AdaAttention(
                embed_dim=256,
                num_heads=4,
                default_precision=precision
            )
            q = torch.randn(2, 64, 256)
            k = torch.randn(2, 64, 256)
            v = torch.randn(2, 64, 256)
            
            output, _ = attn(q, k, v)
            assert output.shape == q.shape

    def test_causal_masking(self):
        """Test causal masking."""
        attn = AdaAttention(embed_dim=256, num_heads=4, causal=True)
        seq_len = 64
        q = torch.randn(2, seq_len, 256)
        k = torch.randn(2, seq_len, 256)
        v = torch.randn(2, seq_len, 256)
        
        output, _ = attn(q, k, v)
        assert output.shape == q.shape

    def test_attention_mask(self):
        """Test with attention mask."""
        attn = AdaAttention(embed_dim=256, num_heads=4)
        q = torch.randn(2, 64, 256)
        k = torch.randn(2, 64, 256)
        v = torch.randn(2, 64, 256)
        mask = torch.randn(2, 64, 64)
        
        output, _ = attn(q, k, v, attention_mask=mask)
        assert output.shape == q.shape

    def test_different_seq_lengths(self):
        """Test with different query and key sequence lengths."""
        attn = AdaAttention(embed_dim=256, num_heads=4)
        q = torch.randn(2, 32, 256)
        k = torch.randn(2, 64, 256)
        v = torch.randn(2, 64, 256)
        
        output, _ = attn(q, k, v)
        assert output.shape == (2, 32, 256)

    def test_statistics_tracking(self):
        """Test statistics tracking over multiple calls."""
        attn = AdaAttention(embed_dim=256, num_heads=4)
        q = torch.randn(2, 64, 256)
        k = torch.randn(2, 64, 256)
        v = torch.randn(2, 64, 256)
        
        # Multiple forward passes
        for _ in range(5):
            _ = attn(q, k, v)
        
        stats = attn.get_statistics()
        assert stats['call_count'] == 5
        
        # Check precision distribution sums to 1
        prec_sum = sum(stats['precision_distribution'].values())
        assert abs(prec_sum - 1.0) < 0.01

    def test_gradient_flow(self):
        """Test gradient flow through the module."""
        attn = AdaAttention(embed_dim=256, num_heads=4)
        q = torch.randn(2, 64, 256, requires_grad=True)
        k = torch.randn(2, 64, 256, requires_grad=True)
        v = torch.randn(2, 64, 256, requires_grad=True)
        
        output, _ = attn(q, k, v)
        loss = output.sum()
        loss.backward()
        
        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None

    def test_dropout(self):
        """Test dropout functionality."""
        torch.manual_seed(42)
        attn = AdaAttention(embed_dim=256, num_heads=4, dropout=0.1)
        attn.train()
        
        q = torch.randn(2, 64, 256)
        k = torch.randn(2, 64, 256)
        v = torch.randn(2, 64, 256)
        
        output1, _ = attn(q, k, v)
        output2, _ = attn(q, k, v)
        
        # Outputs should differ due to dropout
        assert not torch.allclose(output1, output2)
        
        # In eval mode, outputs should be deterministic
        attn.eval()
        output3, _ = attn(q, k, v)
        output4, _ = attn(q, k, v)
        assert torch.allclose(output3, output4)

    def test_batch_size_one(self):
        """Test with batch size of 1."""
        attn = AdaAttention(embed_dim=256, num_heads=4)
        q = torch.randn(1, 64, 256)
        k = torch.randn(1, 64, 256)
        v = torch.randn(1, 64, 256)
        
        output, _ = attn(q, k, v)
        assert output.shape == (1, 64, 256)

    def test_large_sequence(self):
        """Test with large sequence length."""
        attn = AdaAttention(embed_dim=256, num_heads=4)
        q = torch.randn(2, 512, 256)
        k = torch.randn(2, 512, 256)
        v = torch.randn(2, 512, 256)
        
        output, _ = attn(q, k, v)
        assert output.shape == (2, 512, 256)

    def test_extra_repr(self):
        """Test string representation."""
        attn = AdaAttention(embed_dim=256, num_heads=4)
        repr_str = str(attn)
        assert 'AdaAttention' in repr_str
        assert 'adaptive_rank' in repr_str
        assert 'adaptive_precision' in repr_str
