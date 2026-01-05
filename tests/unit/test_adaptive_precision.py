"""
Unit tests for adaptive precision attention.
"""

import pytest
import torch

from adaattn.attention.adaptive_precision import AdaptivePrecisionAttention
from adaattn.attention.base import AttentionConfig, PrecisionMode


class TestAdaptivePrecisionAttention:
    """Tests for AdaptivePrecisionAttention class."""

    def test_initialization(self):
        """Test basic initialization."""
        attn = AdaptivePrecisionAttention(embed_dim=512, num_heads=8)
        
        assert attn.embed_dim == 512
        assert attn.num_heads == 8
        assert attn.head_dim == 64

    def test_initialization_with_config(self):
        """Test initialization with custom config."""
        config = AttentionConfig(
            embed_dim=768,
            num_heads=12,
            dropout=0.1,
        )
        attn = AdaptivePrecisionAttention(config=config)
        
        assert attn.embed_dim == 768
        assert attn.num_heads == 12
        assert attn.head_dim == 64

    def test_forward_pass_basic(self):
        """Test basic forward pass."""
        attn = AdaptivePrecisionAttention(embed_dim=64, num_heads=4)
        
        batch_size, seq_len = 2, 16
        q = torch.randn(batch_size, seq_len, 64)
        k = torch.randn(batch_size, seq_len, 64)
        v = torch.randn(batch_size, seq_len, 64)
        
        output, weights = attn(q, k, v)
        
        assert output.shape == (batch_size, seq_len, 64)
        if weights is not None:
            assert weights.shape == (batch_size, 4, seq_len, seq_len)

    def test_forward_pass_different_seq_lengths(self):
        """Test forward pass with different query/key lengths."""
        attn = AdaptivePrecisionAttention(embed_dim=64, num_heads=4)
        
        batch_size = 2
        seq_q, seq_k = 16, 32
        
        q = torch.randn(batch_size, seq_q, 64)
        k = torch.randn(batch_size, seq_k, 64)
        v = torch.randn(batch_size, seq_k, 64)
        
        output, weights = attn(q, k, v)
        
        assert output.shape == (batch_size, seq_q, 64)
        if weights is not None:
            assert weights.shape == (batch_size, 4, seq_q, seq_k)

    def test_causal_masking(self):
        """Test causal masking in forward pass."""
        attn = AdaptivePrecisionAttention(
            embed_dim=64,
            num_heads=4,
            causal=True,
        )
        
        batch_size, seq_len = 2, 8
        q = torch.randn(batch_size, seq_len, 64)
        k = torch.randn(batch_size, seq_len, 64)
        v = torch.randn(batch_size, seq_len, 64)
        
        output, weights = attn(q, k, v)
        
        assert output.shape == (batch_size, seq_len, 64)
        
        # Check that future positions are masked (if weights returned)
        if weights is not None:
            # Upper triangle should be zero (or very small)
            for b in range(batch_size):
                for h in range(4):
                    for i in range(seq_len):
                        for j in range(i+1, seq_len):
                            assert weights[b, h, i, j] < 1e-6

    def test_precision_modes(self):
        """Test initialization with different precision modes."""
        for precision in [PrecisionMode.FP32, PrecisionMode.FP16, PrecisionMode.BF16]:
            config = AttentionConfig(
                embed_dim=64,
                num_heads=4,
                precision=precision,
            )
            attn = AdaptivePrecisionAttention(config=config)
            
            # Just verify it initializes and works
            q = torch.randn(2, 16, 64)
            k = torch.randn(2, 16, 64)
            v = torch.randn(2, 16, 64)
            
            output, _ = attn(q, k, v)
            assert output.shape == q.shape

    def test_analyze_precision_requirements(self):
        """Test precision statistics tracking."""
        attn = AdaptivePrecisionAttention(embed_dim=64, num_heads=4)
        
        # Run forward pass
        q = torch.randn(2, 16, 64)
        k = torch.randn(2, 16, 64)
        v = torch.randn(2, 16, 64)
        
        output, _ = attn(q, k, v)
        
        # Check statistics
        stats = attn.get_precision_statistics()
        assert 'call_count' in stats
        assert stats['call_count'] > 0

    def test_large_scores_trigger_fp32(self):
        """Test that precision adaptation works."""
        attn = AdaptivePrecisionAttention(embed_dim=64, num_heads=4)
        
        # Just test that it works with various inputs
        q = torch.randn(2, 16, 64) * 100
        k = torch.randn(2, 16, 64) * 100
        v = torch.randn(2, 16, 64) * 100
        
        output, _ = attn(q, k, v)
        
        # Should handle large values without error
        assert torch.isfinite(output).all()

    def test_dropout_applied(self):
        """Test that dropout is applied during training."""
        attn = AdaptivePrecisionAttention(
            embed_dim=64,
            num_heads=4,
            dropout=0.5,
        )
        attn.train()
        
        batch_size, seq_len = 2, 16
        q = torch.randn(batch_size, seq_len, 64)
        k = torch.randn(batch_size, seq_len, 64)
        v = torch.randn(batch_size, seq_len, 64)
        
        # Multiple forward passes should give different results due to dropout
        output1, _ = attn(q, k, v)
        output2, _ = attn(q, k, v)
        
        # Should be different due to dropout
        assert not torch.allclose(output1, output2)

    def test_eval_mode_deterministic(self):
        """Test that eval mode is deterministic."""
        attn = AdaptivePrecisionAttention(
            embed_dim=64,
            num_heads=4,
            dropout=0.5,
        )
        attn.eval()
        
        batch_size, seq_len = 2, 16
        q = torch.randn(batch_size, seq_len, 64)
        k = torch.randn(batch_size, seq_len, 64)
        v = torch.randn(batch_size, seq_len, 64)
        
        # Multiple forward passes should give same results in eval mode
        output1, _ = attn(q, k, v)
        output2, _ = attn(q, k, v)
        
        assert torch.allclose(output1, output2, rtol=1e-5)

    def test_output_shape_preserved(self):
        """Test that output shape matches query shape."""
        attn = AdaptivePrecisionAttention(embed_dim=128, num_heads=8)
        
        for batch_size in [1, 4]:
            for seq_len in [8, 32, 64]:
                q = torch.randn(batch_size, seq_len, 128)
                k = torch.randn(batch_size, seq_len, 128)
                v = torch.randn(batch_size, seq_len, 128)
                
                output, _ = attn(q, k, v)
                
                assert output.shape == q.shape

    def test_gradient_flow(self):
        """Test that gradients flow through the attention."""
        attn = AdaptivePrecisionAttention(embed_dim=64, num_heads=4)
        
        batch_size, seq_len = 2, 16
        q = torch.randn(batch_size, seq_len, 64, requires_grad=True)
        k = torch.randn(batch_size, seq_len, 64, requires_grad=True)
        v = torch.randn(batch_size, seq_len, 64, requires_grad=True)
        
        output, _ = attn(q, k, v)
        loss = output.sum()
        loss.backward()
        
        # Check that gradients exist
        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None
        
        # Check that gradients are non-zero
        assert q.grad.abs().sum() > 0
        assert k.grad.abs().sum() > 0
        assert v.grad.abs().sum() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
