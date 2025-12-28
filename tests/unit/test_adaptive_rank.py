"""Tests for adaptive rank attention."""

import pytest
import torch
import torch.nn as nn

from adaattn.attention.adaptive_rank import AdaptiveRankAttention
from adaattn.attention.base import AttentionConfig


class TestAdaptiveRankAttention:
    """Test suite for AdaptiveRankAttention."""

    @pytest.fixture
    def attention(self):
        """Create an adaptive rank attention module."""
        return AdaptiveRankAttention(
            embed_dim=64,
            num_heads=4,
            dropout=0.1,
            rank_ratio=0.5,
        )

    def test_initialization(self, attention):
        """Test module initialization."""
        assert attention.embed_dim == 64
        assert attention.num_heads == 4
        assert attention.head_dim == 16
        assert attention.rank_ratio == 0.5
        assert attention.min_rank == 1
        assert attention.rank_estimation_method == "entropy"

    def test_forward_shape(self, attention):
        """Test forward pass output shape."""
        batch_size, seq_len = 2, 32
        q = torch.randn(batch_size, seq_len, 64)
        k = torch.randn(batch_size, seq_len, 64)
        v = torch.randn(batch_size, seq_len, 64)

        output, weights = attention(q, k, v)

        assert output.shape == (batch_size, seq_len, 64)
        assert weights is None  # Default behavior

    def test_different_seq_lengths(self, attention):
        """Test with different query and key/value sequence lengths."""
        batch_size = 2
        seq_q, seq_k = 16, 32

        q = torch.randn(batch_size, seq_q, 64)
        k = torch.randn(batch_size, seq_k, 64)
        v = torch.randn(batch_size, seq_k, 64)

        output, _ = attention(q, k, v)

        assert output.shape == (batch_size, seq_q, 64)

    def test_causal_masking(self):
        """Test causal masking."""
        attention = AdaptiveRankAttention(
            embed_dim=64,
            num_heads=4,
            causal=True,
        )

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

        # Create mask (batch, seq_len)
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        mask[:, seq_len // 2 :] = False

        output, _ = attention(q, k, v, attention_mask=mask)

        assert output.shape == (batch_size, seq_len, 64)

    def test_rank_estimation_methods(self):
        """Test different rank estimation methods."""
        methods = ["entropy", "power", "random"]

        for method in methods:
            attention = AdaptiveRankAttention(
                embed_dim=64,
                num_heads=4,
                rank_estimation_method=method,
            )

            batch_size, seq_len = 2, 16
            q = torch.randn(batch_size, seq_len, 64)
            k = torch.randn(batch_size, seq_len, 64)
            v = torch.randn(batch_size, seq_len, 64)

            output, _ = attention(q, k, v)

            assert output.shape == (batch_size, seq_len, 64)

    def test_rank_statistics(self, attention):
        """Test rank statistics tracking."""
        batch_size, seq_len = 2, 16
        q = torch.randn(batch_size, seq_len, 64)
        k = torch.randn(batch_size, seq_len, 64)
        v = torch.randn(batch_size, seq_len, 64)

        # Initial stats
        stats = attention.get_rank_statistics()
        assert stats["call_count"] == 0

        # Run forward pass
        _ = attention(q, k, v)

        # Check updated stats
        stats = attention.get_rank_statistics()
        assert stats["call_count"] == 1
        assert len(stats["low_rank_usage_per_head"]) == 4
        assert 0 <= stats["avg_low_rank_usage"] <= 1

    def test_gradient_flow(self, attention):
        """Test gradient flow through the module."""
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
        assert not torch.isnan(q.grad).any()
        assert not torch.isnan(k.grad).any()
        assert not torch.isnan(v.grad).any()

    def test_dropout(self):
        """Test dropout during training."""
        attention = AdaptiveRankAttention(
            embed_dim=64,
            num_heads=4,
            dropout=0.5,
        )
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

    def test_adaptive_threshold(self):
        """Test different adaptive thresholds."""
        thresholds = [0.5, 0.75, 0.95]

        for threshold in thresholds:
            attention = AdaptiveRankAttention(
                embed_dim=64,
                num_heads=4,
                adaptive_threshold=threshold,
            )

            batch_size, seq_len = 2, 16
            q = torch.randn(batch_size, seq_len, 64)
            k = torch.randn(batch_size, seq_len, 64)
            v = torch.randn(batch_size, seq_len, 64)

            output, _ = attention(q, k, v)
            assert output.shape == (batch_size, seq_len, 64)

    def test_rank_ratio(self):
        """Test different rank ratios."""
        ratios = [0.25, 0.5, 0.75]

        for ratio in ratios:
            attention = AdaptiveRankAttention(
                embed_dim=64,
                num_heads=4,
                rank_ratio=ratio,
            )

            batch_size, seq_len = 2, 16
            q = torch.randn(batch_size, seq_len, 64)
            k = torch.randn(batch_size, seq_len, 64)
            v = torch.randn(batch_size, seq_len, 64)

            output, _ = attention(q, k, v)
            assert output.shape == (batch_size, seq_len, 64)

    def test_min_rank(self):
        """Test minimum rank constraint."""
        attention = AdaptiveRankAttention(
            embed_dim=64,
            num_heads=4,
            min_rank=4,
        )

        assert attention.min_rank == 4

    def test_config_override(self):
        """Test with AttentionConfig override."""
        config = AttentionConfig(
            embed_dim=128,
            num_heads=8,
            dropout=0.2,
            enable_adaptive_rank=True,
        )

        attention = AdaptiveRankAttention(config=config)

        assert attention.embed_dim == 128
        assert attention.num_heads == 8
        assert attention.head_dim == 16
        assert attention.config.dropout == 0.2

    def test_extra_repr(self, attention):
        """Test string representation."""
        repr_str = attention.extra_repr()
        assert "rank_ratio" in repr_str
        assert "method" in repr_str

    def test_batch_size_one(self, attention):
        """Test with batch size 1."""
        q = torch.randn(1, 16, 64)
        k = torch.randn(1, 16, 64)
        v = torch.randn(1, 16, 64)

        output, _ = attention(q, k, v)
        assert output.shape == (1, 16, 64)

    def test_large_sequence(self, attention):
        """Test with large sequence length."""
        batch_size, seq_len = 2, 256
        q = torch.randn(batch_size, seq_len, 64)
        k = torch.randn(batch_size, seq_len, 64)
        v = torch.randn(batch_size, seq_len, 64)

        output, _ = attention(q, k, v)
        assert output.shape == (batch_size, seq_len, 64)
