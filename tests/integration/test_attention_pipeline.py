"""
Integration tests for full attention pipeline.
"""

import pytest
import torch

from adaattn.attention.adaptive_precision import AdaptivePrecisionAttention
from adaattn.attention.base import AttentionConfig, PrecisionMode


class TestAttentionPipeline:
    """Integration tests for complete attention pipeline."""

    def test_end_to_end_forward_pass(self):
        """Test complete forward pass through attention."""
        config = AttentionConfig(
            embed_dim=512,
            num_heads=8,
            dropout=0.1,
        )
        attn = AdaptivePrecisionAttention(config=config)
        attn.eval()

        batch_size, seq_len = 4, 64
        x = torch.randn(batch_size, seq_len, 512)

        # Forward pass
        output, weights = attn(x, x, x)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_backward_pass_gradients(self):
        """Test backward pass and gradient computation."""
        attn = AdaptivePrecisionAttention(embed_dim=256, num_heads=8)

        batch_size, seq_len = 2, 32
        q = torch.randn(batch_size, seq_len, 256, requires_grad=True)
        k = torch.randn(batch_size, seq_len, 256, requires_grad=True)
        v = torch.randn(batch_size, seq_len, 256, requires_grad=True)

        output, _ = attn(q, k, v)
        loss = output.mean()
        loss.backward()

        # Verify gradients exist and are valid
        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None

        assert not torch.isnan(q.grad).any()
        assert not torch.isnan(k.grad).any()
        assert not torch.isnan(v.grad).any()

    def test_batch_processing(self):
        """Test processing multiple batches."""
        attn = AdaptivePrecisionAttention(embed_dim=128, num_heads=4)
        attn.eval()

        outputs = []
        for batch_size in [1, 2, 4, 8]:
            seq_len = 16
            x = torch.randn(batch_size, seq_len, 128)
            output, _ = attn(x, x, x)

            assert output.shape == (batch_size, seq_len, 128)
            outputs.append(output)

    def test_variable_sequence_lengths(self):
        """Test with varying sequence lengths."""
        attn = AdaptivePrecisionAttention(embed_dim=128, num_heads=4)
        attn.eval()

        batch_size = 2
        for seq_len in [8, 16, 32, 64, 128]:
            x = torch.randn(batch_size, seq_len, 128)
            output, _ = attn(x, x, x)

            assert output.shape == (batch_size, seq_len, 128)

    def test_cross_attention(self):
        """Test cross-attention with different query and key sequences."""
        attn = AdaptivePrecisionAttention(embed_dim=256, num_heads=8)
        attn.eval()

        batch_size = 2
        seq_q, seq_k = 32, 64

        q = torch.randn(batch_size, seq_q, 256)
        k = torch.randn(batch_size, seq_k, 256)
        v = torch.randn(batch_size, seq_k, 256)

        output, weights = attn(q, k, v)

        assert output.shape == (batch_size, seq_q, 256)
        if weights is not None:
            assert weights.shape == (batch_size, 8, seq_q, seq_k)

    def test_numerical_stability_large_inputs(self):
        """Test numerical stability with large input values."""
        attn = AdaptivePrecisionAttention(embed_dim=128, num_heads=4)
        attn.eval()

        batch_size, seq_len = 2, 32
        # Large input values
        x = torch.randn(batch_size, seq_len, 128) * 10

        output, _ = attn(x, x, x)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_numerical_stability_small_inputs(self):
        """Test numerical stability with small input values."""
        attn = AdaptivePrecisionAttention(embed_dim=128, num_heads=4)
        attn.eval()

        batch_size, seq_len = 2, 32
        # Small input values
        x = torch.randn(batch_size, seq_len, 128) * 0.01

        output, _ = attn(x, x, x)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_training_mode_consistency(self):
        """Test consistency between training and eval modes."""
        attn = AdaptivePrecisionAttention(embed_dim=128, num_heads=4, dropout=0.0)

        batch_size, seq_len = 2, 32
        x = torch.randn(batch_size, seq_len, 128)

        # Eval mode
        attn.eval()
        output_eval, _ = attn(x, x, x)

        # Train mode
        attn.train()
        output_train, _ = attn(x, x, x)

        # Without dropout, should be very similar
        assert torch.allclose(output_eval, output_train, rtol=1e-4)

    def test_attention_weights_sum_to_one(self):
        """Test that attention weights sum to 1 along key dimension."""
        attn = AdaptivePrecisionAttention(embed_dim=128, num_heads=4)
        attn.eval()

        batch_size, seq_len = 2, 16
        x = torch.randn(batch_size, seq_len, 128)

        output, weights = attn(x, x, x)

        if weights is not None:
            # Sum along key dimension should be 1
            weight_sums = weights.sum(dim=-1)
            assert torch.allclose(weight_sums, torch.ones_like(weight_sums), rtol=1e-5)

    def test_multiple_precision_modes(self):
        """Test attention with different precision modes."""
        batch_size, seq_len = 2, 32
        x = torch.randn(batch_size, seq_len, 128)

        for precision in [PrecisionMode.FP32, PrecisionMode.FP16, PrecisionMode.BF16]:
            config = AttentionConfig(
                embed_dim=128,
                num_heads=4,
                precision=precision,
            )
            attn = AdaptivePrecisionAttention(config=config)
            attn.eval()

            output, _ = attn(x, x, x)

            assert output.shape == x.shape
            assert not torch.isnan(output).any()


class TestMemoryEfficiency:
    """Tests for memory efficiency."""

    def test_memory_usage_scales_linearly(self):
        """Test that memory usage scales linearly with sequence length."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        attn = AdaptivePrecisionAttention(embed_dim=256, num_heads=8).cuda()
        attn.eval()

        memory_usage = []
        for seq_len in [64, 128, 256]:
            torch.cuda.reset_peak_memory_stats()

            x = torch.randn(2, seq_len, 256, device="cuda")
            with torch.no_grad():
                output, _ = attn(x, x, x)

            mem = torch.cuda.max_memory_allocated()
            memory_usage.append(mem)

        # Memory should increase but not quadratically
        ratio_1 = memory_usage[1] / memory_usage[0]
        ratio_2 = memory_usage[2] / memory_usage[1]

        # Should be roughly linear (2x seq -> ~2x memory)
        assert ratio_1 < 3.0
        assert ratio_2 < 3.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
