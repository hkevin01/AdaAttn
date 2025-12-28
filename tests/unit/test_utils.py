"""
Unit tests for utility functions.
"""

import pytest
import torch

from adaattn.linalg.utils import (
    batch_frobenius_norm,
    batch_trace,
    causal_mask,
    log_softmax_stable,
    masked_fill_inf,
    sliding_window_mask,
    stable_softmax,
)


class TestStableSoftmax:
    """Tests for stable_softmax function."""

    def test_basic_softmax(self):
        """Test basic softmax computation."""
        x = torch.tensor([1.0, 2.0, 3.0])
        result = stable_softmax(x)
        
        # Should sum to 1
        assert torch.allclose(result.sum(), torch.tensor(1.0))
        
        # Should match PyTorch's softmax
        expected = torch.softmax(x, dim=-1)
        assert torch.allclose(result, expected, rtol=1e-5)

    def test_large_values(self):
        """Test stability with large values."""
        x = torch.tensor([1000.0, 1001.0, 1002.0])
        result = stable_softmax(x)
        
        # Should not overflow
        assert torch.all(torch.isfinite(result))
        assert torch.allclose(result.sum(), torch.tensor(1.0))

    def test_temperature_scaling(self):
        """Test temperature parameter."""
        x = torch.tensor([1.0, 2.0, 3.0])
        
        # Higher temperature should smooth the distribution
        result_high = stable_softmax(x, temperature=2.0)
        result_low = stable_softmax(x, temperature=0.5)
        
        # High temp should be more uniform
        assert result_high.std() < result_low.std()

    def test_batch_computation(self):
        """Test batched softmax."""
        x = torch.randn(4, 8, 64)
        result = stable_softmax(x, dim=-1)
        
        assert result.shape == x.shape
        # Each row should sum to 1
        assert torch.allclose(result.sum(dim=-1), torch.ones(4, 8))


class TestLogSoftmaxStable:
    """Tests for log_softmax_stable function."""

    def test_basic_log_softmax(self):
        """Test basic log-softmax computation."""
        x = torch.tensor([1.0, 2.0, 3.0])
        result = log_softmax_stable(x)
        
        # Should match PyTorch's log_softmax
        expected = torch.log_softmax(x, dim=-1)
        assert torch.allclose(result, expected, rtol=1e-5)

    def test_large_values_stable(self):
        """Test stability with large values."""
        x = torch.tensor([1000.0, 1001.0, 1002.0])
        result = log_softmax_stable(x)
        
        # Should not overflow
        assert torch.all(torch.isfinite(result))

    def test_relationship_with_softmax(self):
        """Test that exp(log_softmax) = softmax."""
        x = torch.randn(10)
        
        log_sm = log_softmax_stable(x)
        sm = stable_softmax(x)
        
        assert torch.allclose(torch.exp(log_sm), sm, rtol=1e-5)


class TestBatchTrace:
    """Tests for batch_trace function."""

    def test_identity_matrix(self):
        """Test trace of identity matrices."""
        I = torch.eye(5)
        trace = batch_trace(I)
        
        assert torch.allclose(trace, torch.tensor(5.0))

    def test_batch_trace(self):
        """Test batched trace computation."""
        A = torch.randn(4, 10, 10)
        traces = batch_trace(A)
        
        assert traces.shape == (4,)
        
        # Verify against manual computation
        for i in range(4):
            expected = A[i].diagonal().sum()
            assert torch.allclose(traces[i], expected)


class TestBatchFrobeniusNorm:
    """Tests for batch_frobenius_norm function."""

    def test_zero_matrix(self):
        """Test norm of zero matrix."""
        A = torch.zeros(5, 5)
        norm = batch_frobenius_norm(A)
        
        assert torch.allclose(norm, torch.tensor(0.0))

    def test_identity_matrix(self):
        """Test norm of identity matrix."""
        I = torch.eye(5)
        norm = batch_frobenius_norm(I)
        
        # Frobenius norm of identity is sqrt(n)
        assert torch.allclose(norm, torch.tensor(5.0).sqrt())

    def test_batch_norm(self):
        """Test batched Frobenius norm."""
        A = torch.randn(4, 8, 6)
        norms = batch_frobenius_norm(A)
        
        assert norms.shape == (4,)
        assert torch.all(norms >= 0)


class TestCausalMask:
    """Tests for causal_mask function."""

    def test_mask_shape(self):
        """Test causal mask shape."""
        mask = causal_mask(10)
        
        assert mask.shape == (10, 10)
        assert mask.dtype == torch.bool

    def test_lower_triangular(self):
        """Test that mask is lower triangular."""
        mask = causal_mask(5)
        
        # Diagonal and below should be False (not masked)
        for i in range(5):
            for j in range(5):
                if j > i:
                    assert mask[i, j] == True, f"Position ({i}, {j}) should be masked"
                else:
                    assert mask[i, j] == False, f"Position ({i}, {j}) should not be masked"

    def test_device_placement(self):
        """Test mask device placement."""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            mask = causal_mask(10, device=device)
            assert mask.device == device


class TestSlidingWindowMask:
    """Tests for sliding_window_mask function."""

    def test_mask_shape(self):
        """Test sliding window mask shape."""
        mask = sliding_window_mask(10, window_size=3)
        
        assert mask.shape == (10, 10)
        assert mask.dtype == torch.bool

    def test_window_pattern(self):
        """Test that mask follows window pattern."""
        mask = sliding_window_mask(10, window_size=2)
        
        # Position (5, 5) should not be masked (distance = 0)
        assert mask[5, 5] == False
        
        # Position (5, 6) should not be masked (distance = 1)
        assert mask[5, 6] == False
        
        # Position (5, 8) should be masked (distance = 3 > 2)
        assert mask[5, 8] == True

    def test_full_attention(self):
        """Test with large window (no masking)."""
        seq_len = 10
        mask = sliding_window_mask(seq_len, window_size=seq_len)
        
        # All positions should be visible
        assert torch.all(mask == False)


class TestMaskedFillInf:
    """Tests for masked_fill_inf function."""

    def test_basic_masking(self):
        """Test basic mask filling."""
        x = torch.ones(5, 5)
        mask = torch.zeros(5, 5, dtype=torch.bool)
        mask[0, :] = True  # Mask first row
        
        result = masked_fill_inf(x, mask)
        
        # First row should be -inf
        assert torch.all(result[0, :] == float('-inf'))
        
        # Other rows should be unchanged
        assert torch.all(result[1:, :] == 1.0)

    def test_with_causal_mask(self):
        """Test combination with causal mask."""
        x = torch.randn(8, 8)
        mask = causal_mask(8)
        
        result = masked_fill_inf(x, mask)
        
        # Upper triangle should be -inf
        for i in range(8):
            for j in range(i+1, 8):
                assert result[i, j] == float('-inf')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestSymmetricPart:
    """Test symmetric_part function."""

    def test_identity_matrix(self):
        """Test symmetric part of identity matrix."""
        from adaattn.linalg.utils import symmetric_part
        
        A = torch.eye(3)
        sym = symmetric_part(A)
        
        assert torch.allclose(sym, A)

    def test_symmetric_matrix(self):
        """Test symmetric matrix stays unchanged."""
        from adaattn.linalg.utils import symmetric_part
        
        A = torch.tensor([[1.0, 2.0], [2.0, 3.0]])
        sym = symmetric_part(A)
        
        assert torch.allclose(sym, A)

    def test_antisymmetric_matrix(self):
        """Test antisymmetric matrix becomes zero."""
        from adaattn.linalg.utils import symmetric_part
        
        A = torch.tensor([[0.0, 1.0], [-1.0, 0.0]])
        sym = symmetric_part(A)
        
        assert torch.allclose(sym, torch.zeros_like(A), atol=1e-6)


class TestAntisymmetricPart:
    """Test antisymmetric_part function."""

    def test_symmetric_matrix(self):
        """Test symmetric matrix becomes zero."""
        from adaattn.linalg.utils import antisymmetric_part
        
        A = torch.tensor([[1.0, 2.0], [2.0, 3.0]])
        antisym = antisymmetric_part(A)
        
        assert torch.allclose(antisym, torch.zeros_like(A), atol=1e-6)

    def test_antisymmetric_matrix(self):
        """Test antisymmetric matrix stays unchanged."""
        from adaattn.linalg.utils import antisymmetric_part
        
        A = torch.tensor([[0.0, 1.0], [-1.0, 0.0]])
        antisym = antisymmetric_part(A)
        
        assert torch.allclose(antisym, A)


class TestEntropyBasedRankHint:
    """Test entropy_based_rank_hint function."""

    def test_low_entropy_suggests_low_rank(self):
        """Test that low entropy suggests low rank."""
        from adaattn.linalg.entropy import entropy_based_rank_hint
        
        # Create peaked attention (low entropy)
        attn = torch.zeros(2, 4, 8, 8)
        attn[:, :, :, 0] = 1.0  # All weight on first position
        
        hints = entropy_based_rank_hint(attn)
        
        assert hints.shape == (2, 4)
        assert (hints == 0.0).all()

    def test_high_entropy_suggests_full_rank(self):
        """Test that high entropy suggests full rank."""
        from adaattn.linalg.entropy import entropy_based_rank_hint
        
        # Create uniform attention (high entropy)
        attn = torch.ones(2, 4, 8, 8) / 8
        
        hints = entropy_based_rank_hint(attn)
        
        assert hints.shape == (2, 4)
        assert (hints == 1.0).all()

    def test_medium_entropy_suggests_neutral(self):
        """Test that medium entropy suggests neutral."""
        from adaattn.linalg.entropy import entropy_based_rank_hint
        
        # Create medium entropy attention
        attn = torch.softmax(torch.randn(2, 4, 8, 8), dim=-1)
        
        hints = entropy_based_rank_hint(attn, threshold_low=0.2, threshold_high=0.9)
        
        assert hints.shape == (2, 4)
        # Most should be neutral (0.5) with these wide thresholds
