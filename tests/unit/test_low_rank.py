"""
Unit tests for low-rank approximation utilities.
"""

import pytest
import torch

from adaattn.linalg.low_rank import (
    estimate_effective_rank,
    low_rank_approx,
    power_iteration,
    randomized_svd,
    spectral_norm,
)


class TestPowerIteration:
    """Tests for power_iteration function."""

    def test_simple_matrix(self):
        """Test power iteration on simple matrix."""
        # Matrix with known largest singular value
        A = torch.tensor([[3.0, 0.0], [0.0, 1.0]])

        sigma, v = power_iteration(A, num_iterations=10)

        # Largest singular value should be 3.0
        assert torch.allclose(sigma.squeeze(), torch.tensor(3.0), rtol=1e-3)

    def test_batch_computation(self):
        """Test batched power iteration."""
        # Batch of matrices
        A = torch.randn(4, 10, 10)

        sigma, v = power_iteration(A, num_iterations=5)

        assert sigma.shape == (4,)
        assert v.shape == (4, 10)  # Squeezed output


class TestRandomizedSVD:
    """Tests for randomized_svd function."""

    def test_low_rank_approximation(self):
        """Test SVD on low-rank matrix."""
        # Create a rank-2 matrix
        U_true = torch.randn(10, 2)
        V_true = torch.randn(5, 2)
        A = U_true @ V_true.T

        U, S, Vt = randomized_svd(A, rank=2, num_oversampling=5)

        assert U.shape == (10, 2)
        assert S.shape == (2,)
        assert Vt.shape == (2, 5)

        # Reconstruction should be accurate
        A_approx = U @ torch.diag(S) @ Vt
        assert torch.allclose(A, A_approx, rtol=0.05)

    def test_batch_svd(self):
        """Test batched randomized SVD."""
        A = torch.randn(4, 8, 6)

        U, S, Vt = randomized_svd(A, rank=3)

        assert U.shape == (4, 8, 3)
        assert S.shape == (4, 3)
        assert Vt.shape == (4, 3, 6)


class TestEstimateEffectiveRank:
    """Tests for estimate_effective_rank function."""

    def test_full_rank_matrix(self):
        """Full rank matrix should have effective rank close to min(m,n)."""
        A = torch.randn(10, 10)

        rank = estimate_effective_rank(A)

        # Should be close to 10 for full rank (allow some tolerance)
        assert rank > 7.0

    def test_low_rank_matrix(self):
        """Low rank matrix should have small effective rank."""
        # Create rank-2 matrix
        U = torch.randn(10, 2)
        V = torch.randn(8, 2)
        A = U @ V.T

        rank = estimate_effective_rank(A, threshold=0.99)

        # Should detect low rank
        assert rank < 4.0


class TestLowRankApprox:
    """Tests for low_rank_approx function."""

    def test_approximation_quality(self):
        """Test that low-rank approximation preserves structure."""
        A = torch.randn(10, 10)

        A_approx = low_rank_approx(A, rank=5)

        assert A_approx.shape == A.shape

        # Approximation error should be reasonable
        error = torch.norm(A - A_approx) / torch.norm(A)
        assert error < 0.8  # Some error expected with rank reduction


class TestSpectralNorm:
    """Tests for spectral_norm function."""

    def test_known_spectral_norm(self):
        """Test spectral norm on diagonal matrix."""
        # Diagonal matrix: spectral norm = max diagonal entry
        A = torch.diag(torch.tensor([5.0, 3.0, 1.0]))

        norm = spectral_norm(A, num_iterations=20)

        # Allow some tolerance due to power iteration approximation
        assert torch.allclose(norm, torch.tensor(5.0), rtol=0.05)

    def test_batch_spectral_norm(self):
        """Test batched spectral norm computation."""
        A = torch.randn(4, 8, 8)

        norms = spectral_norm(A)

        assert norms.shape == (4,)
        assert torch.all(norms > 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
