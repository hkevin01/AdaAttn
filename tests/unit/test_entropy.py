"""
Unit tests for entropy estimation utilities.
"""

import math

import pytest
import torch

from adaattn.linalg.entropy import (
    attention_entropy,
    entropy_based_rank_hint,
    estimate_entropy,
    normalized_entropy,
)


class TestEstimateEntropy:
    """Tests for estimate_entropy function."""

    def test_uniform_distribution(self):
        """Test entropy of uniform distribution."""
        # Uniform distribution: all elements equal
        probs = torch.ones(10) / 10
        entropy = estimate_entropy(probs)
        
        # Entropy of uniform distribution is log(n)
        expected = math.log(10)
        assert torch.allclose(entropy, torch.tensor(expected), rtol=1e-5)

    def test_peaked_distribution(self):
        """Test entropy of peaked distribution."""
        # Peaked: one element = 1, rest = 0
        probs = torch.zeros(10)
        probs[0] = 1.0
        entropy = estimate_entropy(probs)
        
        # Entropy should be very small (close to 0)
        assert entropy < 0.01

    def test_batch_computation(self):
        """Test batch entropy computation."""
        # Batch of distributions
        probs = torch.ones(4, 8, 10) / 10
        entropy = estimate_entropy(probs, dim=-1)
        
        assert entropy.shape == (4, 8)
        expected = math.log(10)
        assert torch.allclose(entropy, torch.full((4, 8), expected), rtol=1e-5)


class TestNormalizedEntropy:
    """Tests for normalized_entropy function."""

    def test_uniform_gives_one(self):
        """Uniform distribution should give normalized entropy = 1."""
        probs = torch.ones(10) / 10
        norm_entropy = normalized_entropy(probs)
        
        assert torch.allclose(norm_entropy, torch.tensor(1.0), rtol=1e-5)

    def test_peaked_gives_zero(self):
        """Peaked distribution should give normalized entropy ≈ 0."""
        probs = torch.zeros(10)
        probs[0] = 1.0
        norm_entropy = normalized_entropy(probs)
        
        assert norm_entropy < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestAttentionEntropy:
    """Test attention_entropy function."""

    def test_reduce_mean(self):
        """Test mean reduction."""
        from adaattn.linalg.entropy import attention_entropy
        
        # Create attention weights (batch=2, heads=4, seq_q=8, seq_k=8)
        attn = torch.softmax(torch.randn(2, 4, 8, 8), dim=-1)
        
        entropy = attention_entropy(attn, reduce="mean")
        
        assert entropy.shape == (2, 4)
        assert (entropy >= 0).all()

    def test_reduce_sum(self):
        """Test sum reduction."""
        from adaattn.linalg.entropy import attention_entropy
        
        attn = torch.softmax(torch.randn(2, 4, 8, 8), dim=-1)
        
        entropy = attention_entropy(attn, reduce="sum")
        
        assert entropy.shape == (2, 4)
        assert (entropy >= 0).all()

    def test_reduce_none(self):
        """Test no reduction."""
        from adaattn.linalg.entropy import attention_entropy
        
        attn = torch.softmax(torch.randn(2, 4, 8, 8), dim=-1)
        
        entropy = attention_entropy(attn, reduce="none")
        
        assert entropy.shape == (2, 4, 8)
        assert (entropy >= 0).all()

    def test_invalid_reduce(self):
        """Test invalid reduce parameter."""
        from adaattn.linalg.entropy import attention_entropy
        
        attn = torch.softmax(torch.randn(2, 4, 8, 8), dim=-1)
        
        with pytest.raises(ValueError, match="Invalid reduce"):
            attention_entropy(attn, reduce="invalid")

    def test_3d_input(self):
        """Test with 3D input (no head dimension)."""
        from adaattn.linalg.entropy import attention_entropy
        
        attn = torch.softmax(torch.randn(2, 8, 8), dim=-1)
        
        entropy = attention_entropy(attn, reduce="mean")
        
        assert entropy.shape == (2,)
        assert (entropy >= 0).all()
