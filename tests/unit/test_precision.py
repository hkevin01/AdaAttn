"""
Unit tests for precision control utilities.
"""

import pytest
import torch

from adaattn.linalg.precision import (
    PrecisionLevel,
    PrecisionConfig,
    compute_precision_requirements,
    compute_error_bound,
    safe_cast,
    mixed_precision_matmul,
    analyze_numerical_stability,
)


class TestPrecisionLevel:
    """Tests for PrecisionLevel enum."""

    def test_torch_dtype_conversion(self):
        """Test conversion to PyTorch dtypes."""
        assert PrecisionLevel.FP32.torch_dtype == torch.float32
        assert PrecisionLevel.FP16.torch_dtype == torch.float16
        assert PrecisionLevel.BF16.torch_dtype == torch.bfloat16

    def test_bits_property(self):
        """Test bits property."""
        assert PrecisionLevel.FP32.bits == 32
        assert PrecisionLevel.FP16.bits == 16
        assert PrecisionLevel.FP8.bits == 8

    def test_mantissa_bits(self):
        """Test mantissa bits property."""
        assert PrecisionLevel.FP32.mantissa_bits == 23
        assert PrecisionLevel.FP16.mantissa_bits == 10
        assert PrecisionLevel.BF16.mantissa_bits == 7


class TestPrecisionConfig:
    """Tests for PrecisionConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PrecisionConfig()
        assert config.enable_adaptive is True
        assert config.qk_precision == PrecisionLevel.FP16

    def test_custom_config(self):
        """Test custom configuration."""
        config = PrecisionConfig(
            qk_precision=PrecisionLevel.FP32,
            enable_adaptive=False,
        )
        assert config.qk_precision == PrecisionLevel.FP32
        assert config.enable_adaptive is False


class TestComputePrecisionRequirements:
    """Tests for compute_precision_requirements function."""

    def test_normal_scores(self):
        """Test with normal score values."""
        config = PrecisionConfig()
        scores = torch.randn(2, 8, 64, 64)

        compute_prec, accum_prec = compute_precision_requirements(scores, config)

        assert isinstance(compute_prec, PrecisionLevel)
        assert isinstance(accum_prec, PrecisionLevel)

    def test_high_dynamic_range(self):
        """Test with high dynamic range scores."""
        config = PrecisionConfig(magnitude_threshold=100)
        scores = torch.zeros(2, 8, 64, 64)
        scores[0, 0, 0, 0] = 1e5
        scores[0, 0, 0, 1] = 1e-5

        compute_prec, accum_prec = compute_precision_requirements(scores, config)

        # High dynamic range should trigger FP32
        assert compute_prec == PrecisionLevel.FP32

    def test_disabled_adaptive(self):
        """Test with adaptive disabled."""
        config = PrecisionConfig(enable_adaptive=False)
        scores = torch.randn(2, 8, 64, 64) * 1000  # Large values

        compute_prec, accum_prec = compute_precision_requirements(scores, config)

        # Should return configured values, not adapt
        assert compute_prec == config.qk_precision


class TestSafeCast:
    """Tests for safe_cast function."""

    def test_normal_cast(self):
        """Test normal casting."""
        tensor = torch.randn(10, 10)
        casted, overflow = safe_cast(tensor, PrecisionLevel.FP16)

        assert casted.dtype == torch.float16
        assert overflow is False

    def test_overflow_detection(self):
        """Test overflow detection."""
        tensor = torch.tensor([1e10])
        casted, overflow = safe_cast(tensor, PrecisionLevel.FP16)

        assert overflow is True


class TestMixedPrecisionMatmul:
    """Tests for mixed_precision_matmul function."""

    def test_basic_matmul(self):
        """Test basic matrix multiplication."""
        A = torch.randn(32, 64)
        B = torch.randn(64, 32)

        result = mixed_precision_matmul(
            A, B,
            compute_precision=PrecisionLevel.FP16,
            accumulator_precision=PrecisionLevel.FP32,
        )

        assert result.shape == (32, 32)
        assert result.dtype == torch.float32


class TestAnalyzeNumericalStability:
    """Tests for analyze_numerical_stability function."""

    def test_stable_computation(self):
        """Test with numerically stable values."""
        scores = torch.randn(2, 8, 64, 64)
        weights = torch.softmax(scores, dim=-1)

        metrics = analyze_numerical_stability(scores, weights)

        assert "has_nan" in metrics
        assert "has_inf" in metrics
        assert "stability_score" in metrics
        assert metrics["has_nan"] is False
        assert metrics["has_inf"] is False

    def test_unstable_computation(self):
        """Test with unstable values."""
        scores = torch.randn(2, 8, 64, 64)
        weights = torch.full((2, 8, 64, 64), float('nan'))

        metrics = analyze_numerical_stability(scores, weights)

        assert metrics["has_nan"] is True
        assert metrics["stability_score"] == 0.0
