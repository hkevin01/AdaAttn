"""
Unit tests for base attention interface and utilities.
"""

import pytest
import torch

from adaattn.attention.base import (
    AttentionConfig,
    AttentionError,
    AttentionMetrics,
    MemoryError,
    NumericalError,
    PrecisionMode,
    ShapeError,
    Timer,
)


class TestPrecisionMode:
    """Tests for PrecisionMode enum."""

    def test_from_dtype(self):
        """Test conversion from PyTorch dtype."""
        assert PrecisionMode.from_dtype(torch.float32) == PrecisionMode.FP32
        assert PrecisionMode.from_dtype(torch.float16) == PrecisionMode.FP16
        assert PrecisionMode.from_dtype(torch.bfloat16) == PrecisionMode.BF16

    def test_to_dtype(self):
        """Test conversion to PyTorch dtype."""
        assert PrecisionMode.FP32.to_dtype() == torch.float32
        assert PrecisionMode.FP16.to_dtype() == torch.float16
        assert PrecisionMode.BF16.to_dtype() == torch.bfloat16


class TestAttentionMetrics:
    """Tests for AttentionMetrics dataclass."""

    def test_default_initialization(self):
        """Test default metric values."""
        metrics = AttentionMetrics()
        
        assert metrics.forward_time_ms == 0.0
        assert metrics.peak_memory_bytes == 0
        assert metrics.attention_entropy == 0.0

    def test_custom_initialization(self):
        """Test custom metric values."""
        metrics = AttentionMetrics(
            forward_time_ms=5.2,
            batch_size=16,
            seq_len_q=128,
            num_heads=8,
        )
        
        assert metrics.forward_time_ms == 5.2
        assert metrics.batch_size == 16
        assert metrics.seq_len_q == 128
        assert metrics.num_heads == 8

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = AttentionMetrics(
            forward_time_ms=5.0,
            batch_size=4,
            seq_len_q=64,
        )
        
        result = metrics.to_dict()
        
        assert isinstance(result, dict)
        assert "timing" in result
        assert "memory" in result
        assert "quality" in result
        assert "config" in result
        assert result["timing"]["forward_ms"] == 5.0
        assert result["config"]["batch_size"] == 4


class TestAttentionConfig:
    """Tests for AttentionConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = AttentionConfig()
        
        assert config.embed_dim == 512
        assert config.num_heads == 8
        assert config.dropout == 0.0
        assert config.precision == PrecisionMode.AUTO

    def test_head_dim_property(self):
        """Test head_dim computed property."""
        config = AttentionConfig(embed_dim=512, num_heads=8)
        assert config.head_dim == 64
        
        config = AttentionConfig(embed_dim=768, num_heads=12)
        assert config.head_dim == 64

    def test_validation_positive_embed_dim(self):
        """Test validation rejects negative embed_dim."""
        config = AttentionConfig(embed_dim=-1)
        
        with pytest.raises(ValueError, match="embed_dim must be positive"):
            config.validate()

    def test_validation_positive_num_heads(self):
        """Test validation rejects non-positive num_heads."""
        config = AttentionConfig(num_heads=0)
        
        with pytest.raises(ValueError, match="num_heads must be positive"):
            config.validate()

    def test_validation_divisible_embed_dim(self):
        """Test validation requires embed_dim divisible by num_heads."""
        config = AttentionConfig(embed_dim=513, num_heads=8)
        
        with pytest.raises(ValueError, match="must be divisible"):
            config.validate()

    def test_validation_dropout_range(self):
        """Test validation checks dropout range."""
        config = AttentionConfig(dropout=1.5)
        
        with pytest.raises(ValueError, match="dropout must be in"):
            config.validate()

    def test_validation_rank_ratio_range(self):
        """Test validation checks rank ratio range."""
        config = AttentionConfig(min_rank_ratio=0.8, max_rank_ratio=0.5)
        
        with pytest.raises(ValueError, match="Invalid rank ratio"):
            config.validate()

    def test_valid_config_passes(self):
        """Test that valid config passes validation."""
        config = AttentionConfig(
            embed_dim=512,
            num_heads=8,
            dropout=0.1,
        )
        
        # Should not raise
        config.validate()


class TestTimer:
    """Tests for Timer context manager."""

    def test_basic_timing(self):
        """Test basic timing functionality."""
        timer = Timer(sync_cuda=False, unit="ms")
        
        with timer:
            # Simulate some work
            x = torch.randn(1000, 1000)
            y = torch.matmul(x, x)
        
        # Should have non-zero elapsed time
        assert timer.elapsed > 0

    def test_different_units(self):
        """Test timing with different units."""
        import time
        
        timer_ms = Timer(sync_cuda=False, unit="ms")
        timer_us = Timer(sync_cuda=False, unit="us")
        
        with timer_ms:
            time.sleep(0.01)  # 10ms
        
        with timer_us:
            time.sleep(0.01)  # 10ms
        
        # us should be ~1000x larger than ms
        assert timer_us.elapsed > timer_ms.elapsed * 900


class TestExceptions:
    """Tests for custom exception classes."""

    def test_shape_error_basic(self):
        """Test ShapeError creation."""
        error = ShapeError("Shape mismatch")
        assert "Shape mismatch" in str(error)

    def test_shape_error_with_shapes(self):
        """Test ShapeError with shape information."""
        error = ShapeError(
            "Shape mismatch",
            expected_shape=(4, 8, 64),
            actual_shape=(4, 8, 32),
        )
        
        assert "Expected shape" in str(error)
        assert "(4, 8, 64)" in str(error)
        assert "(4, 8, 32)" in str(error)

    def test_memory_error_basic(self):
        """Test MemoryError creation."""
        error = MemoryError("Out of memory")
        assert "Out of memory" in str(error)

    def test_memory_error_with_values(self):
        """Test MemoryError with memory values."""
        error = MemoryError(
            "Insufficient memory",
            required_memory=int(10e9),  # 10GB
            available_memory=int(8e9),  # 8GB
        )
        
        assert "Required: 10.00 GB" in str(error)
        assert "Available: 8.00 GB" in str(error)

    def test_numerical_error(self):
        """Test NumericalError creation."""
        error = NumericalError("NaN detected")
        assert "NaN detected" in str(error)

    def test_exception_hierarchy(self):
        """Test that all exceptions inherit from AttentionError."""
        assert issubclass(ShapeError, AttentionError)
        assert issubclass(MemoryError, AttentionError)
        assert issubclass(NumericalError, AttentionError)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
