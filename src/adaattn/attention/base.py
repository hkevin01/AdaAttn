"""
Base attention interface for AdaAttn.

This module provides the abstract base class for all attention mechanisms,
defining the common interface and providing utilities for timing, error
handling, and memory management.
"""

from __future__ import annotations

import gc
import logging
import time
import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

# Configure module logger
logger = logging.getLogger(__name__)


class AttentionError(Exception):
    """Base exception for attention-related errors."""

    pass


class ShapeError(AttentionError):
    """Exception raised for tensor shape mismatches."""

    def __init__(
        self,
        message: str,
        expected_shape: Optional[Tuple[int, ...]] = None,
        actual_shape: Optional[Tuple[int, ...]] = None,
    ) -> None:
        self.expected_shape = expected_shape
        self.actual_shape = actual_shape
        full_message = message
        if expected_shape and actual_shape:
            full_message = (
                f"{message}. Expected shape: {expected_shape}, "
                f"got: {actual_shape}"
            )
        super().__init__(full_message)


class MemoryError(AttentionError):
    """Exception raised for memory-related issues."""

    def __init__(
        self,
        message: str,
        required_memory: Optional[int] = None,
        available_memory: Optional[int] = None,
    ) -> None:
        self.required_memory = required_memory
        self.available_memory = available_memory
        full_message = message
        if required_memory and available_memory:
            full_message = (
                f"{message}. Required: {required_memory / 1e9:.2f} GB, "
                f"Available: {available_memory / 1e9:.2f} GB"
            )
        super().__init__(full_message)


class NumericalError(AttentionError):
    """Exception raised for numerical instability issues."""

    pass


class PrecisionMode(Enum):
    """Supported precision modes for attention computation."""

    FP32 = auto()
    FP16 = auto()
    BF16 = auto()
    FP8 = auto()
    AUTO = auto()  # Automatically select based on hardware and input

    @classmethod
    def from_dtype(cls, dtype: torch.dtype) -> "PrecisionMode":
        """Convert PyTorch dtype to PrecisionMode."""
        dtype_map = {
            torch.float32: cls.FP32,
            torch.float16: cls.FP16,
            torch.bfloat16: cls.BF16,
        }
        return dtype_map.get(dtype, cls.FP32)

    def to_dtype(self) -> torch.dtype:
        """Convert PrecisionMode to PyTorch dtype."""
        mode_map = {
            PrecisionMode.FP32: torch.float32,
            PrecisionMode.FP16: torch.float16,
            PrecisionMode.BF16: torch.bfloat16,
            PrecisionMode.FP8: torch.float16,  # Fallback for FP8
            PrecisionMode.AUTO: torch.float16,
        }
        return mode_map[self]


@dataclass
class AttentionMetrics:
    """Container for attention performance metrics."""

    # Timing metrics (in milliseconds)
    forward_time_ms: float = 0.0
    backward_time_ms: float = 0.0
    qk_matmul_time_ms: float = 0.0
    softmax_time_ms: float = 0.0
    av_matmul_time_ms: float = 0.0

    # Memory metrics (in bytes)
    peak_memory_bytes: int = 0
    allocated_memory_bytes: int = 0
    cached_memory_bytes: int = 0

    # Quality metrics
    attention_entropy: float = 0.0
    effective_rank: float = 0.0
    sparsity_ratio: float = 0.0

    # Precision tracking
    precision_mode: PrecisionMode = PrecisionMode.FP32
    precision_switches: int = 0

    # Additional metadata
    batch_size: int = 0
    seq_len_q: int = 0
    seq_len_k: int = 0
    num_heads: int = 0
    head_dim: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for logging."""
        return {
            "timing": {
                "forward_ms": self.forward_time_ms,
                "backward_ms": self.backward_time_ms,
                "qk_matmul_ms": self.qk_matmul_time_ms,
                "softmax_ms": self.softmax_time_ms,
                "av_matmul_ms": self.av_matmul_time_ms,
            },
            "memory": {
                "peak_bytes": self.peak_memory_bytes,
                "allocated_bytes": self.allocated_memory_bytes,
                "cached_bytes": self.cached_memory_bytes,
            },
            "quality": {
                "entropy": self.attention_entropy,
                "effective_rank": self.effective_rank,
                "sparsity": self.sparsity_ratio,
            },
            "config": {
                "precision": self.precision_mode.name,
                "precision_switches": self.precision_switches,
                "batch_size": self.batch_size,
                "seq_len_q": self.seq_len_q,
                "seq_len_k": self.seq_len_k,
                "num_heads": self.num_heads,
                "head_dim": self.head_dim,
            },
        }


@dataclass
class AttentionConfig:
    """Configuration for attention mechanism."""

    # Core parameters
    embed_dim: int = 512
    num_heads: int = 8
    dropout: float = 0.0

    # Precision control
    precision: PrecisionMode = PrecisionMode.AUTO
    allow_precision_fallback: bool = True
    min_precision: PrecisionMode = PrecisionMode.FP16

    # Rank adaptation
    enable_adaptive_rank: bool = False
    min_rank_ratio: float = 0.1
    max_rank_ratio: float = 1.0
    rank_threshold: float = 0.95

    # Memory management
    max_memory_fraction: float = 0.9
    enable_gradient_checkpointing: bool = False

    # Numerical stability
    attention_clamp_value: float = 1e4
    softmax_temperature: float = 1.0
    eps: float = 1e-6

    # Debugging and profiling
    enable_profiling: bool = False
    enable_nan_check: bool = True
    verbose: bool = False

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {self.embed_dim}")
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {self.num_heads}")
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if not 0.0 < self.min_rank_ratio <= self.max_rank_ratio <= 1.0:
            raise ValueError(
                f"Invalid rank ratio range: [{self.min_rank_ratio}, {self.max_rank_ratio}]"
            )
        if not 0.0 < self.max_memory_fraction <= 1.0:
            raise ValueError(
                f"max_memory_fraction must be in (0, 1], got {self.max_memory_fraction}"
            )

    @property
    def head_dim(self) -> int:
        """Compute dimension per attention head."""
        return self.embed_dim // self.num_heads


class Timer:
    """Context manager for timing operations with GPU synchronization."""

    def __init__(self, sync_cuda: bool = True, unit: str = "ms") -> None:
        """
        Initialize timer.

        Args:
            sync_cuda: Whether to synchronize CUDA before timing
            unit: Time unit ('s', 'ms', 'us', 'ns')
        """
        self.sync_cuda = sync_cuda
        self.unit = unit
        self._start_time: float = 0.0
        self._end_time: float = 0.0
        self._elapsed: float = 0.0

        self._unit_multipliers = {
            "s": 1.0,
            "ms": 1e3,
            "us": 1e6,
            "ns": 1e9,
        }

    def __enter__(self) -> "Timer":
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        self._end_time = time.perf_counter()
        self._elapsed = self._end_time - self._start_time

    @property
    def elapsed(self) -> float:
        """Get elapsed time in the specified unit."""
        multiplier = self._unit_multipliers.get(self.unit, 1e3)
        return self._elapsed * multiplier


@contextmanager
def memory_efficient_attention(
    max_memory_fraction: float = 0.9,
    enable_gc: bool = True,
):
    """
    Context manager for memory-efficient attention computation.

    Args:
        max_memory_fraction: Maximum fraction of GPU memory to use
        enable_gc: Whether to run garbage collection after exiting

    Yields:
        None
    """
    if torch.cuda.is_available():
        # Record initial memory state
        initial_memory = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()

        # Set memory fraction limit
        try:
            torch.cuda.set_per_process_memory_fraction(max_memory_fraction)
        except RuntimeError:
            # Memory fraction already set or not supported
            pass

    try:
        yield
    except torch.cuda.OutOfMemoryError as e:
        # Attempt graceful recovery
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if enable_gc:
            gc.collect()
        raise MemoryError(
            f"GPU out of memory during attention computation: {e}",
            available_memory=torch.cuda.get_device_properties(0).total_memory
            if torch.cuda.is_available()
            else None,
        )
    finally:
        if enable_gc:
            gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class BaseAttention(nn.Module, ABC):
    """
    Abstract base class for all attention mechanisms.

    This class provides:
    - Common interface for forward pass
    - Input validation and error handling
    - Timing and profiling utilities
    - Memory management hooks
    - Numerical stability checks

    Subclasses must implement:
    - _forward_impl: The actual attention computation
    """

    def __init__(self, config: Optional[AttentionConfig] = None, **kwargs: Any) -> None:
        """
        Initialize base attention.

        Args:
            config: AttentionConfig instance, or None to create from kwargs
            **kwargs: Configuration parameters if config is None
        """
        super().__init__()

        # Create or validate config
        if config is None:
            self.config = AttentionConfig(**kwargs)
        else:
            self.config = config

        self.config.validate()

        # Initialize metrics tracking
        self._metrics: Optional[AttentionMetrics] = None
        self._hooks: List[Callable] = []

        # Set up logging
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @property
    def embed_dim(self) -> int:
        """Get embedding dimension."""
        return self.config.embed_dim

    @property
    def num_heads(self) -> int:
        """Get number of attention heads."""
        return self.config.num_heads

    @property
    def head_dim(self) -> int:
        """Get dimension per head."""
        return self.config.head_dim

    @property
    def metrics(self) -> Optional[AttentionMetrics]:
        """Get the most recent attention metrics."""
        return self._metrics

    def _validate_inputs(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> None:
        """
        Validate input tensors.

        Args:
            query: Query tensor of shape (batch, seq_q, embed_dim)
            key: Key tensor of shape (batch, seq_k, embed_dim)
            value: Value tensor of shape (batch, seq_v, embed_dim)
            attention_mask: Optional mask tensor

        Raises:
            ShapeError: If tensor shapes are invalid
            TypeError: If tensors have wrong types
        """
        # Check types
        if not isinstance(query, Tensor):
            raise TypeError(f"query must be a Tensor, got {type(query)}")
        if not isinstance(key, Tensor):
            raise TypeError(f"key must be a Tensor, got {type(key)}")
        if not isinstance(value, Tensor):
            raise TypeError(f"value must be a Tensor, got {type(value)}")

        # Check dimensions
        if query.ndim != 3:
            raise ShapeError(
                f"query must be 3D (batch, seq, embed_dim)",
                expected_shape=("batch", "seq", "embed_dim"),
                actual_shape=tuple(query.shape),
            )
        if key.ndim != 3:
            raise ShapeError(
                f"key must be 3D (batch, seq, embed_dim)",
                expected_shape=("batch", "seq", "embed_dim"),
                actual_shape=tuple(key.shape),
            )
        if value.ndim != 3:
            raise ShapeError(
                f"value must be 3D (batch, seq, embed_dim)",
                expected_shape=("batch", "seq", "embed_dim"),
                actual_shape=tuple(value.shape),
            )

        # Check batch sizes match
        if query.size(0) != key.size(0) or query.size(0) != value.size(0):
            raise ShapeError(
                f"Batch sizes must match: query={query.size(0)}, "
                f"key={key.size(0)}, value={value.size(0)}"
            )

        # Check key and value sequence lengths match
        if key.size(1) != value.size(1):
            raise ShapeError(
                f"Key and value sequence lengths must match: "
                f"key={key.size(1)}, value={value.size(1)}"
            )

        # Check embedding dimensions
        if query.size(2) != self.embed_dim:
            raise ShapeError(
                f"Query embedding dimension mismatch",
                expected_shape=(None, None, self.embed_dim),
                actual_shape=tuple(query.shape),
            )

        # Check attention mask if provided
        if attention_mask is not None:
            if attention_mask.ndim not in (2, 3, 4):
                raise ShapeError(
                    f"attention_mask must be 2D, 3D, or 4D, got {attention_mask.ndim}D"
                )

    def _check_numerical_stability(
        self, tensor: Tensor, name: str = "tensor"
    ) -> None:
        """
        Check tensor for numerical issues.

        Args:
            tensor: Tensor to check
            name: Name for error messages

        Raises:
            NumericalError: If NaN or Inf values are detected
        """
        if not self.config.enable_nan_check:
            return

        if torch.isnan(tensor).any():
            raise NumericalError(
                f"NaN values detected in {name}. "
                "Consider using higher precision or gradient clipping."
            )

        if torch.isinf(tensor).any():
            raise NumericalError(
                f"Inf values detected in {name}. "
                "Consider reducing learning rate or using gradient clipping."
            )

    def _init_metrics(
        self,
        query: Tensor,
        key: Tensor,
    ) -> AttentionMetrics:
        """Initialize metrics for current forward pass."""
        batch_size, seq_len_q, _ = query.shape
        _, seq_len_k, _ = key.shape

        return AttentionMetrics(
            batch_size=batch_size,
            seq_len_q=seq_len_q,
            seq_len_k=seq_len_k,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            precision_mode=self.config.precision,
        )

    @abstractmethod
    def _forward_impl(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Implement the actual attention computation.

        Args:
            query: Query tensor (batch, seq_q, embed_dim)
            key: Key tensor (batch, seq_k, embed_dim)
            value: Value tensor (batch, seq_k, embed_dim)
            attention_mask: Optional attention mask
            need_weights: Whether to return attention weights

        Returns:
            Tuple of:
            - Output tensor (batch, seq_q, embed_dim)
            - Attention weights if need_weights=True, else None
        """
        pass

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Compute attention.

        This method handles validation, profiling, and error handling,
        delegating the actual computation to _forward_impl.

        Args:
            query: Query tensor (batch, seq_q, embed_dim)
            key: Key tensor (batch, seq_k, embed_dim)
            value: Value tensor (batch, seq_k, embed_dim)
            attention_mask: Optional attention mask
            need_weights: Whether to return attention weights

        Returns:
            Tuple of:
            - Output tensor (batch, seq_q, embed_dim)
            - Attention weights if need_weights=True, else None

        Raises:
            ShapeError: If input shapes are invalid
            MemoryError: If GPU memory is insufficient
            NumericalError: If numerical instability is detected
        """
        # Validate inputs
        self._validate_inputs(query, key, value, attention_mask)

        # Initialize metrics
        if self.config.enable_profiling:
            self._metrics = self._init_metrics(query, key)

        try:
            # Run forward pass with optional profiling
            with memory_efficient_attention(
                max_memory_fraction=self.config.max_memory_fraction,
                enable_gc=True,
            ):
                if self.config.enable_profiling:
                    with Timer(sync_cuda=True, unit="ms") as timer:
                        output, weights = self._forward_impl(
                            query, key, value, attention_mask, need_weights
                        )
                    self._metrics.forward_time_ms = timer.elapsed
                else:
                    output, weights = self._forward_impl(
                        query, key, value, attention_mask, need_weights
                    )

            # Check numerical stability
            self._check_numerical_stability(output, "attention output")

            # Record memory metrics
            if self.config.enable_profiling and torch.cuda.is_available():
                self._metrics.peak_memory_bytes = torch.cuda.max_memory_allocated()
                self._metrics.allocated_memory_bytes = torch.cuda.memory_allocated()
                self._metrics.cached_memory_bytes = torch.cuda.memory_reserved()

            return output, weights

        except torch.cuda.OutOfMemoryError as e:
            self._logger.error(f"GPU OOM during attention: {e}")
            # Attempt recovery
            torch.cuda.empty_cache()
            gc.collect()
            raise MemoryError(f"GPU out of memory: {e}")

        except Exception as e:
            self._logger.error(f"Attention computation failed: {e}")
            raise

    def extra_repr(self) -> str:
        """Return extra representation for print."""
        return (
            f"embed_dim={self.embed_dim}, "
            f"num_heads={self.num_heads}, "
            f"head_dim={self.head_dim}, "
            f"dropout={self.config.dropout}"
        )
