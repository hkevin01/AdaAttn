"""
Logging utilities for AdaAttn.

Provides structured logging for attention statistics.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def setup_logging(
    level: int = logging.INFO,
    format_string: Optional[str] = None,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Set up logging for AdaAttn.

    Args:
        level: Logging level
        format_string: Custom format string
        log_file: Optional file to log to

    Returns:
        Configured logger
    """
    if format_string is None:
        format_string = (
            "%(asctime)s | %(levelname)-8s | "
            "%(name)s:%(lineno)d | %(message)s"
        )

    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # File handler (optional)
    handlers = [console_handler]
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    # Configure root logger
    logging.basicConfig(
        level=level,
        handlers=handlers,
        force=True,
    )

    logger = logging.getLogger("adaattn")
    logger.setLevel(level)

    return logger


def get_logger(name: str = "adaattn") -> logging.Logger:
    """Get a logger with the given name."""
    return logging.getLogger(name)


@dataclass
class AttentionLogEntry:
    """Single attention log entry."""

    layer_idx: int
    batch_idx: int
    head_idx: int
    seq_len: int
    entropy: float
    effective_rank: float
    precision_used: str
    rank_used: int
    time_ms: float
    memory_mb: float


class AttentionLogger:
    """
    Logger for attention statistics.

    Collects and aggregates attention behavior across layers and batches.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.entries: List[AttentionLogEntry] = []
        self.logger = get_logger("adaattn.attention")

    def log(
        self,
        layer_idx: int,
        batch_idx: int = 0,
        head_idx: int = 0,
        seq_len: int = 0,
        entropy: float = 0.0,
        effective_rank: float = 0.0,
        precision_used: str = "fp16",
        rank_used: int = -1,
        time_ms: float = 0.0,
        memory_mb: float = 0.0,
    ):
        """Log attention statistics for a single operation."""
        if not self.enabled:
            return

        entry = AttentionLogEntry(
            layer_idx=layer_idx,
            batch_idx=batch_idx,
            head_idx=head_idx,
            seq_len=seq_len,
            entropy=entropy,
            effective_rank=effective_rank,
            precision_used=precision_used,
            rank_used=rank_used,
            time_ms=time_ms,
            memory_mb=memory_mb,
        )

        self.entries.append(entry)

        self.logger.debug(
            f"Layer {layer_idx} Head {head_idx}: "
            f"entropy={entropy:.3f}, rank={rank_used}, "
            f"precision={precision_used}, time={time_ms:.2f}ms"
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self.entries:
            return {}

        total_entries = len(self.entries)

        # Compute aggregates
        avg_entropy = sum(e.entropy for e in self.entries) / total_entries
        avg_rank = sum(e.effective_rank for e in self.entries) / total_entries
        total_time = sum(e.time_ms for e in self.entries)
        total_memory = sum(e.memory_mb for e in self.entries)

        # Count precision usage
        precision_counts = {}
        for e in self.entries:
            precision_counts[e.precision_used] = (
                precision_counts.get(e.precision_used, 0) + 1
            )

        # Count rank usage
        rank_histogram = {}
        for e in self.entries:
            bucket = (e.rank_used // 16) * 16  # 16-bucket histogram
            rank_histogram[bucket] = rank_histogram.get(bucket, 0) + 1

        return {
            "total_entries": total_entries,
            "avg_entropy": avg_entropy,
            "avg_effective_rank": avg_rank,
            "total_time_ms": total_time,
            "total_memory_mb": total_memory,
            "precision_distribution": precision_counts,
            "rank_histogram": rank_histogram,
        }

    def get_layer_summary(self, layer_idx: int) -> Dict[str, Any]:
        """Get summary for a specific layer."""
        layer_entries = [e for e in self.entries if e.layer_idx == layer_idx]

        if not layer_entries:
            return {}

        return {
            "num_entries": len(layer_entries),
            "avg_entropy": sum(e.entropy for e in layer_entries) / len(layer_entries),
            "avg_rank": sum(e.effective_rank for e in layer_entries) / len(layer_entries),
            "avg_time_ms": sum(e.time_ms for e in layer_entries) / len(layer_entries),
        }

    def clear(self):
        """Clear all logged entries."""
        self.entries = []

    def save(self, filepath: str):
        """Save logs to a JSON file."""
        import json

        data = {
            "summary": self.get_summary(),
            "entries": [
                {
                    "layer_idx": e.layer_idx,
                    "batch_idx": e.batch_idx,
                    "head_idx": e.head_idx,
                    "seq_len": e.seq_len,
                    "entropy": e.entropy,
                    "effective_rank": e.effective_rank,
                    "precision_used": e.precision_used,
                    "rank_used": e.rank_used,
                    "time_ms": e.time_ms,
                    "memory_mb": e.memory_mb,
                }
                for e in self.entries
            ],
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        self.logger.info(f"Saved attention logs to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "AttentionLogger":
        """Load logs from a JSON file."""
        import json

        with open(filepath, "r") as f:
            data = json.load(f)

        logger = cls(enabled=False)

        for entry_data in data.get("entries", []):
            logger.entries.append(AttentionLogEntry(**entry_data))

        return logger
