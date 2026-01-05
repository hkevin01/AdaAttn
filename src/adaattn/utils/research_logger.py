"""Advanced research logging system for AdaAttn experiments."""

import logging
import json
import time
import psutil
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from contextlib import contextmanager
import torch
import numpy as np
from dataclasses import dataclass, asdict
from collections import defaultdict, deque


@dataclass
class ExperimentMetrics:
    """Container for experiment metrics."""
    timestamp: float
    epoch: Optional[int] = None
    step: Optional[int] = None
    loss: Optional[float] = None
    accuracy: Optional[float] = None
    perplexity: Optional[float] = None
    learning_rate: Optional[float] = None
    gpu_memory_used: Optional[float] = None
    gpu_memory_total: Optional[float] = None
    cpu_memory_used: Optional[float] = None
    cpu_memory_percent: Optional[float] = None
    attention_stats: Optional[Dict[str, Any]] = None
    model_params: Optional[Dict[str, Any]] = None
    custom_metrics: Optional[Dict[str, float]] = None


class MetricsBuffer:
    """Thread-safe buffer for collecting metrics."""
    
    def __init__(self, max_size: int = 10000):
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
    
    def add(self, metrics: ExperimentMetrics):
        """Add metrics to buffer."""
        with self.lock:
            self.buffer.append(metrics)
    
    def get_recent(self, n: int = 100) -> List[ExperimentMetrics]:
        """Get n most recent metrics."""
        with self.lock:
            return list(self.buffer)[-n:]
    
    def clear(self):
        """Clear the buffer."""
        with self.lock:
            self.buffer.clear()


class ResearchLogger:
    """Advanced research logging system with real-time monitoring."""
    
    def __init__(
        self,
        experiment_name: str,
        log_dir: str = "logs",
        log_level: str = "INFO",
        enable_wandb: bool = False,
        enable_tensorboard: bool = True,
        enable_mlflow: bool = False,
        auto_monitor: bool = True,
        monitor_interval: float = 5.0
    ):
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self.logger = self._setup_logger(log_level)
        
        # Initialize metrics buffer
        self.metrics_buffer = MetricsBuffer()
        
        # Experiment tracking
        self.experiment_start_time = time.time()
        self.current_epoch = 0
        self.current_step = 0
        
        # External logging integration
        self.wandb_enabled = enable_wandb
        self.tensorboard_enabled = enable_tensorboard
        self.mlflow_enabled = enable_mlflow
        
        self.wandb_run = None
        self.tensorboard_writer = None
        self.mlflow_client = None
        
        # Auto monitoring
        self.auto_monitor = auto_monitor
        self.monitor_interval = monitor_interval
        self.monitor_thread = None
        self.monitor_stop_event = threading.Event()
        
        self._initialize_integrations()
        
        if auto_monitor:
            self.start_monitoring()
    
    def _setup_logger(self, log_level: str) -> logging.Logger:
        """Set up the main logger."""
        logger = logging.getLogger(f"adaattn.research.{self.experiment_name}")
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)
        
        # File handler
        log_file = self.log_dir / f"{self.experiment_name}.log"
        file_handler = logging.FileHandler(log_file)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
        
        return logger
    
    def _initialize_integrations(self):
        """Initialize external logging integrations."""
        if self.wandb_enabled:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=f"adaattn_{self.experiment_name}",
                    name=self.experiment_name,
                    reinit=True
                )
                self.logger.info("W&B integration enabled")
            except ImportError:
                self.logger.warning("W&B not available, disabling integration")
                self.wandb_enabled = False
        
        if self.tensorboard_enabled:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = self.log_dir / "tensorboard" / self.experiment_name
                self.tensorboard_writer = SummaryWriter(str(tb_dir))
                self.logger.info(f"TensorBoard logging to: {tb_dir}")
            except ImportError:
                self.logger.warning("TensorBoard not available, disabling integration")
                self.tensorboard_enabled = False
        
        if self.mlflow_enabled:
            try:
                import mlflow
                self.mlflow_client = mlflow
                mlflow.start_run(run_name=self.experiment_name)
                self.logger.info("MLflow integration enabled")
            except ImportError:
                self.logger.warning("MLflow not available, disabling integration")
                self.mlflow_enabled = False
    
    def start_monitoring(self):
        """Start automatic system monitoring."""
        if self.monitor_thread is None or not self.monitor_thread.is_alive():
            self.monitor_stop_event.clear()
            self.monitor_thread = threading.Thread(
                target=self._monitoring_loop,
                daemon=True
            )
            self.monitor_thread.start()
            self.logger.info("Started automatic monitoring")
    
    def stop_monitoring(self):
        """Stop automatic system monitoring."""
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_stop_event.set()
            self.monitor_thread.join(timeout=5.0)
            self.logger.info("Stopped automatic monitoring")
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        while not self.monitor_stop_event.is_set():
            try:
                # Collect system metrics
                metrics = self._collect_system_metrics()
                self.metrics_buffer.add(metrics)
                
                # Log to external systems
                self._log_to_integrations(metrics)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
            
            # Wait for next interval
            self.monitor_stop_event.wait(self.monitor_interval)
    
    def _collect_system_metrics(self) -> ExperimentMetrics:
        """Collect current system metrics."""
        timestamp = time.time()
        
        # CPU/Memory metrics
        cpu_memory = psutil.virtual_memory()
        
        # GPU metrics
        gpu_memory_used = None
        gpu_memory_total = None
        if torch.cuda.is_available():
            try:
                gpu_memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
                gpu_memory_total = torch.cuda.max_memory_allocated() / 1024**3  # GB
            except:
                pass
        
        return ExperimentMetrics(
            timestamp=timestamp,
            cpu_memory_used=cpu_memory.used / 1024**3,  # GB
            cpu_memory_percent=cpu_memory.percent,
            gpu_memory_used=gpu_memory_used,
            gpu_memory_total=gpu_memory_total
        )
    
    def log_metrics(
        self,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        loss: Optional[float] = None,
        accuracy: Optional[float] = None,
        perplexity: Optional[float] = None,
        learning_rate: Optional[float] = None,
        attention_stats: Optional[Dict[str, Any]] = None,
        custom_metrics: Optional[Dict[str, float]] = None,
        **kwargs
    ):
        """Log training metrics."""
        # Update current state
        if epoch is not None:
            self.current_epoch = epoch
        if step is not None:
            self.current_step = step
        
        # Collect system metrics
        system_metrics = self._collect_system_metrics()
        
        # Create full metrics object
        metrics = ExperimentMetrics(
            timestamp=system_metrics.timestamp,
            epoch=epoch or self.current_epoch,
            step=step or self.current_step,
            loss=loss,
            accuracy=accuracy,
            perplexity=perplexity,
            learning_rate=learning_rate,
            gpu_memory_used=system_metrics.gpu_memory_used,
            gpu_memory_total=system_metrics.gpu_memory_total,
            cpu_memory_used=system_metrics.cpu_memory_used,
            cpu_memory_percent=system_metrics.cpu_memory_percent,
            attention_stats=attention_stats,
            custom_metrics=custom_metrics
        )
        
        # Add to buffer
        self.metrics_buffer.add(metrics)
        
        # Log to console
        self._log_to_console(metrics)
        
        # Log to external systems
        self._log_to_integrations(metrics)
    
    def _log_to_console(self, metrics: ExperimentMetrics):
        """Log metrics to console."""
        msg_parts = []
        
        if metrics.epoch is not None:
            msg_parts.append(f"Epoch {metrics.epoch}")
        if metrics.step is not None:
            msg_parts.append(f"Step {metrics.step}")
        if metrics.loss is not None:
            msg_parts.append(f"Loss: {metrics.loss:.4f}")
        if metrics.accuracy is not None:
            msg_parts.append(f"Acc: {metrics.accuracy:.3f}")
        if metrics.gpu_memory_used is not None:
            msg_parts.append(f"GPU Mem: {metrics.gpu_memory_used:.1f}GB")
        
        if msg_parts:
            self.logger.info(" | ".join(msg_parts))
    
    def _log_to_integrations(self, metrics: ExperimentMetrics):
        """Log metrics to external integrations."""
        metrics_dict = asdict(metrics)
        
        # Remove None values
        metrics_dict = {k: v for k, v in metrics_dict.items() if v is not None}
        
        # W&B logging
        if self.wandb_enabled and self.wandb_run:
            try:
                import wandb
                wandb.log(metrics_dict, step=metrics.step)
            except Exception as e:
                self.logger.warning(f"W&B logging failed: {e}")
        
        # TensorBoard logging
        if self.tensorboard_enabled and self.tensorboard_writer:
            try:
                for key, value in metrics_dict.items():
                    if isinstance(value, (int, float)):
                        self.tensorboard_writer.add_scalar(
                            key, value, metrics.step or 0
                        )
            except Exception as e:
                self.logger.warning(f"TensorBoard logging failed: {e}")
        
        # MLflow logging
        if self.mlflow_enabled and self.mlflow_client:
            try:
                for key, value in metrics_dict.items():
                    if isinstance(value, (int, float)):
                        self.mlflow_client.log_metric(key, value, step=metrics.step)
            except Exception as e:
                self.logger.warning(f"MLflow logging failed: {e}")
    
    def log_attention_analysis(self, attention_weights: torch.Tensor, layer_name: str = ""):
        """Log detailed attention analysis."""
        try:
            with torch.no_grad():
                # Basic statistics
                stats = {
                    f"attention/{layer_name}/mean": attention_weights.mean().item(),
                    f"attention/{layer_name}/std": attention_weights.std().item(),
                    f"attention/{layer_name}/max": attention_weights.max().item(),
                    f"attention/{layer_name}/min": attention_weights.min().item(),
                }
                
                # Entropy analysis
                # Reshape to [batch_size * num_heads, seq_len, seq_len]
                attn_flat = attention_weights.view(-1, attention_weights.size(-2), attention_weights.size(-1))
                
                # Calculate entropy for each head
                eps = 1e-8
                entropy_per_head = -(attn_flat * torch.log(attn_flat + eps)).sum(dim=-1).mean(dim=-1)
                
                stats[f"attention/{layer_name}/entropy_mean"] = entropy_per_head.mean().item()
                stats[f"attention/{layer_name}/entropy_std"] = entropy_per_head.std().item()
                
                # Concentration analysis (how peaked are the attention weights)
                max_attention = attn_flat.max(dim=-1)[0]
                stats[f"attention/{layer_name}/concentration"] = max_attention.mean().item()
                
                self.log_metrics(custom_metrics=stats)
                
        except Exception as e:
            self.logger.warning(f"Failed to log attention analysis: {e}")
    
    def log_model_info(self, model: torch.nn.Module):
        """Log model information and parameters."""
        try:
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            model_info = {
                "model/total_parameters": total_params,
                "model/trainable_parameters": trainable_params,
                "model/parameter_ratio": trainable_params / total_params if total_params > 0 else 0
            }
            
            # Log parameter sizes by layer type
            param_by_type = defaultdict(int)
            for name, param in model.named_parameters():
                layer_type = name.split('.')[0] if '.' in name else name
                param_by_type[layer_type] += param.numel()
            
            for layer_type, count in param_by_type.items():
                model_info[f"model/params_{layer_type}"] = count
            
            self.log_metrics(model_params=model_info)
            self.logger.info(f"Model: {total_params:,} total params, {trainable_params:,} trainable")
            
        except Exception as e:
            self.logger.warning(f"Failed to log model info: {e}")
    
    @contextmanager
    def timer(self, name: str):
        """Context manager for timing operations."""
        start_time = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            self.log_metrics(custom_metrics={f"timing/{name}": elapsed})
            self.logger.info(f"⏱️ {name}: {elapsed:.3f}s")
    
    def save_experiment_summary(self) -> Path:
        """Save comprehensive experiment summary."""
        summary_file = self.log_dir / f"{self.experiment_name}_summary.json"
        
        # Get all metrics
        all_metrics = self.metrics_buffer.get_recent(len(self.metrics_buffer.buffer))
        
        # Calculate summary statistics
        if all_metrics:
            summary = {
                "experiment_name": self.experiment_name,
                "start_time": self.experiment_start_time,
                "end_time": time.time(),
                "duration": time.time() - self.experiment_start_time,
                "total_metrics": len(all_metrics),
                "final_epoch": all_metrics[-1].epoch if all_metrics[-1].epoch else 0,
                "final_step": all_metrics[-1].step if all_metrics[-1].step else 0,
            }
            
            # Add metric summaries
            metrics_with_loss = [m for m in all_metrics if m.loss is not None]
            if metrics_with_loss:
                losses = [m.loss for m in metrics_with_loss]
                summary["loss"] = {
                    "final": losses[-1],
                    "best": min(losses),
                    "mean": np.mean(losses),
                    "std": np.std(losses)
                }
            
            metrics_with_acc = [m for m in all_metrics if m.accuracy is not None]
            if metrics_with_acc:
                accuracies = [m.accuracy for m in metrics_with_acc]
                summary["accuracy"] = {
                    "final": accuracies[-1],
                    "best": max(accuracies),
                    "mean": np.mean(accuracies),
                    "std": np.std(accuracies)
                }
        else:
            summary = {
                "experiment_name": self.experiment_name,
                "start_time": self.experiment_start_time,
                "end_time": time.time(),
                "duration": time.time() - self.experiment_start_time,
                "total_metrics": 0,
            }
        
        # Save summary
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"📊 Experiment summary saved to: {summary_file}")
        return summary_file
    
    def cleanup(self):
        """Clean up resources."""
        # Stop monitoring
        self.stop_monitoring()
        
        # Save summary
        self.save_experiment_summary()
        
        # Close integrations
        if self.tensorboard_enabled and self.tensorboard_writer:
            self.tensorboard_writer.close()
        
        if self.wandb_enabled and self.wandb_run:
            try:
                import wandb
                wandb.finish()
            except:
                pass
        
        if self.mlflow_enabled and self.mlflow_client:
            try:
                self.mlflow_client.end_run()
            except:
                pass
        
        self.logger.info("🔬 Research logging cleanup complete")
    
    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.cleanup()
        except:
            pass


# Global research logger instance
_global_logger: Optional[ResearchLogger] = None


def get_research_logger(
    experiment_name: Optional[str] = None,
    **kwargs
) -> ResearchLogger:
    """Get or create global research logger."""
    global _global_logger
    
    if _global_logger is None or experiment_name is not None:
        if experiment_name is None:
            experiment_name = f"adaattn_exp_{int(time.time())}"
        
        if _global_logger:
            _global_logger.cleanup()
        
        _global_logger = ResearchLogger(experiment_name, **kwargs)
    
    return _global_logger


def log_metrics(**kwargs):
    """Convenience function to log metrics to global logger."""
    logger = get_research_logger()
    logger.log_metrics(**kwargs)
