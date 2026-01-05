"""Real-time monitoring dashboard for AdaAttn research."""

import time
import json
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import psutil
import torch
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn
from dataclasses import dataclass
from collections import deque, defaultdict


@dataclass
class SystemStatus:
    """Current system status."""
    cpu_percent: float
    memory_percent: float
    gpu_memory_used: Optional[float] = None
    gpu_memory_total: Optional[float] = None
    gpu_utilization: Optional[float] = None
    disk_usage: Optional[float] = None
    network_io: Optional[Dict[str, int]] = None


@dataclass
class ExperimentStatus:
    """Current experiment status."""
    name: str
    start_time: float
    current_epoch: int = 0
    current_step: int = 0
    current_loss: Optional[float] = None
    current_accuracy: Optional[float] = None
    learning_rate: Optional[float] = None
    samples_processed: int = 0
    estimated_time_remaining: Optional[float] = None


class ResearchMonitor:
    """Real-time research monitoring dashboard."""
    
    def __init__(
        self,
        experiment_name: str,
        log_dir: str = "logs",
        update_interval: float = 1.0,
        max_history: int = 1000
    ):
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.update_interval = update_interval
        
        # Console and display
        self.console = Console()
        self.live = None
        
        # Status tracking
        self.system_history = deque(maxlen=max_history)
        self.experiment_status = ExperimentStatus(
            name=experiment_name,
            start_time=time.time()
        )
        
        # Monitoring thread
        self.monitor_thread = None
        self.stop_event = threading.Event()
        self.is_running = False
        
        # Metrics tracking
        self.metrics_history = deque(maxlen=max_history)
        self.attention_stats = defaultdict(list)
        
        # Performance tracking
        self.throughput_history = deque(maxlen=100)
        self.last_step_time = None
        
    def start(self):
        """Start the monitoring dashboard."""
        if self.is_running:
            return
        
        self.is_running = True
        self.stop_event.clear()
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitor_thread.start()
        
        # Start live display
        self._start_live_display()
    
    def stop(self):
        """Stop the monitoring dashboard."""
        if not self.is_running:
            return
        
        self.is_running = False
        self.stop_event.set()
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        if self.live:
            self.live.stop()
        
        # Save final report
        self._save_monitoring_report()
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while not self.stop_event.is_set():
            try:
                # Collect system metrics
                status = self._collect_system_status()
                self.system_history.append(status)
                
                # Update display
                if self.live:
                    self.live.update(self._create_layout())
                
            except Exception as e:
                self.console.print(f"[red]Monitoring error: {e}[/red]")
            
            # Wait for next update
            self.stop_event.wait(self.update_interval)
    
    def _collect_system_status(self) -> SystemStatus:
        """Collect current system status."""
        # CPU and Memory
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        # GPU metrics
        gpu_memory_used = None
        gpu_memory_total = None
        gpu_utilization = None
        
        if torch.cuda.is_available():
            try:
                # Memory
                gpu_memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
                gpu_memory_total = torch.cuda.max_memory_allocated() / 1024**3  # GB
                
                # Try to get utilization if nvidia-ml-py is available
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_utilization = util.gpu
                except:
                    pass
            except:
                pass
        
        # Disk usage
        disk_usage = None
        try:
            disk = psutil.disk_usage('/')
            disk_usage = (disk.used / disk.total) * 100
        except:
            pass
        
        # Network I/O
        network_io = None
        try:
            net_io = psutil.net_io_counters()
            network_io = {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv
            }
        except:
            pass
        
        return SystemStatus(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            gpu_memory_used=gpu_memory_used,
            gpu_memory_total=gpu_memory_total,
            gpu_utilization=gpu_utilization,
            disk_usage=disk_usage,
            network_io=network_io
        )
    
    def update_experiment_status(
        self,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        loss: Optional[float] = None,
        accuracy: Optional[float] = None,
        learning_rate: Optional[float] = None,
        **kwargs
    ):
        """Update experiment status."""
        if epoch is not None:
            self.experiment_status.current_epoch = epoch
        if step is not None:
            # Calculate throughput
            if self.last_step_time and step > self.experiment_status.current_step:
                steps_delta = step - self.experiment_status.current_step
                time_delta = time.time() - self.last_step_time
                throughput = steps_delta / time_delta if time_delta > 0 else 0
                self.throughput_history.append(throughput)
            
            self.experiment_status.current_step = step
            self.last_step_time = time.time()
        
        if loss is not None:
            self.experiment_status.current_loss = loss
        if accuracy is not None:
            self.experiment_status.current_accuracy = accuracy
        if learning_rate is not None:
            self.experiment_status.learning_rate = learning_rate
        
        # Store metrics
        self.metrics_history.append({
            'timestamp': time.time(),
            'epoch': self.experiment_status.current_epoch,
            'step': self.experiment_status.current_step,
            'loss': loss,
            'accuracy': accuracy,
            'learning_rate': learning_rate
        })
    
    def log_attention_stats(self, stats: Dict[str, Any]):
        """Log attention statistics."""
        for key, value in stats.items():
            if isinstance(value, (int, float)):
                self.attention_stats[key].append({
                    'timestamp': time.time(),
                    'value': value
                })
                # Keep only recent entries
                if len(self.attention_stats[key]) > 100:
                    self.attention_stats[key] = self.attention_stats[key][-100:]
    
    def _create_layout(self) -> Layout:
        """Create the main dashboard layout."""
        layout = Layout()
        
        # Main sections
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=5)
        )
        
        # Split main into left and right
        layout["main"].split_row(
            Layout(name="left", ratio=2),
            Layout(name="right", ratio=1)
        )
        
        # Split left into experiment and system
        layout["left"].split_column(
            Layout(name="experiment", ratio=2),
            Layout(name="system", ratio=1)
        )
        
        # Fill sections
        layout["header"].update(self._create_header())
        layout["experiment"].update(self._create_experiment_panel())
        layout["system"].update(self._create_system_panel())
        layout["right"].update(self._create_metrics_panel())
        layout["footer"].update(self._create_footer())
        
        return layout
    
    def _create_header(self) -> Panel:
        """Create header panel."""
        elapsed = time.time() - self.experiment_status.start_time
        elapsed_str = str(timedelta(seconds=int(elapsed)))
        
        header_text = Text()
        header_text.append("🔬 AdaAttn Research Monitor", style="bold blue")
        header_text.append(f" | Experiment: {self.experiment_name}", style="cyan")
        header_text.append(f" | Runtime: {elapsed_str}", style="green")
        
        if self.experiment_status.estimated_time_remaining:
            eta_str = str(timedelta(seconds=int(self.experiment_status.estimated_time_remaining)))
            header_text.append(f" | ETA: {eta_str}", style="yellow")
        
        return Panel(header_text, style="blue")
    
    def _create_experiment_panel(self) -> Panel:
        """Create experiment status panel."""
        table = Table(title="📊 Experiment Status", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Current", style="green")
        table.add_column("Best", style="yellow")
        table.add_column("Progress", style="white")
        
        # Current values
        current_loss = f"{self.experiment_status.current_loss:.4f}" if self.experiment_status.current_loss else "N/A"
        current_acc = f"{self.experiment_status.current_accuracy:.3f}" if self.experiment_status.current_accuracy else "N/A"
        current_lr = f"{self.experiment_status.learning_rate:.2e}" if self.experiment_status.learning_rate else "N/A"
        
        # Best values from history
        best_loss = "N/A"
        best_acc = "N/A"
        if self.metrics_history:
            losses = [m.get('loss') for m in self.metrics_history if m.get('loss') is not None]
            accuracies = [m.get('accuracy') for m in self.metrics_history if m.get('accuracy') is not None]
            
            if losses:
                best_loss = f"{min(losses):.4f}"
            if accuracies:
                best_acc = f"{max(accuracies):.3f}"
        
        # Progress bars (simplified)
        epoch_progress = "●" * min(10, self.experiment_status.current_epoch) + "○" * max(0, 10 - self.experiment_status.current_epoch)
        step_progress = f"Step {self.experiment_status.current_step}"
        
        table.add_row("Epoch", str(self.experiment_status.current_epoch), "", epoch_progress)
        table.add_row("Step", str(self.experiment_status.current_step), "", step_progress)
        table.add_row("Loss", current_loss, best_loss, "📉" if current_loss != "N/A" else "")
        table.add_row("Accuracy", current_acc, best_acc, "📈" if current_acc != "N/A" else "")
        table.add_row("Learning Rate", current_lr, "", "")
        
        # Throughput
        if self.throughput_history:
            avg_throughput = np.mean(list(self.throughput_history)[-10:])  # Last 10 measurements
            table.add_row("Throughput", f"{avg_throughput:.2f} steps/s", "", "⚡")
        
        return Panel(table)
    
    def _create_system_panel(self) -> Panel:
        """Create system status panel."""
        table = Table(title="🖥️ System Status", show_header=True, header_style="bold cyan")
        table.add_column("Resource", style="cyan")
        table.add_column("Usage", style="green")
        table.add_column("Status", style="white")
        
        if self.system_history:
            latest = self.system_history[-1]
            
            # CPU
            cpu_status = "🔥" if latest.cpu_percent > 80 else "✅" if latest.cpu_percent < 50 else "⚠️"
            table.add_row("CPU", f"{latest.cpu_percent:.1f}%", cpu_status)
            
            # Memory
            mem_status = "🔥" if latest.memory_percent > 80 else "✅" if latest.memory_percent < 50 else "⚠️"
            table.add_row("RAM", f"{latest.memory_percent:.1f}%", mem_status)
            
            # GPU
            if latest.gpu_memory_used is not None and latest.gpu_memory_total is not None:
                gpu_percent = (latest.gpu_memory_used / latest.gpu_memory_total) * 100 if latest.gpu_memory_total > 0 else 0
                gpu_status = "🔥" if gpu_percent > 80 else "✅" if gpu_percent < 50 else "⚠️"
                table.add_row("GPU Memory", f"{latest.gpu_memory_used:.1f}/{latest.gpu_memory_total:.1f}GB", gpu_status)
                
                if latest.gpu_utilization is not None:
                    util_status = "🔥" if latest.gpu_utilization > 80 else "✅"
                    table.add_row("GPU Util", f"{latest.gpu_utilization:.1f}%", util_status)
            
            # Disk
            if latest.disk_usage is not None:
                disk_status = "🔥" if latest.disk_usage > 90 else "✅" if latest.disk_usage < 70 else "⚠️"
                table.add_row("Disk", f"{latest.disk_usage:.1f}%", disk_status)
        
        return Panel(table)
    
    def _create_metrics_panel(self) -> Panel:
        """Create metrics panel."""
        # Simple text-based plots
        content = []
        
        if self.metrics_history:
            content.append("[bold]📈 Recent Metrics[/bold]\n")
            
            # Loss trend
            recent_losses = [m.get('loss') for m in list(self.metrics_history)[-20:] if m.get('loss') is not None]
            if recent_losses:
                content.append("Loss Trend:")
                # Simple ASCII plot
                min_loss, max_loss = min(recent_losses), max(recent_losses)
                if max_loss > min_loss:
                    for loss in recent_losses[-10:]:  # Last 10
                        normalized = (loss - min_loss) / (max_loss - min_loss)
                        bar_length = int(normalized * 20)
                        bar = "█" * bar_length + "░" * (20 - bar_length)
                        content.append(f"  {bar} {loss:.4f}")
                else:
                    content.append(f"  Stable at {recent_losses[-1]:.4f}")
                content.append("")
            
            # Accuracy trend
            recent_accs = [m.get('accuracy') for m in list(self.metrics_history)[-20:] if m.get('accuracy') is not None]
            if recent_accs:
                content.append("Accuracy Trend:")
                min_acc, max_acc = min(recent_accs), max(recent_accs)
                if max_acc > min_acc:
                    for acc in recent_accs[-10:]:  # Last 10
                        normalized = (acc - min_acc) / (max_acc - min_acc)
                        bar_length = int(normalized * 20)
                        bar = "█" * bar_length + "░" * (20 - bar_length)
                        content.append(f"  {bar} {acc:.3f}")
                else:
                    content.append(f"  Stable at {recent_accs[-1]:.3f}")
                content.append("")
        
        # Attention statistics
        if self.attention_stats:
            content.append("[bold]🎯 Attention Stats[/bold]\n")
            for key, values in list(self.attention_stats.items())[:5]:  # Show top 5
                if values:
                    latest_value = values[-1]['value']
                    content.append(f"{key}: {latest_value:.4f}")
        
        if not content:
            content = ["[dim]No metrics available yet...[/dim]"]
        
        text_content = "\n".join(content)
        return Panel(text_content, title="📊 Metrics", border_style="green")
    
    def _create_footer(self) -> Panel:
        """Create footer panel."""
        footer_text = Text()
        footer_text.append("Commands: ", style="bold")
        footer_text.append("[Q]uit ", style="red")
        footer_text.append("[R]eset ", style="yellow")
        footer_text.append("[S]ave Report ", style="green")
        footer_text.append("[P]ause ", style="blue")
        footer_text.append(f" | Update: {self.update_interval:.1f}s", style="dim")
        
        return Panel(footer_text, style="dim")
    
    def _start_live_display(self):
        """Start the live display."""
        try:
            self.live = Live(
                self._create_layout(),
                console=self.console,
                refresh_per_second=1.0 / self.update_interval,
                screen=True
            )
            self.live.start()
        except Exception as e:
            self.console.print(f"[red]Failed to start live display: {e}[/red]")
    
    def _save_monitoring_report(self):
        """Save monitoring report."""
        report_file = self.log_dir / f"{self.experiment_name}_monitor_report.json"
        
        report = {
            "experiment": {
                "name": self.experiment_name,
                "start_time": self.experiment_status.start_time,
                "end_time": time.time(),
                "final_epoch": self.experiment_status.current_epoch,
                "final_step": self.experiment_status.current_step,
                "final_loss": self.experiment_status.current_loss,
                "final_accuracy": self.experiment_status.current_accuracy,
            },
            "system_summary": {},
            "metrics_summary": {},
            "attention_summary": {}
        }
        
        # System summary
        if self.system_history:
            cpu_usage = [s.cpu_percent for s in self.system_history]
            memory_usage = [s.memory_percent for s in self.system_history]
            
            report["system_summary"] = {
                "cpu_avg": np.mean(cpu_usage),
                "cpu_max": np.max(cpu_usage),
                "memory_avg": np.mean(memory_usage),
                "memory_max": np.max(memory_usage),
            }
            
            # GPU summary if available
            gpu_memory_usage = [s.gpu_memory_used for s in self.system_history if s.gpu_memory_used is not None]
            if gpu_memory_usage:
                report["system_summary"]["gpu_memory_avg"] = np.mean(gpu_memory_usage)
                report["system_summary"]["gpu_memory_max"] = np.max(gpu_memory_usage)
        
        # Metrics summary
        if self.metrics_history:
            losses = [m.get('loss') for m in self.metrics_history if m.get('loss') is not None]
            accuracies = [m.get('accuracy') for m in self.metrics_history if m.get('accuracy') is not None]
            
            if losses:
                report["metrics_summary"]["loss"] = {
                    "final": losses[-1],
                    "best": min(losses),
                    "worst": max(losses),
                    "mean": np.mean(losses),
                    "std": np.std(losses)
                }
            
            if accuracies:
                report["metrics_summary"]["accuracy"] = {
                    "final": accuracies[-1],
                    "best": max(accuracies),
                    "worst": min(accuracies),
                    "mean": np.mean(accuracies),
                    "std": np.std(accuracies)
                }
        
        # Attention summary
        for key, values in self.attention_stats.items():
            if values:
                value_list = [v['value'] for v in values]
                report["attention_summary"][key] = {
                    "final": value_list[-1],
                    "mean": np.mean(value_list),
                    "std": np.std(value_list),
                    "min": min(value_list),
                    "max": max(value_list)
                }
        
        # Save report
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.console.print(f"📊 Monitoring report saved: {report_file}")
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


def create_monitor(experiment_name: str, **kwargs) -> ResearchMonitor:
    """Create a research monitor instance."""
    return ResearchMonitor(experiment_name, **kwargs)
