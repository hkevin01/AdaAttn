#!/usr/bin/env python3
"""
AdaAttn Research CLI Tool

A comprehensive command-line interface for running AdaAttn research experiments.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import torch
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
from rich.tree import Tree

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adaattn.utils.research_logger import ResearchLogger, get_research_logger
from adaattn.utils.research_monitor import ResearchMonitor
from adaattn.attention import AdaAttention, AdaptiveRankAttention, AdaptivePrecisionAttention, DenseAttention


class ResearchCLI:
    """Interactive research CLI for AdaAttn experiments."""
    
    def __init__(self):
        self.console = Console()
        self.current_experiment = None
        self.logger = None
        self.monitor = None
        
        # Project paths
        self.project_root = Path(__file__).parent.parent
        self.configs_dir = self.project_root / "configs"
        self.logs_dir = self.project_root / "logs"
        self.results_dir = self.project_root / "results"
        
        # Ensure directories exist
        self.configs_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
    
    def print_banner(self):
        """Print welcome banner."""
        banner = """
    ╔═══════════════════════════════════════════════════╗
    ║                                                   ║
    ║          🔬 AdaAttn Research CLI v1.0             ║
    ║                                                   ║
    ║        Adaptive Attention Research Platform       ║
    ║                                                   ║
    ╚═══════════════════════════════════════════════════╝
        """
        self.console.print(banner, style="bold blue")
    
    def show_main_menu(self):
        """Show the main menu."""
        table = Table(title="🔬 Research Commands", show_header=False)
        table.add_column("Command", style="cyan", width=20)
        table.add_column("Description", style="white")
        
        commands = [
            ("1. quick-test", "Run quick attention mechanism test"),
            ("2. benchmark", "Run comprehensive benchmarks"),
            ("3. experiment", "Start custom experiment"),
            ("4. monitor", "Launch real-time monitoring"),
            ("5. analysis", "Analyze experiment results"),
            ("6. config", "Manage experiment configurations"),
            ("7. status", "Check system and environment status"),
            ("8. help", "Show detailed help"),
            ("9. exit", "Exit the research CLI")
        ]
        
        for cmd, desc in commands:
            table.add_row(cmd, desc)
        
        self.console.print(table)
        self.console.print()
    
    def check_system_status(self):
        """Check and display system status."""
        self.console.print("🖥️ [bold]System Status[/bold]")
        
        # Python environment
        self.console.print(f"Python: {sys.version.split()[0]}")
        
        # PyTorch status
        self.console.print(f"PyTorch: {torch.__version__}")
        self.console.print(f"CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            self.console.print(f"CUDA Devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                device_name = torch.cuda.get_device_name(i)
                memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                self.console.print(f"  GPU {i}: {device_name} ({memory_total:.1f} GB)")
        
        # AdaAttn components
        try:
            from adaattn.attention import AdaAttention
            self.console.print("✅ AdaAttention: Available")
        except ImportError as e:
            self.console.print(f"❌ AdaAttention: {e}")
        
        # FlashAttention
        try:
            import flash_attn
            self.console.print("✅ FlashAttention: Available")
        except ImportError:
            self.console.print("⚠️  FlashAttention: Not available (optional)")
        
        self.console.print()
    
    def run_quick_test(self):
        """Run a quick test of attention mechanisms."""
        self.console.print("🚀 [bold]Running Quick Attention Test[/bold]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            
            # Test parameters
            batch_size = 2
            seq_len = 128
            hidden_size = 256
            num_heads = 8
            
            task = progress.add_task("Setting up test...", total=100)
            
            # Create test data
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            query = torch.randn(batch_size, seq_len, hidden_size, device=device)
            key = torch.randn(batch_size, seq_len, hidden_size, device=device)
            value = torch.randn(batch_size, seq_len, hidden_size, device=device)
            
            progress.update(task, advance=20, description="Creating attention modules...")
            
            # Test different attention types
            attention_types = {
                "Dense": DenseAttention(hidden_size, num_heads),
                "AdaptiveRank": AdaptiveRankAttention(hidden_size, num_heads),
                "AdaptivePrecision": AdaptivePrecisionAttention(hidden_size, num_heads),
                "AdaAttention": AdaAttention(hidden_size, num_heads, enable_gpu_optimization=torch.cuda.is_available())
            }
            
            results = {}
            
            for i, (name, attention) in enumerate(attention_types.items()):
                progress.update(task, advance=15, description=f"Testing {name}...")
                
                attention = attention.to(device)
                attention.eval()
                
                with torch.no_grad():
                    start_time = time.time()
                    
                    try:
                        output = attention(query, key, value)
                        end_time = time.time()
                        
                        results[name] = {
                            "success": True,
                            "time": end_time - start_time,
                            "output_shape": list(output.shape),
                            "memory_usage": torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
                        }
                        
                        # Get statistics if available
                        if hasattr(attention, 'get_statistics'):
                            stats = attention.get_statistics()
                            results[name]["statistics"] = stats
                        
                    except Exception as e:
                        results[name] = {
                            "success": False,
                            "error": str(e)
                        }
            
            progress.update(task, advance=25, description="Completed!")
        
        # Display results
        self.console.print("\n📊 [bold]Test Results[/bold]")
        
        table = Table()
        table.add_column("Attention Type", style="cyan")
        table.add_column("Status", style="white")
        table.add_column("Time (ms)", style="green")
        table.add_column("Memory (GB)", style="yellow")
        table.add_column("Notes", style="white")
        
        for name, result in results.items():
            if result["success"]:
                status = "✅ Success"
                time_ms = f"{result['time'] * 1000:.2f}"
                memory_gb = f"{result['memory_usage']:.3f}"
                notes = f"Shape: {result['output_shape']}"
                
                if "statistics" in result:
                    stats = result["statistics"]
                    if "low_rank_usage" in stats:
                        notes += f", Low-rank: {stats['low_rank_usage']:.1%}"
            else:
                status = "❌ Failed"
                time_ms = "N/A"
                memory_gb = "N/A"
                notes = result["error"]
            
            table.add_row(name, status, time_ms, memory_gb, notes)
        
        self.console.print(table)
        self.console.print()
    
    def run_benchmark(self):
        """Run comprehensive benchmarks."""
        self.console.print("⚡ [bold]Running Comprehensive Benchmarks[/bold]")
        
        # Get benchmark parameters
        batch_sizes = [1, 2, 4]
        seq_lengths = [128, 512, 1024]
        
        if Confirm.ask("Run extended benchmarks? (includes larger sequences)", default=False):
            batch_sizes.extend([8, 16])
            seq_lengths.extend([2048, 4096])
        
        hidden_size = int(Prompt.ask("Hidden size", default="512"))
        num_heads = int(Prompt.ask("Number of heads", default="8"))
        
        # Initialize logger
        experiment_name = f"benchmark_{int(time.time())}"
        self.logger = get_research_logger(
            experiment_name=experiment_name,
            log_dir=str(self.logs_dir),
            enable_tensorboard=True
        )
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.console.print(f"Running on: {device}")
        
        # Benchmark configurations
        configs = {
            "dense": DenseAttention(hidden_size, num_heads),
            "adaptive_rank": AdaptiveRankAttention(hidden_size, num_heads),
            "adaptive_precision": AdaptivePrecisionAttention(hidden_size, num_heads),
            "ada_attention": AdaAttention(hidden_size, num_heads, enable_gpu_optimization=torch.cuda.is_available())
        }
        
        results = {}
        
        total_tests = len(batch_sizes) * len(seq_lengths) * len(configs)
        
        with Progress(console=self.console) as progress:
            task = progress.add_task("Running benchmarks...", total=total_tests)
            
            for batch_size in batch_sizes:
                for seq_len in seq_lengths:
                    for config_name, attention in configs.items():
                        
                        progress.update(task, description=f"B{batch_size}_S{seq_len}_{config_name}")
                        
                        # Setup
                        attention = attention.to(device)
                        attention.eval()
                        
                        query = torch.randn(batch_size, seq_len, hidden_size, device=device)
                        key = torch.randn(batch_size, seq_len, hidden_size, device=device)
                        value = torch.randn(batch_size, seq_len, hidden_size, device=device)
                        
                        # Warm up
                        for _ in range(3):
                            with torch.no_grad():
                                _ = attention(query, key, value)
                        
                        # Benchmark
                        times = []
                        memory_usage = []
                        
                        for _ in range(5):  # 5 runs
                            torch.cuda.empty_cache() if torch.cuda.is_available() else None
                            
                            if torch.cuda.is_available():
                                torch.cuda.reset_max_memory_allocated()
                            
                            start_time = time.time()
                            
                            with torch.no_grad():
                                output = attention(query, key, value)
                            
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                            
                            end_time = time.time()
                            
                            times.append(end_time - start_time)
                            
                            if torch.cuda.is_available():
                                memory_usage.append(torch.cuda.max_memory_allocated() / 1024**3)
                        
                        # Store results
                        key = f"{config_name}_B{batch_size}_S{seq_len}"
                        results[key] = {
                            "config": config_name,
                            "batch_size": batch_size,
                            "seq_len": seq_len,
                            "mean_time": np.mean(times),
                            "std_time": np.std(times),
                            "mean_memory": np.mean(memory_usage) if memory_usage else 0,
                            "output_shape": list(output.shape)
                        }
                        
                        # Log to research logger
                        self.logger.log_metrics(
                            custom_metrics={
                                f"benchmark/{key}/time": np.mean(times),
                                f"benchmark/{key}/memory": np.mean(memory_usage) if memory_usage else 0
                            }
                        )
                        
                        progress.update(task, advance=1)
        
        # Save benchmark results
        results_file = self.results_dir / f"benchmark_results_{experiment_name}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Display summary
        self.console.print("\n📊 [bold]Benchmark Summary[/bold]")
        
        # Create summary table
        table = Table()
        table.add_column("Configuration", style="cyan")
        table.add_column("Avg Time (ms)", style="green")
        table.add_column("Avg Memory (GB)", style="yellow")
        table.add_column("Best Config", style="white")
        
        # Aggregate by configuration
        config_summary = {}
        for key, result in results.items():
            config = result["config"]
            if config not in config_summary:
                config_summary[config] = {"times": [], "memories": []}
            
            config_summary[config]["times"].append(result["mean_time"])
            config_summary[config]["memories"].append(result["mean_memory"])
        
        best_time_config = min(config_summary.keys(), 
                              key=lambda c: np.mean(config_summary[c]["times"]))
        best_memory_config = min(config_summary.keys(), 
                                key=lambda c: np.mean(config_summary[c]["memories"]))
        
        for config, data in config_summary.items():
            avg_time = np.mean(data["times"]) * 1000  # Convert to ms
            avg_memory = np.mean(data["memories"])
            
            best = ""
            if config == best_time_config:
                best += "⚡ Fastest "
            if config == best_memory_config:
                best += "💾 Memory Efficient"
            
            table.add_row(config, f"{avg_time:.2f}", f"{avg_memory:.3f}", best)
        
        self.console.print(table)
        self.console.print(f"\n📁 Detailed results saved to: {results_file}")
        
        if self.logger:
            self.logger.cleanup()
    
    def start_custom_experiment(self):
        """Start a custom experiment."""
        self.console.print("🧪 [bold]Custom Experiment Setup[/bold]")
        
        # Load or create config
        config_files = list(self.configs_dir.glob("*.yaml"))
        
        if config_files:
            self.console.print("Available configurations:")
            for i, config_file in enumerate(config_files):
                self.console.print(f"  {i+1}. {config_file.stem}")
            self.console.print(f"  {len(config_files)+1}. Create new configuration")
            
            choice = int(Prompt.ask("Select configuration", default=str(len(config_files)+1)))
            
            if choice <= len(config_files):
                config_file = config_files[choice-1]
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                self.console.print(f"Loaded configuration: {config_file.stem}")
            else:
                config = self._create_config()
        else:
            config = self._create_config()
        
        # Start experiment
        experiment_name = config["experiment"]["name"]
        self.current_experiment = experiment_name
        
        # Initialize logging
        self.logger = get_research_logger(
            experiment_name=experiment_name,
            log_dir=str(self.logs_dir),
            **config.get("logging", {})
        )
        
        self.console.print(f"🚀 Started experiment: {experiment_name}")
        self.console.print("Use the monitoring command to track progress in real-time.")
    
    def _create_config(self) -> Dict[str, Any]:
        """Create a new experiment configuration."""
        self.console.print("Creating new experiment configuration...")
        
        name = Prompt.ask("Experiment name", default=f"experiment_{int(time.time())}")
        
        config = {
            "experiment": {
                "name": name,
                "description": Prompt.ask("Description", default="AdaAttn research experiment")
            },
            "model": {
                "hidden_size": int(Prompt.ask("Hidden size", default="512")),
                "num_heads": int(Prompt.ask("Number of heads", default="8")),
                "max_seq_len": int(Prompt.ask("Max sequence length", default="1024"))
            },
            "attention": {
                "type": Prompt.ask("Attention type", choices=["ada", "adaptive_rank", "adaptive_precision", "dense"], default="ada"),
                "enable_gpu_optimization": Confirm.ask("Enable GPU optimization?", default=True)
            },
            "logging": {
                "level": "INFO",
                "enable_tensorboard": Confirm.ask("Enable TensorBoard?", default=True),
                "enable_wandb": Confirm.ask("Enable W&B?", default=False)
            }
        }
        
        # Save configuration
        config_file = self.configs_dir / f"{name}.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f, indent=2)
        
        self.console.print(f"Configuration saved: {config_file}")
        return config
    
    def launch_monitor(self):
        """Launch real-time monitoring."""
        if not self.current_experiment:
            self.console.print("⚠️  No active experiment. Start an experiment first.")
            return
        
        self.console.print(f"🔍 Launching monitor for: {self.current_experiment}")
        
        try:
            self.monitor = ResearchMonitor(
                experiment_name=self.current_experiment,
                log_dir=str(self.logs_dir)
            )
            
            with self.monitor:
                self.console.print("Monitor started. Press Ctrl+C to stop.")
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    self.console.print("\nStopping monitor...")
                    
        except Exception as e:
            self.console.print(f"❌ Failed to start monitor: {e}")
    
    def analyze_results(self):
        """Analyze experiment results."""
        self.console.print("📊 [bold]Experiment Analysis[/bold]")
        
        # Find experiment log files
        log_files = list(self.logs_dir.glob("*.log"))
        result_files = list(self.results_dir.glob("*.json"))
        
        if not log_files and not result_files:
            self.console.print("No experiment results found.")
            return
        
        self.console.print("Available experiments:")
        
        all_files = log_files + result_files
        for i, file_path in enumerate(all_files[:10]):  # Show top 10
            self.console.print(f"  {i+1}. {file_path.stem}")
        
        if len(all_files) > 10:
            self.console.print(f"  ... and {len(all_files) - 10} more")
        
        # Basic analysis would go here
        # For now, just show file info
        self.console.print(f"\n📁 Total log files: {len(log_files)}")
        self.console.print(f"📁 Total result files: {len(result_files)}")
    
    def manage_configs(self):
        """Manage experiment configurations."""
        self.console.print("⚙️ [bold]Configuration Management[/bold]")
        
        config_files = list(self.configs_dir.glob("*.yaml"))
        
        if not config_files:
            self.console.print("No configurations found.")
            if Confirm.ask("Create a new configuration?"):
                self._create_config()
            return
        
        self.console.print("Available configurations:")
        for i, config_file in enumerate(config_files):
            self.console.print(f"  {i+1}. {config_file.stem}")
        
        self.console.print(f"  {len(config_files)+1}. Create new configuration")
        self.console.print(f"  {len(config_files)+2}. Delete configuration")
        
        choice = int(Prompt.ask("Select action", default="1"))
        
        if choice <= len(config_files):
            # View configuration
            config_file = config_files[choice-1]
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            self.console.print(f"\n📋 Configuration: {config_file.stem}")
            self.console.print(yaml.dump(config, indent=2))
            
        elif choice == len(config_files) + 1:
            # Create new
            self._create_config()
            
        elif choice == len(config_files) + 2:
            # Delete configuration
            self.console.print("Select configuration to delete:")
            for i, config_file in enumerate(config_files):
                self.console.print(f"  {i+1}. {config_file.stem}")
            
            del_choice = int(Prompt.ask("Delete which config?")) - 1
            if 0 <= del_choice < len(config_files):
                config_file = config_files[del_choice]
                if Confirm.ask(f"Delete {config_file.stem}?", default=False):
                    config_file.unlink()
                    self.console.print(f"Deleted: {config_file.stem}")
    
    def show_help(self):
        """Show detailed help."""
        help_text = """
🔬 AdaAttn Research CLI Help

COMMANDS:
  quick-test    - Run a fast test of all attention mechanisms
  benchmark     - Comprehensive performance benchmarking
  experiment    - Start a custom research experiment
  monitor       - Real-time monitoring dashboard
  analysis      - Analyze experiment results
  config        - Manage experiment configurations
  status        - Check system and environment status

WORKFLOW:
  1. Check system status
  2. Run quick test to verify everything works
  3. Create/select experiment configuration
  4. Start experiment
  5. Use monitor to track progress
  6. Analyze results

TIPS:
  - Use GPU when available for best performance
  - TensorBoard logs are saved automatically
  - Experiment configs are saved in configs/
  - Results are saved in results/
  - Logs are saved in logs/

For more details, see docs/ directory.
        """
        
        self.console.print(Panel(help_text, title="Help", border_style="blue"))
    
    def run(self):
        """Run the interactive CLI."""
        self.print_banner()
        self.check_system_status()
        
        while True:
            self.show_main_menu()
            
            try:
                choice = Prompt.ask("Enter command", default="1").strip().lower()
                
                if choice in ["1", "quick-test", "test"]:
                    self.run_quick_test()
                elif choice in ["2", "benchmark", "bench"]:
                    self.run_benchmark()
                elif choice in ["3", "experiment", "exp"]:
                    self.start_custom_experiment()
                elif choice in ["4", "monitor", "mon"]:
                    self.launch_monitor()
                elif choice in ["5", "analysis", "analyze"]:
                    self.analyze_results()
                elif choice in ["6", "config", "cfg"]:
                    self.manage_configs()
                elif choice in ["7", "status", "stat"]:
                    self.check_system_status()
                elif choice in ["8", "help", "h"]:
                    self.show_help()
                elif choice in ["9", "exit", "quit", "q"]:
                    break
                else:
                    self.console.print("❌ Invalid choice. Try again.")
                    
            except KeyboardInterrupt:
                self.console.print("\n👋 Goodbye!")
                break
            except Exception as e:
                self.console.print(f"❌ Error: {e}")
        
        # Cleanup
        if self.logger:
            self.logger.cleanup()
        if self.monitor:
            self.monitor.stop()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="AdaAttn Research CLI")
    parser.add_argument("command", nargs="?", help="Command to run directly")
    args = parser.parse_args()
    
    cli = ResearchCLI()
    
    if args.command:
        # Direct command execution
        if args.command == "status":
            cli.check_system_status()
        elif args.command == "test":
            cli.run_quick_test()
        elif args.command == "benchmark":
            cli.run_benchmark()
        else:
            cli.console.print(f"Unknown command: {args.command}")
    else:
        # Interactive mode
        cli.run()


if __name__ == "__main__":
    main()
