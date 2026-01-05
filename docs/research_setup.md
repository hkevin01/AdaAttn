# AdaAttn Research Setup Guide

This guide will help you set up a complete research environment for AdaAttn experiments.

## Quick Start with Docker

### Prerequisites

- Docker and Docker Compose
- NVIDIA Docker runtime (for GPU support)
- Git

### 1. Clone and Build

```bash
git clone <repository-url>
cd AdaAttn
docker-compose up --build -d
```

### 2. Access Services

- **Jupyter Lab**: http://localhost:8888 (no token required)
- **TensorBoard**: http://localhost:6006
- **MLflow UI**: http://localhost:5001

### 3. Start Research

Open Jupyter Lab and navigate to `examples/notebooks/01_research_quickstart.ipynb`.

## Manual Setup

### Prerequisites

- Python 3.11+
- CUDA 12.1+ (optional, for GPU acceleration)
- Git

### 1. Create Virtual Environment

```bash
python -m venv adaattn_env
source adaattn_env/bin/activate  # Linux/Mac
# or
adaattn_env\Scripts\activate     # Windows
```

### 2. Install Dependencies

```bash
pip install --upgrade pip

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install FlashAttention (optional, but recommended for GPU)
pip install flash-attn --no-build-isolation

# Install AdaAttn and dependencies
pip install -e .

# Install research tools
pip install jupyter jupyterlab matplotlib seaborn plotly
pip install wandb tensorboard mlflow optuna hydra-core
pip install rich typer click tqdm pandas numpy scipy scikit-learn
pip install transformers datasets accelerate deepspeed
pip install memory-profiler py-spy line_profiler
```

### 3. Verify Installation

```bash
python scripts/research_cli.py status
```

## Research Environment Components

### 1. Research Logger

Advanced logging system with multiple backends:

```python
from adaattn.utils.research_logger import get_research_logger

logger = get_research_logger(
    experiment_name="my_experiment",
    enable_tensorboard=True,
    enable_wandb=False,
    enable_mlflow=False
)

# Log metrics
logger.log_metrics(
    epoch=1,
    step=100,
    loss=0.5,
    accuracy=0.8
)
```

### 2. Real-time Monitor

Live monitoring dashboard:

```python
from adaattn.utils.research_monitor import create_monitor

with create_monitor("my_experiment") as monitor:
    # Your training loop here
    monitor.update_experiment_status(
        epoch=1,
        step=100,
        loss=0.5
    )
```

### 3. Research CLI

Interactive command-line interface:

```bash
python scripts/research_cli.py

# Or direct commands
python scripts/research_cli.py test
python scripts/research_cli.py benchmark
```

## Directory Structure

```
AdaAttn/
├── configs/              # Experiment configurations
├── logs/                 # All logging outputs
│   ├── tensorboard/      # TensorBoard logs
│   ├── mlflow/          # MLflow artifacts
│   └── *.log            # Text log files
├── results/             # Experiment results
├── datasets/            # Research datasets
├── models/              # Trained model checkpoints
├── examples/
│   └── notebooks/       # Jupyter research notebooks
├── scripts/
│   └── research_cli.py  # Main research CLI
└── src/adaattn/
    └── utils/
        ├── research_logger.py   # Advanced logging
        └── research_monitor.py  # Real-time monitoring
```

## Configuration Management

### Creating Experiment Configs

```yaml
# configs/my_experiment.yaml
experiment:
  name: "attention_comparison"
  description: "Compare adaptive vs standard attention"
  tags: ["research", "comparison"]

model:
  hidden_size: 512
  num_heads: 8
  max_seq_len: 1024
  dropout: 0.1

attention:
  type: "adaattn"
  enable_gpu_optimization: true
  rank_adaptation:
    enabled: true
    rank_ratio: 0.5
    entropy_threshold: 0.5
  precision_adaptation:
    enabled: true
    policy: "balanced"

training:
  batch_size: 8
  learning_rate: 1e-4
  num_epochs: 10

logging:
  level: "INFO"
  log_attention_stats: true
  log_gpu_memory: true
  enable_tensorboard: true
  enable_wandb: false
```

### Loading Configs

```python
import yaml
with open('configs/my_experiment.yaml', 'r') as f:
    config = yaml.safe_load(f)
```

## Research Workflows

### 1. Quick Testing

```bash
# Test all attention mechanisms
python scripts/research_cli.py quick-test

# Or via notebook
jupyter lab examples/notebooks/01_research_quickstart.ipynb
```

### 2. Benchmarking

```bash
# Interactive benchmarking
python scripts/research_cli.py benchmark

# Or programmatically
python -c "
from adaattn.attention import *
# Run your benchmarks
"
```

### 3. Custom Experiments

```python
from adaattn.utils.research_logger import get_research_logger
from adaattn.attention import AdaAttention

# Setup experiment
logger = get_research_logger("my_experiment")
attention = AdaAttention(512, 8, enable_gpu_optimization=True)

# Your research code here
for epoch in range(10):
    # Training step
    loss = train_step(attention, data)
    
    # Log metrics
    logger.log_metrics(epoch=epoch, loss=loss)
    
    # Log attention statistics
    stats = attention.get_statistics()
    logger.log_metrics(custom_metrics=stats)

# Cleanup
logger.cleanup()
```

### 4. Real-time Monitoring

```python
from adaattn.utils.research_monitor import create_monitor

with create_monitor("my_experiment") as monitor:
    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            # Training step
            loss, accuracy = train_step(batch)
            
            # Update monitor
            monitor.update_experiment_status(
                epoch=epoch,
                step=step,
                loss=loss,
                accuracy=accuracy
            )
```

## Integration with External Tools

### W&B (Weights & Biases)

```bash
# Set API key
export WANDB_API_KEY=your_key

# Enable in logger
logger = get_research_logger(
    experiment_name="my_experiment",
    enable_wandb=True
)
```

### TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir=logs/tensorboard --port=6006

# Or use Docker Compose (automatic)
docker-compose up tensorboard
```

### MLflow

```bash
# Start MLflow UI
mlflow ui --backend-store-uri file:///path/to/logs/mlflow

# Or use Docker Compose (automatic)
docker-compose up mlflow-server
```

## Performance Optimization

### GPU Setup

1. Ensure CUDA is properly installed
2. Install FlashAttention for best performance:
   ```bash
   pip install flash-attn --no-build-isolation
   ```
3. Enable GPU optimization in AdaAttn:
   ```python
   attention = AdaAttention(512, 8, enable_gpu_optimization=True)
   ```

### Memory Management

```python
# Monitor memory usage
from adaattn.utils.research_logger import get_research_logger

logger = get_research_logger("experiment")
logger.log_model_info(model)  # Log parameter counts

# Enable memory logging
logger.log_metrics(
    gpu_memory_used=torch.cuda.memory_allocated() / 1024**3
)
```

### Profiling

```python
# Use context manager for timing
with logger.timer("forward_pass"):
    output = attention(query, key, value)

# Memory profiling
from memory_profiler import profile

@profile
def my_training_function():
    # Your training code
    pass
```

## Best Practices

### 1. Experiment Organization

- Use descriptive experiment names
- Tag experiments consistently
- Save configurations for reproducibility
- Document hypothesis and goals

### 2. Logging Strategy

- Log early and often
- Include system metrics (GPU, memory)
- Save intermediate checkpoints
- Use consistent metric names

### 3. Monitoring

- Start monitoring before training
- Set up alerts for resource usage
- Monitor adaptive behavior trends
- Save monitoring reports

### 4. Results Management

- Export results in multiple formats
- Create summary visualizations
- Archive experiment artifacts
- Document conclusions

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Enable gradient checkpointing
   - Use mixed precision training

2. **FlashAttention Installation**
   ```bash
   pip install flash-attn --no-build-isolation --no-cache-dir
   ```

3. **Docker GPU Access**
   ```bash
   # Install nvidia-docker2
   docker run --gpus all nvidia/cuda:12.1-base nvidia-smi
   ```

4. **Port Conflicts**
   - Check `docker-compose.yml` ports
   - Use different ports if needed
   - Kill existing services

### Debug Mode

```bash
# Enable debug logging
python scripts/research_cli.py --log-level=DEBUG

# Or in code
import logging
logging.getLogger('adaattn').setLevel(logging.DEBUG)
```

## Advanced Features

### Custom Kernels

```python
# Enable custom CUDA kernels (when available)
from adaattn.kernels import CUDAManager

if CUDAManager.is_available():
    attention = AdaAttention(
        512, 8,
        enable_gpu_optimization=True,
        kernel_selection="auto"  # auto, flash, sdpa, manual
    )
```

### Distributed Training

```python
# Multi-GPU setup
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

# Initialize process group
dist.init_process_group("nccl")

# Wrap model
attention = AdaAttention(512, 8)
attention = DistributedDataParallel(attention)
```

### Experiment Sweeps

```python
import optuna

def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    rank_ratio = trial.suggest_uniform('rank_ratio', 0.1, 0.9)
    
    # Train model with suggested params
    accuracy = train_model(lr=lr, rank_ratio=rank_ratio)
    
    return accuracy

# Run sweep
study = optuna.create_study()
study.optimize(objective, n_trials=100)
```

## Resources

- [AdaAttn Documentation](../docs/)
- [Example Notebooks](../examples/notebooks/)
- [Configuration Examples](../configs/)
- [Benchmarking Results](../results/)

## Support

For questions and issues:
1. Check the documentation
2. Run the system status check
3. Look at example notebooks
4. File an issue on GitHub
