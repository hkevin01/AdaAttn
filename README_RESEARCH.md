# 🔬 AdaAttn Research Environment

Complete setup for adaptive attention research with Docker, monitoring, and analysis tools.

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone repository
git clone <repository-url>
cd AdaAttn

# Start research environment
docker-compose up --build -d

# Access services
# - Jupyter Lab: http://localhost:8888
# - TensorBoard: http://localhost:6006  
# - MLflow UI: http://localhost:5001
```

### Option 2: Manual Setup

```bash
# Install dependencies
pip install -r requirements.txt
pip install -e .

# Test installation
python scripts/research_cli.py status

# Start research CLI
python scripts/research_cli.py
```

## 🛠️ Research Tools

### 1. Research CLI
Interactive command-line tool for experiments:

```bash
python scripts/research_cli.py

# Available commands:
# - quick-test: Test all attention mechanisms  
# - benchmark: Performance benchmarking
# - experiment: Start custom experiments
# - monitor: Real-time monitoring dashboard
# - config: Manage configurations
```

### 2. Jupyter Notebooks
Research quickstart notebooks:

```bash
jupyter lab examples/notebooks/01_research_quickstart.ipynb
```

### 3. Advanced Logging

```python
from adaattn.utils.research_logger import get_research_logger

logger = get_research_logger("my_experiment", enable_tensorboard=True)
logger.log_metrics(epoch=1, loss=0.5, accuracy=0.8)
```

### 4. Real-time Monitoring

```python
from adaattn.utils.research_monitor import create_monitor

with create_monitor("experiment") as monitor:
    monitor.update_experiment_status(epoch=1, loss=0.5)
```

## 📊 Experiment Management

### Configuration Files

```yaml
# configs/my_experiment.yaml
experiment:
  name: "adaptive_attention_research"
  description: "Testing adaptive mechanisms"

attention:
  type: "adaattn"
  enable_gpu_optimization: true
  rank_adaptation:
    enabled: true
    rank_ratio: 0.5
  precision_adaptation:
    enabled: true
    policy: "balanced"

logging:
  enable_tensorboard: true
  log_attention_stats: true
```

### Research Workflow

1. **Test Setup**: `python scripts/research_cli.py quick-test`
2. **Benchmark**: `python scripts/research_cli.py benchmark`
3. **Create Config**: Edit `configs/my_experiment.yaml`
4. **Run Experiment**: `python scripts/research_cli.py experiment`
5. **Monitor**: `python scripts/research_cli.py monitor`
6. **Analyze**: Check TensorBoard at http://localhost:6006

## 🔧 Environment Features

### Core Components
- ✅ AdaAttention with adaptive rank & precision
- ✅ GPU optimization support (CUDA + FlashAttention)
- ✅ Comprehensive benchmarking
- ✅ Real-time monitoring dashboard
- ✅ Multi-backend logging (TensorBoard, W&B, MLflow)
- ✅ Jupyter notebook environment
- ✅ Docker containerization

### Advanced Features  
- 📈 Performance profiling and memory tracking
- 🎯 Attention pattern analysis
- ⚙️ Hyperparameter optimization with Optuna
- 🔄 Experiment configuration management
- 📊 Automated result visualization
- 🔍 Hardware-aware optimization selection

### Integration Support
- **TensorBoard**: Automatic logging and visualization
- **W&B**: Cloud experiment tracking (set WANDB_API_KEY)
- **MLflow**: Model and experiment management
- **Docker**: Containerized research environment
- **Jupyter**: Interactive research notebooks

## 📁 Directory Structure

```
AdaAttn/
├── configs/              # Experiment configurations
├── logs/                 # All experiment logs
│   ├── tensorboard/      # TensorBoard logs
│   └── mlflow/          # MLflow artifacts
├── results/             # Experiment results
├── examples/notebooks/  # Jupyter research notebooks  
├── scripts/research_cli.py  # Main research interface
├── docker-compose.yml   # Docker environment
└── src/adaattn/utils/   # Research utilities
    ├── research_logger.py   # Advanced logging
    └── research_monitor.py  # Real-time monitoring
```

## 🎯 Research Use Cases

### 1. Attention Mechanism Comparison

```python
from adaattn.attention import AdaAttention, DenseAttention

# Compare performance
ada_attention = AdaAttention(512, 8, enable_gpu_optimization=True)
dense_attention = DenseAttention(512, 8)

# Benchmark both
python scripts/research_cli.py benchmark
```

### 2. Adaptive Behavior Analysis

```python
# Analyze adaptation patterns
logger.log_attention_analysis(attention_weights, "layer_0")

# Track statistics
stats = ada_attention.get_statistics()
logger.log_metrics(custom_metrics=stats)
```

### 3. Hyperparameter Optimization

```python
import optuna

def objective(trial):
    rank_ratio = trial.suggest_uniform('rank_ratio', 0.1, 0.9)
    attention = AdaAttention(512, 8, rank_ratio=rank_ratio)
    return evaluate_model(attention)

study = optuna.create_study()
study.optimize(objective, n_trials=100)
```

### 4. Production Deployment Testing

```python
# Test with different configurations
attention = AdaAttention(
    hidden_size=512,
    num_heads=8,
    enable_gpu_optimization=torch.cuda.is_available(),
    precision_policy="speed"  # quality, balanced, speed
)

# Monitor deployment metrics
with logger.timer("inference"):
    output = attention(query, key, value)
```

## 🔄 Advanced Workflows

### Distributed Training

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

# Multi-GPU setup
attention = AdaAttention(512, 8)
attention = DistributedDataParallel(attention)
```

### Memory Optimization

```python
# Enable memory tracking
logger.log_model_info(model)
logger.log_metrics(gpu_memory=torch.cuda.memory_allocated())

# Use memory-efficient variants
attention = AdaAttention(512, 8, memory_efficient=True)
```

### Custom Experiments

```python
# Create experiment
config = {
    'experiment': {'name': 'custom_research'},
    'attention': {'type': 'adaattn'},
    'logging': {'enable_tensorboard': True}
}

# Run with monitoring  
with create_monitor("custom_research") as monitor:
    for epoch in range(epochs):
        # Your training loop
        monitor.update_experiment_status(epoch=epoch, loss=loss)
```

## 📈 Performance Optimization

### GPU Acceleration
- CUDA kernel optimization
- FlashAttention integration  
- Mixed precision support
- Memory-efficient attention

### Monitoring & Profiling
- Real-time system monitoring
- GPU memory tracking
- Throughput measurement
- Bottleneck identification

### Adaptive Mechanisms
- Entropy-based rank selection
- Hardware-aware precision control
- Dynamic optimization switching
- Performance-quality trade-offs

## 🔍 Debugging & Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size or enable memory-efficient mode
   attention = AdaAttention(512, 8, memory_efficient=True)
   ```

2. **FlashAttention Not Available**
   ```bash
   pip install flash-attn --no-build-isolation
   ```

3. **Docker GPU Issues**
   ```bash
   # Install nvidia-docker2
   docker run --gpus all nvidia/cuda:12.1-base nvidia-smi
   ```

### Debug Mode

```bash
# Enable detailed logging
python scripts/research_cli.py --log-level=DEBUG

# Or in code
import logging
logging.getLogger('adaattn').setLevel(logging.DEBUG)
```

## 📚 Documentation & Resources

- [Setup Guide](docs/research_setup.md) - Detailed installation
- [API Documentation](docs/) - Technical documentation  
- [Examples](examples/notebooks/) - Research notebooks
- [Benchmarks](benchmarks/) - Performance analysis
- [Configuration](configs/) - Experiment templates

## 🎉 Getting Started

1. **Quick Test**: `python scripts/research_cli.py quick-test`
2. **Open Jupyter**: `jupyter lab examples/notebooks/01_research_quickstart.ipynb`
3. **Start Experiment**: `python scripts/research_cli.py experiment`
4. **Monitor Progress**: TensorBoard at http://localhost:6006

Ready to start your adaptive attention research! 🚀
