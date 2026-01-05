#!/bin/bash

# AdaAttn Research Environment Entrypoint

echo "🔬 AdaAttn Research Environment Starting..."
echo "🐳 Container: $(hostname)"
echo "🗂️  Workspace: $(pwd)"
echo ""

# Initialize research environment
if [ ! -f "/workspace/.initialized" ]; then
    echo "🚀 First time setup..."
    
    # Set up git if credentials are provided
    if [ ! -z "$GIT_USER_NAME" ] && [ ! -z "$GIT_USER_EMAIL" ]; then
        git config --global user.name "$GIT_USER_NAME"
        git config --global user.email "$GIT_USER_EMAIL"
        echo "✅ Git configured"
    fi
    
    # Initialize wandb if API key is provided
    if [ ! -z "$WANDB_API_KEY" ]; then
        wandb login $WANDB_API_KEY
        echo "✅ W&B initialized"
    fi
    
    # Create default experiment config
    cat > /workspace/configs/default.yaml << 'YAML'
# Default AdaAttn Research Configuration
experiment:
  name: "adaattn_research"
  project: "adaptive_attention"
  tags: ["research", "adaptive", "attention"]

model:
  hidden_size: 512
  num_heads: 8
  max_seq_len: 1024
  dropout: 0.1

attention:
  type: "adaattn"  # adaattn, adaptive_rank, adaptive_precision, dense
  enable_gpu_optimization: true
  rank_adaptation:
    enabled: true
    rank_ratio: 0.5
    entropy_threshold: 0.5
  precision_adaptation:
    enabled: true
    policy: "balanced"  # quality, balanced, speed

training:
  batch_size: 8
  learning_rate: 1e-4
  num_epochs: 10
  warmup_steps: 1000
  gradient_clipping: 1.0

logging:
  level: "INFO"
  log_attention_stats: true
  log_gpu_memory: true
  save_checkpoints: true
  checkpoint_freq: 1000

monitoring:
  wandb:
    enabled: false
    project: "adaattn_research"
  tensorboard:
    enabled: true
    log_dir: "/workspace/logs/tensorboard"
  mlflow:
    enabled: false
    tracking_uri: "file:///workspace/logs/mlflow"
YAML
    
    touch /workspace/.initialized
    echo "✅ Research environment initialized"
fi

# Display available commands
echo ""
echo "📚 Available Research Commands:"
echo "  jlab          - Start Jupyter Lab (port 8888)"
echo "  tensorboard   - Start TensorBoard (port 6006)"
echo "  mlflow        - Start MLflow UI (port 5000)"
echo "  research-cli  - Interactive research CLI"
echo "  run-experiment <config> - Run experiment with config"
echo ""

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "🔥 GPU Status:"
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits | \
        awk -F', ' '{printf "  %s: %.1f/%.1f GB (%.1f%% used)\n", $1, $3/1024, $2/1024, ($3/$2)*100}'
    echo ""
fi

# Check Python environment
echo "🐍 Python Environment:"
echo "  Python: $(python3 --version)"
echo "  PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "  CUDA Available: $(python3 -c 'import torch; print(torch.cuda.is_available())')"
if python3 -c 'import flash_attn' 2>/dev/null; then
    echo "  FlashAttention: ✅ Available"
else
    echo "  FlashAttention: ❌ Not available"
fi
echo "  AdaAttn: $(python3 -c 'import adaattn; print("✅ Ready")')"
echo ""

# Execute the command
exec "$@"
