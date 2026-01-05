# AdaAttn Research Environment
FROM nvidia/cuda:12.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    curl \
    wget \
    vim \
    htop \
    nvtop \
    tmux \
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    && rm -rf /var/lib/apt/lists/*

# Create research user
RUN useradd -m -s /bin/bash researcher && \
    usermod -aG sudo researcher && \
    echo "researcher ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Set working directory
WORKDIR /workspace/AdaAttn

# Copy requirements first for better caching
COPY requirements.txt ./
COPY pyproject.toml ./

# Install Python packages
RUN pip3 install --upgrade pip setuptools wheel && \
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip3 install flash-attn --no-build-isolation && \
    pip3 install -r requirements.txt

# Install Jupyter and research tools
RUN pip3 install \
    jupyter \
    jupyterlab \
    jupyter-contrib-nbextensions \
    matplotlib \
    seaborn \
    plotly \
    wandb \
    tensorboard \
    mlflow \
    optuna \
    hydra-core \
    rich \
    typer \
    click \
    tqdm \
    pandas \
    numpy \
    scipy \
    scikit-learn \
    transformers \
    datasets \
    accelerate \
    deepspeed \
    memory-profiler \
    py-spy \
    line_profiler

# Copy source code
COPY . .

# Install AdaAttn in development mode
RUN pip3 install -e .

# Set up Jupyter
RUN jupyter contrib nbextension install --system && \
    jupyter nbextension enable --py --sys-prefix widgetsnbextension

# Create research directories
RUN mkdir -p /workspace/experiments && \
    mkdir -p /workspace/logs && \
    mkdir -p /workspace/results && \
    mkdir -p /workspace/datasets && \
    mkdir -p /workspace/models && \
    mkdir -p /workspace/configs && \
    chown -R researcher:researcher /workspace

# Switch to research user
USER researcher

# Set up shell environment
RUN echo 'export PATH="/home/researcher/.local/bin:$PATH"' >> ~/.bashrc && \
    echo 'alias ll="ls -la"' >> ~/.bashrc && \
    echo 'alias jlab="jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root"' >> ~/.bashrc

# Expose ports for Jupyter, TensorBoard, MLflow
EXPOSE 8888 6006 5000

# Set entrypoint
COPY docker/entrypoint.sh /entrypoint.sh
USER root
RUN chmod +x /entrypoint.sh
USER researcher

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
