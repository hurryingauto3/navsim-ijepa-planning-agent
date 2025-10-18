#!/bin/bash
# =============================================================================
# Build a Container with NAVSIM Environment for Torch
# 
# Based on Torch documentation:
# - Writable overlays are NOT reliable
# - Need to build a proper container image
# - Or copy from Greene if you have one
# =============================================================================

set -e

echo "=============================================="
echo "NAVSIM Container Setup for Torch"
echo "=============================================="
echo ""

# This script creates a definition file for building a Singularity container
# with NAVSIM installed. You'll need to build this on a machine where you
# have sudo/root access (not on Torch itself).

DEFINITION_FILE="navsim_pytorch.def"

echo "Creating Singularity definition file: ${DEFINITION_FILE}"
echo ""

cat > ${DEFINITION_FILE} << 'EOF'
Bootstrap: docker
From: nvidia/cuda:12.8.1-cudnn9-devel-ubuntu24.04

%post
    # Update and install dependencies
    apt-get update && apt-get install -y \
        wget \
        git \
        build-essential \
        libgl1-mesa-glx \
        libglib2.0-0 \
        && rm -rf /var/lib/apt/lists/*
    
    # Install Miniconda
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p /opt/conda
    rm /tmp/miniconda.sh
    
    # Add conda to path
    export PATH="/opt/conda/bin:$PATH"
    
    # Initialize conda
    /opt/conda/bin/conda init bash
    
    # Create NAVSIM environment from scratch (since we can't copy environment.yml yet)
    /opt/conda/bin/conda create -n navsim python=3.10 -y
    
    # Activate and install packages
    . /opt/conda/etc/profile.d/conda.sh
    conda activate navsim
    
    # Install PyTorch with CUDA 12.8
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
    
    # Install common dependencies
    pip install \
        lightning \
        hydra-core \
        opencv-python \
        scipy \
        scikit-learn \
        pandas \
        matplotlib \
        tqdm \
        wandb \
        omegaconf \
        einops \
        timm
    
    # These will be installed when we mount the NAVSIM repo
    # pip install -e /scratch/ah7072/navsim_workspace/navsim

%environment
    export PATH="/opt/conda/bin:$PATH"
    export CONDA_DEFAULT_ENV=navsim

%runscript
    #!/bin/bash
    . /opt/conda/etc/profile.d/conda.sh
    conda activate navsim
    exec "$@"

%labels
    Author ali.hamza
    Version 1.0
    Description NAVSIM container with PyTorch and CUDA 12.8
EOF

echo "=============================================="
echo "Definition file created: ${DEFINITION_FILE}"
echo "=============================================="
echo ""
echo "Next steps:"
echo ""
echo "OPTION 1: Build locally (requires sudo)"
echo "  1. Transfer this file to a machine with sudo access"
echo "  2. Build: sudo singularity build navsim_pytorch.sif ${DEFINITION_FILE}"
echo "  3. Transfer to Torch: scp navsim_pytorch.sif cs649:/scratch/$USER/"
echo ""
echo "OPTION 2: Use existing Greene conda (RECOMMENDED)"
echo "  See: setup_greene_conda_transfer.sh"
echo ""
echo "OPTION 3: Use system Python on Torch (NOT RECOMMENDED)"
echo "  Torch docs warn against this - system will be upgraded"
echo ""
