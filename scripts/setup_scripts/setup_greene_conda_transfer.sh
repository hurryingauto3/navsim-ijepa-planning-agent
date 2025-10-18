#!/bin/bash
# =============================================================================
# Simple Solution: Transfer Your Greene Conda Environment to Torch
# 
# Based on Torch docs: "If you already have container-based Conda environments 
# on Greene, they can be copied to Torch with minor edits."
# =============================================================================

set -e

echo "=============================================="
echo "Greene → Torch Environment Transfer"
echo "=============================================="
echo ""

echo "Do you have a working NAVSIM environment on Greene? (y/n)"
read -r response

if [[ "$response" != "y" ]]; then
    echo ""
    echo "You need to either:"
    echo "  1. Set up NAVSIM on Greene first, then copy to Torch"
    echo "  2. Build a Singularity container with NAVSIM"
    echo "  3. Use a pre-built PyTorch container"
    echo ""
    echo "See: https://sites.google.com/nyu.edu/nyu-hpc/hpc-systems/torch for options"
    exit 1
fi

echo ""
echo "Option 1: Copy entire conda environment from Greene"
echo "=========================================="
echo "On Greene, run:"
echo "  conda pack -n navsim -o navsim_env.tar.gz"
echo ""
echo "Then on Torch, run:"
echo "  mkdir -p ~/.conda/envs/navsim"
echo "  cd ~/.conda/envs/navsim"
echo "  tar -xzf /path/to/navsim_env.tar.gz"
echo "  conda-unpack"
echo ""
echo ""
echo "Option 2: Mount Greene home directory (if available)"
echo "=========================================="
echo "Check if Greene home is accessible from Torch:"
echo "  ls /gpfsnyu/home/$USER"
echo ""
echo "If yes, you can directly use Greene conda environment"
echo ""
echo ""
echo "Option 3: Recreate environment on Torch"
echo "=========================================="
echo "On Torch:"
echo "  cd /scratch/$USER/navsim_workspace/navsim"
echo "  conda env create -n navsim -f environment.yml"
echo ""
echo "WARNING: Torch docs say NOT to install directly on host"
echo "         Host system will be upgraded and may break packages"
echo ""
