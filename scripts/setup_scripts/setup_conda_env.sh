#!/bin/bash
# =============================================================================
# Setup Conda Environment for NAVSIM on Torch
# Run this once before using any training/caching scripts
# =============================================================================

set -e

echo "=============================================="
echo "Setting up NAVSIM Conda Environment"
echo "=============================================="
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please load conda module first:"
    echo "  module load anaconda3"
    exit 1
fi

# Navigate to NAVSIM workspace
NAVSIM_DIR="/scratch/ah7072/navsim_workspace/navsim"
cd "${NAVSIM_DIR}"

echo "Working directory: $(pwd)"
echo ""

# Check if environment already exists
ENV_NAME="navsim"
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "✓ Environment '${ENV_NAME}' already exists"
    echo ""
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n ${ENV_NAME} -y
    else
        echo "Using existing environment"
        exit 0
    fi
fi

echo "Creating conda environment from environment.yml..."
echo "This may take 10-15 minutes..."
echo ""

# Create environment
conda env create -n ${ENV_NAME} -f environment.yml

echo ""
echo "=============================================="
echo "Environment Setup Complete!"
echo "=============================================="
echo ""
echo "To activate the environment, run:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "Or it will be automatically activated in Slurm jobs."
echo ""
echo "Next steps:"
echo "  1. Run setup to download data: ./scripts/setup_all.sh"
echo "  2. Test with smoke test: sbatch scripts/torch_smoke_test.slurm"
echo ""
