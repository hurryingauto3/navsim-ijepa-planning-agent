#!/bin/bash
# =============================================================================
# One-Command Setup for NAVSIM on Torch
# This automates the entire setup process
# =============================================================================

set -e

echo "=============================================="
echo "NAVSIM Torch Setup - One-Click Install"
echo "=============================================="
echo ""
echo "This will:"
echo "  1. Load anaconda3 module"
echo "  2. Create navsim conda environment"
echo "  3. Verify PyTorch + CUDA"
echo "  4. Set up environment variables in ~/.bashrc"
echo ""
echo "Time estimate: 10-15 minutes"
echo ""
read -p "Continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Step 1: Load module
echo ""
echo "=============================================="
echo "Step 1/4: Loading anaconda3 module"
echo "=============================================="
module load anaconda3/2025.06

# Verify conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found after loading module"
    exit 1
fi

echo "✓ Conda available: $(which conda)"
conda --version

# Step 2: Create environment
echo ""
echo "=============================================="
echo "Step 2/4: Creating navsim conda environment"
echo "=============================================="
echo "This may take 10-15 minutes..."
echo ""

NAVSIM_DIR="/scratch/ah7072/navsim_workspace/navsim"
cd "${NAVSIM_DIR}"

# Check if environment already exists
if conda env list | grep -q "^navsim "; then
    echo "Environment 'navsim' already exists."
    read -p "Recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n navsim -y
    else
        echo "Using existing environment"
    fi
fi

# Create if doesn't exist
if ! conda env list | grep -q "^navsim "; then
    conda env create -n navsim -f environment.yml
fi

echo "✓ Environment created"

# Step 3: Verify installation
echo ""
echo "=============================================="
echo "Step 3/4: Verifying PyTorch + CUDA"
echo "=============================================="

eval "$(conda shell.bash hook)"
conda activate navsim

python -c "
import torch
import sys
print(f'✓ Python: {sys.version.split()[0]}')
print(f'✓ PyTorch: {torch.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version: {torch.version.cuda}')
    print(f'  GPU count: {torch.cuda.device_count()}')
else:
    print('  Note: CUDA check from login node - will work on compute node')
"

echo ""
echo "✓ Verification complete"

# Step 4: Set up bashrc
echo ""
echo "=============================================="
echo "Step 4/4: Configuring environment variables"
echo "=============================================="

BASHRC="$HOME/.bashrc"
MARKER="# NAVSIM Torch Setup"

# Remove old configuration if exists
if grep -q "$MARKER" "$BASHRC"; then
    echo "Removing old configuration..."
    sed -i "/$MARKER/,/# End NAVSIM Setup/d" "$BASHRC"
fi

# Add new configuration
cat >> "$BASHRC" << 'EOF'

# NAVSIM Torch Setup
if [[ $(hostname) == *"cs649"* ]]; then
    # Load anaconda module
    module load anaconda3/2025.06 2>/dev/null || true
    
    # NAVSIM environment variables
    export NAVSIM_DEVKIT_ROOT="/scratch/ah7072/navsim_workspace/navsim"
    export OPENSCENE_DATA_ROOT="/scratch/ah7072/navsim_workspace/dataset"
    export NUPLAN_MAPS_ROOT="/scratch/ah7072/navsim_workspace/dataset/maps"
    export NAVSIM_EXP_ROOT="/scratch/ah7072/navsim_workspace/exp"
    export APPTAINER_BINDPATH=/scratch,/state/partition1,/mnt,/share/apps
    
    # Convenience alias
    alias navsim='conda activate navsim'
fi
# End NAVSIM Setup
EOF

echo "✓ Environment variables added to ~/.bashrc"

# Final summary
echo ""
echo "=============================================="
echo "Setup Complete! 🎉"
echo "=============================================="
echo ""
echo "Your environment is ready. Next steps:"
echo ""
echo "1. Reload your shell or run:"
echo "   source ~/.bashrc"
echo ""
echo "2. Activate environment:"
echo "   conda activate navsim"
echo "   # or just: navsim"
echo ""
echo "3. Check if OpenScene data is available:"
echo "   ls /scratch/ah7072/navsim_workspace/dataset/openscene/"
echo ""
echo "   If missing, download it:"
echo "   cd /scratch/ah7072/data"
echo "   ./download_quick_start.sh"
echo ""
echo "4. Run smoke test:"
echo "   cd /scratch/ah7072/GTRS/scripts"
echo "   sbatch torch_smoke_test.slurm"
echo ""
echo "5. Monitor the job:"
echo "   squeue -u ah7072"
echo "   tail -f /scratch/ah7072/navsim_workspace/exp/logs/smoke_*.out"
echo ""
echo "6. If smoke test passes, launch training:"
echo "   ./launch_all_torch.sh"
echo ""
echo "For help, see: /scratch/ah7072/GTRS/scripts/TORCH_SETUP_GUIDE.md"
echo ""
