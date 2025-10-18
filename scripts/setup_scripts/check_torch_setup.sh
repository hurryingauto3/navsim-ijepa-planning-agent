#!/bin/bash
# =============================================================================
# Quick Start Script for NAVSIM on Torch HPC
# Run this first to check your environment and guide you through setup
# =============================================================================

set -e

echo "=============================================="
echo "NAVSIM Torch HPC Setup Checker"
echo "=============================================="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check 1: Are we on Torch?
echo "Step 1: Verifying we're on Torch..."
if [[ $(hostname) == *"cs649"* ]]; then
    echo -e "${GREEN}✓${NC} Running on Torch (cs649)"
else
    echo -e "${YELLOW}⚠${NC} Not on Torch. Current host: $(hostname)"
    echo "   Connect with: ssh greene.hpc.nyu.edu, then ssh cs649"
fi
echo ""

# Check 2: Module system
echo "Step 2: Checking for module system..."
if command -v module &> /dev/null; then
    echo -e "${GREEN}✓${NC} Module system available"
    echo ""
    echo "   Available Python/Conda modules:"
    module avail 2>&1 | grep -iE "python|conda|anaconda|miniconda" | head -5 || echo "   None found"
else
    echo -e "${YELLOW}⚠${NC} No module system found"
fi
echo ""

# Check 3: Conda availability
echo "Step 3: Checking for conda..."
if command -v conda &> /dev/null; then
    echo -e "${GREEN}✓${NC} Conda found at: $(which conda)"
    conda --version
    echo ""
    echo "   Available environments:"
    conda env list | grep -E "navsim|conda_gtrs" || echo "   No NAVSIM environment found"
else
    echo -e "${YELLOW}⚠${NC} Conda not found in PATH"
    echo ""
    echo "   Checking Greene home accessibility..."
    if [ -d "/gpfsnyu/home/$USER" ]; then
        echo -e "   ${GREEN}✓${NC} Greene home accessible"
        if [ -d "/gpfsnyu/home/$USER/.conda" ]; then
            echo -e "   ${GREEN}✓${NC} Greene conda found - you can use this!"
        fi
    else
        echo -e "   ${RED}✗${NC} Greene home not accessible"
    fi
fi
echo ""

# Check 4: NAVSIM repository
echo "Step 4: Checking NAVSIM installation..."
NAVSIM_DIR="/scratch/$USER/navsim_workspace/navsim"
if [ -d "$NAVSIM_DIR" ]; then
    echo -e "${GREEN}✓${NC} NAVSIM found at: $NAVSIM_DIR"
    if [ -f "$NAVSIM_DIR/environment.yml" ]; then
        echo -e "${GREEN}✓${NC} environment.yml exists"
    fi
else
    echo -e "${RED}✗${NC} NAVSIM not found at: $NAVSIM_DIR"
fi
echo ""

# Check 5: Data availability
echo "Step 5: Checking data..."
if [ -d "/scratch/$USER/navsim_workspace/dataset" ]; then
    echo -e "${GREEN}✓${NC} Dataset directory exists"
    
    # Check specific components
    if [ -d "/scratch/$USER/navsim_workspace/dataset/maps" ]; then
        echo -e "${GREEN}  ✓${NC} Maps downloaded"
    else
        echo -e "${RED}  ✗${NC} Maps missing"
    fi
    
    if [ -d "/scratch/$USER/navsim_workspace/dataset/openscene" ]; then
        echo -e "${GREEN}  ✓${NC} OpenScene data exists"
    else
        echo -e "${RED}  ✗${NC} OpenScene data missing"
    fi
else
    echo -e "${RED}✗${NC} Dataset directory not found"
fi
echo ""

# Check 6: Prerequisites
echo "Step 6: Checking training prerequisites..."
if [ -f "/scratch/$USER/navsim_workspace/dataset/models/dd3d_det_final.pth" ]; then
    echo -e "${GREEN}✓${NC} DD3D backbone downloaded"
else
    echo -e "${YELLOW}⚠${NC} DD3D backbone missing"
fi

if [ -f "/scratch/$USER/navsim_workspace/dataset/traj_pdm_v2/ori/navtrain_16384.pkl" ]; then
    echo -e "${GREEN}✓${NC} PDM ground truths downloaded"
else
    echo -e "${YELLOW}⚠${NC} PDM ground truths missing"
fi
echo ""

# Summary and recommendations
echo "=============================================="
echo "Summary and Next Steps"
echo "=============================================="
echo ""

# Determine what to do
NEEDS_CONDA=false
NEEDS_DATA=false
NEEDS_PREREQS=false

if ! command -v conda &> /dev/null; then
    NEEDS_CONDA=true
fi

if [ ! -d "/scratch/$USER/navsim_workspace/dataset/maps" ]; then
    NEEDS_DATA=true
fi

if [ ! -f "/scratch/$USER/navsim_workspace/dataset/models/dd3d_det_final.pth" ]; then
    NEEDS_PREREQS=true
fi

if $NEEDS_CONDA; then
    echo -e "${YELLOW}ACTION REQUIRED:${NC} Set up conda environment"
    echo ""
    echo "Choose one of these options:"
    echo ""
    echo "Option 1: Use module system (if available)"
    echo "  module load anaconda3"
    echo "  cd $NAVSIM_DIR"
    echo "  conda env create -n navsim -f environment.yml"
    echo ""
    echo "Option 2: Transfer from Greene (if you have it there)"
    echo "  # On Greene:"
    echo "  conda pack -n navsim -o /scratch/$USER/navsim_env.tar.gz"
    echo "  # On Torch:"
    echo "  mkdir -p ~/.conda/envs/navsim"
    echo "  cd ~/.conda/envs/navsim"
    echo "  tar -xzf /scratch/$USER/navsim_env.tar.gz"
    echo "  source bin/activate"
    echo "  conda-unpack"
    echo ""
    echo "See: /scratch/ah7072/GTRS/scripts/TORCH_SETUP_GUIDE.md"
    echo ""
fi

if $NEEDS_DATA; then
    echo -e "${YELLOW}ACTION REQUIRED:${NC} Download NAVSIM data"
    echo ""
    echo "Follow the data download guide:"
    echo "  cd /scratch/ah7072/data"
    echo "  ./download_quick_start.sh"
    echo ""
fi

if $NEEDS_PREREQS; then
    echo -e "${YELLOW}ACTION REQUIRED:${NC} Download training prerequisites"
    echo ""
    echo "Run the setup script:"
    echo "  cd /scratch/ah7072/GTRS/scripts"
    echo "  ./setup_all.sh"
    echo ""
fi

if ! $NEEDS_CONDA && ! $NEEDS_DATA && ! $NEEDS_PREREQS; then
    echo -e "${GREEN}✓ All prerequisites met!${NC}"
    echo ""
    echo "You're ready to run experiments:"
    echo ""
    echo "1. Test with smoke test (30 min):"
    echo "   cd /scratch/ah7072/GTRS/scripts"
    echo "   sbatch torch_smoke_test.slurm"
    echo ""
    echo "2. Launch full training (48h each):"
    echo "   ./launch_all_torch.sh"
    echo ""
    echo "3. Monitor jobs:"
    echo "   squeue -u $USER"
    echo "   tail -f /scratch/$USER/navsim_workspace/exp/logs/*.out"
    echo ""
fi

echo "For detailed setup instructions, see:"
echo "  /scratch/ah7072/GTRS/scripts/TORCH_SETUP_GUIDE.md"
echo ""
