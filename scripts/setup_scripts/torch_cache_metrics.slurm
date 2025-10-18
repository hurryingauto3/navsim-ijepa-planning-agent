#!/bin/bash
#SBATCH --job-name=cache_metrics
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80GB
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/ah7072/navsim_workspace/exp/logs/cache_metrics_%j.out
#SBATCH --error=/scratch/ah7072/navsim_workspace/exp/logs/cache_metrics_%j.err

# =============================================================================
# Metric Caching Script for NAVSIM (Torch HPC)
# 
# This pre-computes expensive operations like map access and coordinate
# transformations for faster evaluation. Run this ONCE before evaluation.
# 
# Caches are split-specific, so you need to run for navtest, navmini, etc.
# =============================================================================

# Environment setup
export NAVSIM_DEVKIT_ROOT="/scratch/ah7072/navsim_workspace/navsim"
export OPENSCENE_DATA_ROOT="/scratch/ah7072/navsim_workspace/dataset"
export NUPLAN_MAPS_ROOT="/scratch/ah7072/navsim_workspace/dataset/maps"
export NAVSIM_EXP_ROOT="/scratch/ah7072/navsim_workspace/exp"
export APPTAINER_BINDPATH=/scratch,/state/partition1,/mnt,/share/apps

# Parse split argument (default: navtest)
SPLIT="${1:-navtest}"

echo "=============================================="
echo "Metric Caching for Split: $SPLIT"
echo "=============================================="
echo "Cache will be saved to: ${NAVSIM_EXP_ROOT}/metric_cache"
echo ""

# Create cache directory
CACHE_PATH="${NAVSIM_EXP_ROOT}/metric_cache"
mkdir -p "${CACHE_PATH}"

# Navigate to repo
cd "${NAVSIM_DEVKIT_ROOT}"

# Try to activate conda environment
# Option 1: If module system is available
if command -v module &> /dev/null; then
    module load anaconda3/2025.06 2>/dev/null || true
fi

# Option 2: Activate conda from standard locations
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate navsim 2>/dev/null || conda activate conda_gtrs 2>/dev/null || {
        echo "ERROR: No conda environment found!"
        echo "Please create it first:"
        echo "  cd /scratch/ah7072/navsim_workspace/navsim"
        echo "  conda env create -n navsim -f environment.yml"
        exit 1
    }
else
    echo "ERROR: conda not found!"
    echo "Please load conda module or set up conda environment"
    exit 1
fi

# Verify Python is available
python --version || { echo "ERROR: Python not available after conda activation"; exit 1; }

# Run metric caching
python navsim/planning/script/run_metric_caching.py \
    train_test_split=${SPLIT} \
    metric_cache_path=${CACHE_PATH}

echo ""
echo "=============================================="
echo "Metric caching complete for split: ${SPLIT}"
echo "Cache location: ${CACHE_PATH}/${SPLIT}"
echo "=============================================="
