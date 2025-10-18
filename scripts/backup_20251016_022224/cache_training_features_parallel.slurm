#!/bin/bash
#SBATCH --job-name=cache_training
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=256GB
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/ah7072/navsim_workspace/exp/logs/cache_%j.out
#SBATCH --error=/scratch/ah7072/navsim_workspace/exp/logs/cache_%j.err

# =============================================================================
# Precompute Training Features for Hydra-MDP
# This creates a cache so training can load preprocessed features directly
# =============================================================================

echo "=============================================="
echo "Feature Caching for Hydra-MDP Training"
echo "CPUs: 128"
echo "Node: $SLURMD_NODENAME"
echo "=============================================="

# Environment setup
export NAVSIM_DEVKIT_ROOT="/scratch/ah7072/GTRS"
export OPENSCENE_DATA_ROOT="/scratch/ah7072/data/openscene"
export NUPLAN_MAPS_ROOT="/scratch/ah7072/data/maps"
export NAVSIM_EXP_ROOT="/scratch/ah7072/experiments"
export DP_PREDS="none"

# Cache output directory
CACHE_DIR="${NAVSIM_EXP_ROOT}/training_cache"
mkdir -p "${CACHE_DIR}"

cd "${NAVSIM_DEVKIT_ROOT}"

# Load conda environment
module load anaconda3/2025.06
eval "$(conda shell.bash hook)"
conda activate navsim

echo "Using Python: $(which python)"
echo "Python version: $(python --version)"
echo ""
echo "Cache directory: ${CACHE_DIR}"
echo "Starting feature caching at $(date)"
echo ""

# Run dataset caching with Hydra-MDP agent
# This will parallelize using Ray (default worker, threads_per_node=null means use all CPUs)
python navsim/planning/script/run_dataset_caching.py \
    agent=hydra_mdp_v8192_w_ep \
    experiment_name="cache_training_$(date +%Y%m%d_%H%M%S)" \
    split=trainval \
    train_test_split=navtrain \
    cache_path="${CACHE_DIR}" \
    force_cache_computation=true

echo ""
echo "=============================================="
echo "Feature caching completed at $(date)"
echo "Cache saved to: ${CACHE_DIR}"
echo ""
echo "To use the cache, run training with:"
echo "  cache_path=${CACHE_DIR}"
echo "=============================================="
