#!/bin/bash
#SBATCH --job-name=cache_training_gpu
#SBATCH --partition=h200_tandon
#SBATCH --account=torch_pr_68_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256GB
#SBATCH --gres=gpu:h200:1
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/ah7072/navsim_workspace/exp/logs/cache_train_gpu_%j.out
#SBATCH --error=/scratch/ah7072/navsim_workspace/exp/logs/cache_train_gpu_%j.err
#SBATCH --mail-user=ah7072@nyu.edu
#SBATCH --mail-type=END,FAIL

# =============================================================================
# Training Feature Cache Creation (GPU-based)
# The agent needs GPU to load VoV backbone, so we use 1 GPU + 64 CPUs
# =============================================================================

echo "=============================================="
echo "Creating Training Feature Cache (GPU + 64 CPUs)"
echo "GPUs: 1x H200"
echo "CPUs: 64"
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

# Run dataset caching with Hydra-MDP agent on GPU
# Ray will parallelize feature extraction across 64 CPUs
# The agent (VoV backbone) runs on GPU
python navsim/planning/script/run_dataset_caching.py \
    agent=hydra_mdp_v8192_w_ep \
    experiment_name="cache_training_gpu_$(date +%Y%m%d_%H%M%S)" \
    split=trainval \
    train_test_split=navtrain \
    cache_path="${CACHE_DIR}" \
    force_cache_computation=true

echo ""
echo "=============================================="
echo "Feature caching completed at $(date)"
echo "Cache saved to: ${CACHE_DIR}"
echo "=============================================="
