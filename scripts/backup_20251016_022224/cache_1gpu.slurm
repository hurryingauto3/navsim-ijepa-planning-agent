#!/bin/bash
#SBATCH --job-name=hydra_1gpu
#SBATCH --partition=h200_tandon
#SBATCH --account=torch_pr_68_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=196GB
#SBATCH --gres=gpu:h200:1
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/ah7072/navsim_workspace/exp/logs/train_%j.out
#SBATCH --error=/scratch/ah7072/navsim_workspace/exp/logs/train_%j.err
#SBATCH --mail-user=ah7072@nyu.edu
#SBATCH --mail-type=END,FAIL

# =============================================================================
# Single H200 GPU Training - Baseline to Get Training Working
# =============================================================================

echo "=============================================="
echo "Hydra-MDP Training (1x H200 GPU)"
echo "GPUs: 1x H200"
echo "Batch size: 22"
echo "Node: $SLURMD_NODENAME"
echo "=============================================="

# Environment setup
export NAVSIM_DEVKIT_ROOT="/scratch/ah7072/GTRS"
export OPENSCENE_DATA_ROOT="/scratch/ah7072/data/openscene"
export NUPLAN_MAPS_ROOT="/scratch/ah7072/data/maps"
export NAVSIM_EXP_ROOT="/scratch/ah7072/experiments"
export DP_PREDS="none"

# Create experiment directory
EXPERIMENT_NAME="hydra_w_ep_1gpu_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${NAVSIM_EXP_ROOT}/${EXPERIMENT_NAME}/notes"

cd "${NAVSIM_DEVKIT_ROOT}"

# Load conda environment
module load anaconda3/2025.06
eval "$(conda shell.bash hook)"
conda activate navsim

echo "Using Python: $(which python)"
echo "Python version: $(python --version)"
echo ""
echo "Starting training at $(date)"
echo ""

# Create training feature cache with GPU + Ray parallelization
# Cache directory
CACHE_DIR="${NAVSIM_EXP_ROOT}/training_cache"
mkdir -p "${CACHE_DIR}"

python navsim/planning/script/run_dataset_caching.py \
    agent=hydra_mdp_v8192_w_ep \
    experiment_name="${EXPERIMENT_NAME}" \
    split=trainval \
    train_test_split=navtrain \
    cache_path="${CACHE_DIR}" \
    force_cache_computation=true \
    worker=single_machine_thread_pool \
    +worker.max_workers=16

echo ""
echo "=============================================="
echo "Training completed at $(date)"
echo "Checkpoints saved to: ${NAVSIM_EXP_ROOT}/${EXPERIMENT_NAME}"
echo "=============================================="
