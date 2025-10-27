#!/bin/bash
#SBATCH --job-name=hydra_8gpu_ddp
#SBATCH --partition=h200_tandon
#SBATCH --account=torch_pr_68_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:8
#SBATCH --mem=512GB
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/ah7072/navsim_workspace/exp/logs/train_%j.out
#SBATCH --error=/scratch/ah7072/navsim_workspace/exp/logs/train_%j.err

# =============================================================================
# CORRECT Multi-GPU DDP Training (8 GPUs) - FIXED VERSION
# Key Insight: PyTorch Lightning auto-detects GPUs from CUDA_VISIBLE_DEVICES!
# =============================================================================

echo "=============================================="
echo "Hydra-MDP DDP Training (8x H200 GPUs)"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "=============================================="

# Environment setup
export NAVSIM_DEVKIT_ROOT="/scratch/ah7072/GTRS"
export OPENSCENE_DATA_ROOT="/scratch/ah7072/data/openscene"
export NUPLAN_MAPS_ROOT="/scratch/ah7072/data/maps"
export NAVSIM_EXP_ROOT="/scratch/ah7072/experiments"
export DP_PREDS="none"

# CRITICAL: Make all 8 GPUs visible
# PyTorch Lightning will auto-detect and use ALL visible GPUs with DDP
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Create experiment directory
EXPERIMENT_NAME="hydra_w_ep_8gpu_ddp_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${NAVSIM_EXP_ROOT}/${EXPERIMENT_NAME}/notes"

cd "${NAVSIM_DEVKIT_ROOT}"

# Load conda environment: prefer Miniconda in /scratch/$USER if available
CONDA_ROOT="/scratch/$USER/miniconda3"
if [ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]; then
    source "${CONDA_ROOT}/etc/profile.d/conda.sh"
    conda activate navsim
else
    module purge || true
    module load anaconda3/2025.06 || true
    if command -v conda &> /dev/null; then
        source $(conda info --base)/etc/profile.d/conda.sh || true
        conda activate navsim || source activate navsim || true
    else
        echo "WARNING: conda not found; training may fail"
    fi
fi

echo "Environment check:"
echo "  Python: $(which python)"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "  GPUs detected: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo ""

# KEY INSIGHTS:
# 1. run_training_dense.py HARDCODES strategy=DDPStrategy (line 159)
# 2. Config only has: accelerator, strategy, num_nodes (NO devices parameter!)
# 3. Lightning auto-detects GPUs from CUDA_VISIBLE_DEVICES
# 4. With strategy=ddp + 8 visible GPUs → Lightning spawns 8 processes automatically

# We can only override what's IN the config:
# - accelerator=gpu (already default)
# - num_nodes=1 (already default)
# - batch_size, num_workers, etc.

python navsim/planning/script/run_training_dense.py \
    agent=hydra_mdp_v8192_w_ep \
    experiment_name="${EXPERIMENT_NAME}" \
    train_test_split=navtrain \
    cache_path=null \
    ~trainer.params.strategy \
    trainer.params.max_epochs=1 \
    trainer.params.accelerator=gpu \
    trainer.params.num_nodes=1 \
    trainer.params.precision=16-mixed \
    dataloader.params.batch_size=22 \
    dataloader.params.num_workers=2 \
    dataloader.params.pin_memory=true

echo ""
echo "=============================================="
echo "Training complete at $(date)"
echo "=============================================="
