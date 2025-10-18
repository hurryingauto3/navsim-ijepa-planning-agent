#!/bin/bash
#SBATCH --job-name=hydra_8gpu_ddp
#SBATCH --partition=h200_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:8
#SBATCH --mem=512GB
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/ah7072/navsim_workspace/exp/logs/train_%j.out
#SBATCH --error=/scratch/ah7072/navsim_workspace/exp/logs/train_%j.err

# =============================================================================
# CORRECT Multi-GPU DDP Training (8 GPUs)
# Key: Lightning DDP needs MULTIPLE GPUS visible, not multiple tasks!
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

# CRITICAL: These tell Lightning that 8 GPUs are available
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Create experiment directory
EXPERIMENT_NAME="hydra_w_ep_8gpu_ddp_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${NAVSIM_EXP_ROOT}/${EXPERIMENT_NAME}/notes"

cd "${NAVSIM_DEVKIT_ROOT}"

# Load conda environment
module purge
module load anaconda3/2025.06
source activate navsim

echo "Environment check:"
echo "  Python: $(which python)"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "  GPUs detected: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo ""

# CRITICAL SETTINGS:
# - devices=8: tells Lightning to use 8 GPUs
# - strategy=ddp: Lightning will spawn 8 processes internally
# - num_workers=8: data loading per GPU (8 GPUs × 8 workers = 64 total)
# - batch_size=22: per GPU (22 × 8 = 176 effective batch size)

python navsim/planning/script/run_training_dense.py \
    agent=hydra_mdp_v8192_w_ep \
    experiment_name="${EXPERIMENT_NAME}" \
    train_test_split=navtrain \
    cache_path=null \
    trainer.params.devices=8 \
    trainer.params.strategy=ddp \
    trainer.params.num_nodes=1 \
    trainer.params.accelerator=gpu \
    trainer.params.max_epochs=40 \
    trainer.params.precision=16-mixed \
    dataloader.params.batch_size=22 \
    dataloader.params.num_workers=8 \
    dataloader.params.pin_memory=true

echo ""
echo "=============================================="
echo "Training complete at $(date)"
echo "=============================================="
