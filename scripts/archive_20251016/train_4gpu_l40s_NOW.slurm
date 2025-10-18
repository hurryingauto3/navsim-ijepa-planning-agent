#!/bin/bash
#SBATCH --job-name=hydra_4gpu_l40s
#SBATCH --partition=l40s_public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --mem=256GB
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/ah7072/navsim_workspace/exp/logs/train_%j.out
#SBATCH --error=/scratch/ah7072/navsim_workspace/exp/logs/train_%j.err

# =============================================================================
# NAVSIM Training on L40S (4 GPUs) - STARTS IMMEDIATELY!
# L40S: 4x GPUs, 48GB VRAM each = plenty for 83M param model
# =============================================================================

echo "=============================================="
echo "Hydra-MDP DDP Training (4x L40S GPUs)"
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

# Make 4 GPUs visible (Lightning auto-detects)
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Create experiment directory
EXPERIMENT_NAME="hydra_w_ep_4gpu_l40s_$(date +%Y%m%d_%H%M%S)"
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
echo "  GPU type: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
echo ""

# Training with 4 L40S GPUs
# Batch size: 22 per GPU × 4 = 88 total (vs 176 with 8 GPUs)
# Expected: ~18 hours for 40 epochs (vs 10 hours with 8× H200)

python navsim/planning/script/run_training_dense.py \
    agent=hydra_mdp_v8192_w_ep \
    experiment_name="${EXPERIMENT_NAME}" \
    train_test_split=navtrain \
    cache_path=null \
    ~trainer.params.strategy \
    trainer.params.max_epochs=40 \
    trainer.params.accelerator=gpu \
    trainer.params.num_nodes=1 \
    trainer.params.precision=16-mixed \
    dataloader.params.batch_size=22 \
    dataloader.params.num_workers=8 \
    dataloader.params.pin_memory=true

echo ""
echo "=============================================="
echo "Training complete at $(date)"
echo "=============================================="
