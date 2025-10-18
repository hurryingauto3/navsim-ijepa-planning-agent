#!/bin/bash
#SBATCH --job-name=hydra_8gpu
#SBATCH --partition=h200_tandon
#SBATCH --account=torch_pr_68_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --mem=256GB
#SBATCH --gres=gpu:h200:1
#SBATCH --time=8:00:00
#SBATCH --output=/scratch/ah7072/navsim_workspace/exp/logs/train_%j.out
#SBATCH --error=/scratch/ah7072/navsim_workspace/exp/logs/train_%j.err
#SBATCH --mail-user=ah7072@nyu.edu
#SBATCH --mail-type=END,FAIL

# =============================================================================
# Fast Single-Node Training with 8x H200 GPUs
# More likely to start quickly than 16 GPU multi-node job
# =============================================================================

echo "=============================================="
echo "Hydra-MDP Training (8x H200 GPUs - Single Node)"
echo "GPUs: 8x H200"
echo "Total batch size: 176 (22 per GPU)"
echo "Node: $SLURMD_NODENAME"
echo "=============================================="

# Environment setup
export NAVSIM_DEVKIT_ROOT="/scratch/ah7072/GTRS"
export OPENSCENE_DATA_ROOT="/scratch/ah7072/data/openscene"
export NUPLAN_MAPS_ROOT="/scratch/ah7072/data/maps"
export NAVSIM_EXP_ROOT="/scratch/ah7072/experiments"
export DP_PREDS="none"

# Create experiment directory
EXPERIMENT_NAME="hydra_w_ep_8gpu_$(date +%Y%m%d_%H%M%S)"
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

# Training with 1 H200 GPU (simplified to avoid DDP issues)
# Expected: ~2-3 batches/sec → ~3 min/epoch → ~2 hours for 40 epochs
python navsim/planning/script/run_training_dense.py \
    agent=hydra_mdp_v8192_w_ep \
    experiment_name="${EXPERIMENT_NAME}" \
    train_test_split=navtrain \
    cache_path=null \
    ~trainer.params.strategy \
    trainer.params.max_epochs=40 \
    trainer.params.check_val_every_n_epoch=1 \
    trainer.params.val_check_interval=1.0 \
    trainer.params.limit_train_batches=1.0 \
    trainer.params.limit_val_batches=1.0 \
    dataloader.params.batch_size=22 \
    dataloader.params.num_workers=8 \
    dataloader.params.pin_memory=true \
    dataloader.params.prefetch_factor=2

echo ""
echo "=============================================="
echo "Training completed at $(date)"
echo "Checkpoints saved to: ${NAVSIM_EXP_ROOT}/${EXPERIMENT_NAME}"
echo "=============================================="
