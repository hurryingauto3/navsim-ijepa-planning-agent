#!/bin/bash
#SBATCH --job-name=hydra_smoke
#SBATCH --cpus-per-task=8
#SBATCH --mem=96GB
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/ah7072/navsim_workspace/exp/logs/smoke_%j.out
#SBATCH --error=/scratch/ah7072/navsim_workspace/exp/logs/smoke_%j.err
#SBATCH --mail-user=ah7072@nyu.edu
#SBATCH --mail-type=ALL

# =============================================================================
# Smoke test for Hydra-MDP configs (1 epoch, limited batches)
# Run this first to verify everything works before full training
# =============================================================================

# Environment setup
export NAVSIM_DEVKIT_ROOT="/scratch/ah7072/navsim_workspace/navsim"
export OPENSCENE_DATA_ROOT="/scratch/ah7072/navsim_workspace/dataset"
export NUPLAN_MAPS_ROOT="/scratch/ah7072/navsim_workspace/dataset/maps"
export NAVSIM_EXP_ROOT="/scratch/ah7072/navsim_workspace/exp"
export DP_PREDS="none"  # Set to none for training without DP predictions

CONFIG="${1:-hydra_mdp_v8192_w_ep}"  # Default to best variant

echo "=============================================="
echo "Smoke Test: $CONFIG"
echo "=============================================="
echo "This will run 1 epoch with limited batches"
echo "to verify configuration and data loading."
echo "=============================================="

mkdir -p /scratch/$USER/experiments/logs

cd "${NAVSIM_DEVKIT_ROOT}"

# Load conda environment
module load anaconda3/2025.06
eval "$(conda shell.bash hook)"
conda activate navsim

# Verify environment
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate conda environment 'navsim'"
    exit 1
fi

echo "Using Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

python navsim/planning/script/run_training_dense.py \
    agent=${CONFIG} \
    experiment_name="smoke_test_${CONFIG}" \
    cache_path=null \
    train_test_split=navtrain \
    ~trainer.params.strategy \
    trainer.params.max_epochs=1 \
    trainer.params.limit_train_batches=20 \
    trainer.params.limit_val_batches=10 \
    trainer.params.accelerator=gpu \
    dataloader.params.batch_size=2 \
    dataloader.params.num_workers=4

echo ""
echo "=============================================="
echo "Smoke test complete!"
echo "If no errors, you're ready for full training."
echo "=============================================="