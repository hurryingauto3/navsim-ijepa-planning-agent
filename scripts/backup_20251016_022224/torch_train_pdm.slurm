#!/bin/bash
#SBATCH --job-name=hydra_pdm
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --gres=gpu:h200:1
#SBATCH --time=48:00:00
#SBATCH --comment="preemption=yes;requeue=yes"
#SBATCH --output=/scratch/ah7072/navsim_workspace/exp/logs/hydra_pdm_%j.out
#SBATCH --error=/scratch/ah7072/navsim_workspace/exp/logs/hydra_pdm_%j.err

# =============================================================================
# Hydra-MDP PDM-only Training (Table 1: Score 80.2)
# This recreates the baseline that learns from overall PDM score only
# Expected: NC=97.5, DAC=88.9, EP=74.8, TTC=92.5, C=100, Score=80.2
# =============================================================================

# Environment setup
export NAVSIM_DEVKIT_ROOT="/scratch/ah7072/navsim_workspace/navsim"
export OPENSCENE_DATA_ROOT="/scratch/ah7072/navsim_workspace/dataset"
export NUPLAN_MAPS_ROOT="/scratch/ah7072/navsim_workspace/dataset/maps"
export NAVSIM_EXP_ROOT="/scratch/ah7072/navsim_workspace/exp"
export DP_PREDS="none"

# Experiment metadata
EXPERIMENT_NAME="hydra_pdm_v8192_$(date +%Y%m%d_%H%M%S)"

echo "=============================================="
echo "Hydra-MDP PDM-only Training"
echo "Target: Score 80.2 (Table 1)"
echo "Started: $(date)"
echo "=============================================="

cd "${NAVSIM_DEVKIT_ROOT}"

# Load conda environment
module load anaconda3/2025.06
eval "$(conda shell.bash hook)"
conda activate navsim

echo "Using Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Training command - matches GTRS methodology exactly
python navsim/planning/script/run_training_dense.py \
    agent=hydra_mdp_v8192_pdm \
    experiment_name="${EXPERIMENT_NAME}" \
    cache_path=null \
    train_test_split=navtrain \
    ~trainer.params.strategy \
    trainer.params.max_epochs=20 \
    trainer.params.precision=32 \
    trainer.params.accelerator=gpu \
    +trainer.log_every_n_steps=2 \
    dataloader.params.batch_size=8 \
    dataloader.params.num_workers=8
    

echo ""
echo "=============================================="
echo "Training completed at $(date)"
echo "Logs: ${NAVSIM_EXP_ROOT}/${EXPERIMENT_NAME}"
echo "=============================================="
