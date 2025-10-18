#!/bin/bash
#SBATCH --job-name=hydra_smoke
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="h200|l40s"
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/ah7072/navsim_workspace/exp/logs/smoke_%j.out
#SBATCH --error=/scratch/ah7072/navsim_workspace/exp/logs/smoke_%j.err

# =============================================================================
# Smoke test for Hydra-MDP configs (1 epoch, limited batches)
# Run this first to verify everything works before full training
# =============================================================================

# Environment setup
export NAVSIM_DEVKIT_ROOT="/scratch/ah7072/navsim_workspace/navsim"
export OPENSCENE_DATA_ROOT="/scratch/ah7072/navsim_workspace/dataset"
export NUPLAN_MAPS_ROOT="/scratch/ah7072/navsim_workspace/dataset/maps"
export NAVSIM_EXP_ROOT="/scratch/ah7072/navsim_workspace/exp"
export APPTAINER_BINDPATH=/scratch,/state/partition1,/mnt,/share/apps

CONFIG="${1:-hydra_mdp_v8192_w_ep}"  # Default to best variant

echo "=============================================="
echo "Smoke Test: $CONFIG"
echo "=============================================="
echo "This will run 1 epoch with limited batches"
echo "to verify configuration and data loading."
echo "=============================================="

mkdir -p /scratch/$USER/experiments/logs

cd "${NAVSIM_DEVKIT_ROOT}"

# Try to activate conda environment
if command -v module &> /dev/null; then
    module load anaconda3/2025.06 2>/dev/null || true
fi

if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate navsim 2>/dev/null || conda activate conda_gtrs 2>/dev/null || {
        echo "ERROR: No conda environment found! Create it with:"
        echo "  conda env create -n navsim -f environment.yml"
        exit 1
    }
else
    echo "ERROR: conda not found! Load module or set up conda."
    exit 1
fi

python navsim/planning/script/run_training_dense.py \
    agent=${CONFIG} \
    experiment_name="smoke_test_${CONFIG}" \
    experiment_root="${NAVSIM_EXP_ROOT}" \
    trainer.max_epochs=1 \
    trainer.limit_train_batches=20 \
    trainer.limit_val_batches=10 \
    trainer.devices=1 \
    trainer.accelerator=gpu \
    data.batch_size=4 \
    data.num_workers=2 \
    train_test_split=navmini \
    metric_cache_path="${NAVSIM_EXP_ROOT}/metric_cache" \

echo ""
echo "=============================================="
echo "Smoke test complete!"
echo "If no errors, you're ready for full training."
echo "=============================================="
