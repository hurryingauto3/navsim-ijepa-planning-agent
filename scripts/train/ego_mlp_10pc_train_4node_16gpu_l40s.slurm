#!/bin/bash
#SBATCH --job-name=ego_mlp_10pct
#SBATCH --partition=l40s_public
#SBATCH --account=torch_pr_68_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=64GB
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/ah7072/experiments/logs/output/train_ego_mlp_10_%j.out
#SBATCH --error=/scratch/ah7072/experiments/logs/error/train_ego_mlp_10_%j.err
#SBATCH --requeue

# =============================================================================
# Ego-Status MLP @ 10% Training: 1 Node × 2 L40S GPUs = 2 GPUs Total
# Purpose: Data efficiency baseline (no vision)
# Expected: ~1 hour, PDMS ~60-62
# =============================================================================

echo "=============================================="
echo "EXP-A3: Ego-Status MLP @ 10% navtrain"
echo "Configuration: 1 node × 2 L40S GPUs = 2 GPUs"
echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "=============================================="

# Environment setup
export NAVSIM_DEVKIT_ROOT="/scratch/ah7072/navsim"
export OPENSCENE_DATA_ROOT="/scratch/ah7072/data"
export NUPLAN_MAPS_ROOT="/scratch/ah7072/data/maps"
export NAVSIM_EXP_ROOT="/scratch/ah7072/experiments"
export DP_PREDS="none"

# Single-node DDP (simpler than multi-node)
export MASTER_PORT=12358
export MASTER_ADDR=localhost

echo "Single-node DDP setup:"
echo "  Master node: $MASTER_ADDR"
echo "  Master port: $MASTER_PORT"
echo "  Total GPUs: 2"
echo ""

# Create experiment directory
EXPERIMENT_NAME="exp_a3_ego_mlp_10pct_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${NAVSIM_EXP_ROOT}/${EXPERIMENT_NAME}/notes"

cd "${NAVSIM_DEVKIT_ROOT}"

# Load conda environment
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

# Set PYTHONPATH to include navsim module
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1

echo "Environment check:"
echo "  Python: $(which python)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  PYTHONPATH: ${PYTHONPATH}"
echo ""

echo "Experiment configuration:"
echo "  Agent: ego_status_mlp_agent"
echo "  Architecture: IDENTICAL to 100% experiment"
echo "  Trainable params: ~469K"
echo "  Data: 10% navtrain (~10.3K scenarios) ← KEY DIFFERENCE"
echo "  Batch size: 32 per GPU (64 effective)"
echo "  Epochs: 50 (same duration for fair comparison)"
echo "  Expected time: ~1 hour"
echo ""

# Launch training with srun
srun python navsim/planning/script/run_training.py \
    agent=ego_status_mlp_agent \
    experiment_name="${EXPERIMENT_NAME}" \
    train_test_split=navtrain \
    cache_path=null \
    trainer.params.max_epochs=50 \
    trainer.params.accelerator=gpu \
    trainer.params.num_nodes=1 \
    trainer.params.limit_train_batches=0.1 \
    dataloader.params.batch_size=32 \
    dataloader.params.num_workers=2 \
    dataloader.params.pin_memory=true \


echo ""
echo "=============================================="
echo "Training complete at $(date)"
echo "Results saved to: ${NAVSIM_EXP_ROOT}/${EXPERIMENT_NAME}"
echo "=============================================="