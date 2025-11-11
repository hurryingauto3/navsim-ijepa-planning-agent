#!/bin/bash
#SBATCH --job-name=ego_mlp_100pct
#SBATCH --partition=l40s_public
#SBATCH --account=torch_pr_68_tandon_advanced
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:4
#SBATCH --mem=200GB
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/ah7072/experiments/logs/output/train_ego_mlp_100_%j.out
#SBATCH --error=/scratch/ah7072/experiments/logs/error/train_ego_mlp_100_%j.err
#SBATCH --requeue

# =============================================================================
# Ego-Status MLP @ 100% Training: 2 Nodes × 4 L40S GPUs = 8 GPUs Total
# Purpose: Baseline learning without vision (ego velocity/acceleration only)
# Expected: ~8 hours, PDMS ~65.6
# =============================================================================

echo "=============================================="
echo "EXP-A2: Ego-Status MLP @ 100% navtrain"
echo "Configuration: 2 nodes × 4 L40S GPUs = 8 GPUs"
echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "=============================================="

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=NET
export NCCL_ASYNC_ERROR_HANDLING=1      # helps avoid hard crashes, better error propagation
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export OMP_NUM_THREADS=1                # avoid CPU oversubscription
export NCCL_P2P_DISABLE=0  

# Environment setup
export NAVSIM_DEVKIT_ROOT="/scratch/ah7072/navsim"
export OPENSCENE_DATA_ROOT="/scratch/ah7072/data/openscene"
export NUPLAN_MAPS_ROOT="/scratch/ah7072/data/maps"
export NAVSIM_EXP_ROOT="/scratch/ah7072/experiments"
export DP_PREDS="none"

# Multi-node DDP environment variables
export MASTER_PORT=12357
export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)

echo "Multi-node DDP setup:"
echo "  Master node: $MASTER_ADDR"
echo "  Master port: $MASTER_PORT"
echo "  Total tasks: $SLURM_NTASKS"
echo "  Tasks per node: $SLURM_NTASKS_PER_NODE"
echo ""

# Create experiment directory
EXPERIMENT_NAME="exp_a2_ego_mlp_100pct_$(date +%Y%m%d_%H%M%S)"
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
echo "  Architecture: 3-layer MLP [8 → 256 → 256 → 256 → 24]"
echo "  Trainable params: ~469K"
echo "  Input: Ego velocity (2D) + acceleration (2D) + command (4D one-hot)"
echo "  Output: 8 poses × 3 coords (x, y, heading)"
echo "  Data: 100% navtrain (~103K scenarios)"
echo "  Batch size: 32 per GPU (256 effective)"
echo "  Epochs: 50"
echo "  Expected time: ~8 hours"
echo ""

# Launch training with srun for proper multi-node DDP
srun python navsim/planning/script/run_training.py \
    agent=ego_status_mlp_agent \
    experiment_name="${EXPERIMENT_NAME}" \
    train_test_split=navtrain \
    cache_path=null \
    trainer.params.max_epochs=50 \
    trainer.params.accelerator=gpu \
    trainer.params.num_nodes=2 \
    dataloader.params.batch_size=32 \
    dataloader.params.num_workers=2 \
    dataloader.params.pin_memory=true \

echo ""
echo "=============================================="
echo "Training complete at $(date)"
echo "Results saved to: ${NAVSIM_EXP_ROOT}/${EXPERIMENT_NAME}"
echo "=============================================="