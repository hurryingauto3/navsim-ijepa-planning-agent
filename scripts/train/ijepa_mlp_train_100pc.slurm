#!/bin/bash
#SBATCH --job-name=ijepa_mlp_100pct
#SBATCH --partition=l40s_public
#SBATCH --account=torch_pr_68_tandon_advanced
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:4
#SBATCH --mem=200GB
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/ah7072/experiments/logs/output/train_ijepa_mlp_100_%j.out
#SBATCH --error=/scratch/ah7072/experiments/logs/error/train_ijepa_mlp_100_%j.err
#SBATCH --requeue

# =============================================================================
# I-JEPA MLP @ 100% Training: 4 Nodes × 4 L40S GPUs = 16 GPUs Total
# Purpose: Multi-modal baseline (Camera + LiDAR)
# Expected: ~20 hours, PDMS ~82-84
# =============================================================================

echo "=============================================="
echo "EXP-A4: I-JEPA MLP @ 100% navtrain"
echo "Configuration: 4 nodes × 4 L40S GPUs = 16 GPUs"
echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "=============================================="

# Environment setup
export HYDRA_FULL_ERROR=1
export NAVSIM_DEVKIT_ROOT="/scratch/ah7072/navsim"
export OPENSCENE_DATA_ROOT="/scratch/ah7072/data"
export NUPLAN_MAPS_ROOT="/scratch/ah7072/data/maps"
export NAVSIM_EXP_ROOT="/scratch/ah7072/experiments"
export DP_PREDS="none"

# Limit per-rank threading so communication has headroom
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Multi-node DDP environment variables
export MASTER_PORT=12360
export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)

echo "Multi-node DDP setup:"
echo "  Master node: $MASTER_ADDR"
echo "  Master port: $MASTER_PORT"
echo "  Total tasks: $SLURM_NTASKS"
echo "  Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "  GPUs per node: 4"
echo "  Total GPUs: 16"
echo ""

# Create experiment directory
EXPERIMENT_NAME="exp_a5_ijepa_mlp_100pct_$(date +%Y%m%d_%H%M%S)"
mkdir -p /scratch/ah7072/experiments/logs/output
mkdir -p /scratch/ah7072/experiments/logs/error

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

# Set PYTHONPATH
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}:${PYTHONPATH:-}"

# Distributed training debug (optional, comment out if too verbose)
# export NCCL_DEBUG=INFO
# export TORCH_DISTRIBUTED_DEBUG=INFO

echo "Environment check:"
echo "  Python: $(which python)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "  PYTHONPATH: ${PYTHONPATH}"
echo ""

echo "Experiment configuration:"
echo "  Agent: ijepa_planning_agent"
echo "  Architecture: I-JEPA MLP"
echo "  Trainable params: ~45M"
echo "  Input: 1 front camera"
echo "  Data: 100% navtrain (~103K scenarios)"
echo "  Nodes: ${SLURM_JOB_NUM_NODES}"
echo "  GPUs total: ${SLURM_NTASKS}"
echo "  Epochs: 20"
echo "  Expected time: ~20 hours"
echo ""
TOTAL_GLOBAL_BATCH=16
COMPUTE_NODES=${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-1}}
if [ "${COMPUTE_NODES}" -lt 1 ]; then
    COMPUTE_NODES=1
fi
PER_NODE_BATCH=$((TOTAL_GLOBAL_BATCH / COMPUTE_NODES))
if [ "${PER_NODE_BATCH}" -lt 1 ]; then
    PER_NODE_BATCH=1
fi
echo "  Batch size: ${TOTAL_GLOBAL_BATCH} (per node: ${PER_NODE_BATCH})"
echo "  Workers per task: ${SLURM_CPUS_PER_TASK}"

# Derive dataloader worker count; leave a couple of CPUs for comms/logging
NUM_DATALOADER_WORKERS=$((SLURM_CPUS_PER_TASK - 2))
if [ "${NUM_DATALOADER_WORKERS}" -lt 1 ]; then
    NUM_DATALOADER_WORKERS=1
fi
echo "  Dataloader workers used: ${NUM_DATALOADER_WORKERS}"

# CRITICAL: Use srun for proper multi-node DDP
srun python navsim/planning/script/run_training.py \
    agent=ijepa_planning_agent \
    experiment_name="${EXPERIMENT_NAME}" \
    train_test_split=navtrain \
    cache_path=null \
    trainer.params.max_epochs=20 \
    trainer.params.accelerator=gpu \
    trainer.params.num_nodes=${SLURM_JOB_NUM_NODES} \
    dataloader.params.batch_size=16 \
    dataloader.params.num_workers=${NUM_DATALOADER_WORKERS} \
    dataloader.params.pin_memory=true \

echo ""
echo "=============================================="
echo "Training complete at $(date)"
echo "Results saved to: ${NAVSIM_EXP_ROOT}/${EXPERIMENT_NAME}"
echo "Checkpoint: ${NAVSIM_EXP_ROOT}/${EXPERIMENT_NAME}/checkpoints/last.ckpt"
echo "=============================================="
