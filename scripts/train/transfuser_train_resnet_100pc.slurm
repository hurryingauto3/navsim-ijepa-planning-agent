#!/bin/bash
#SBATCH --job-name=transfuser_100pct
#SBATCH --partition=l40s_public
#SBATCH --account=torch_pr_68_tandon_advanced
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:4
#SBATCH --mem=200GB
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/ah7072/experiments/logs/output/train_transfuser_100_%j.out
#SBATCH --error=/scratch/ah7072/experiments/logs/error/train_transfuser_100_%j.err
#SBATCH --requeue

# =============================================================================
# TransFuser @ 100% Training: 4 Nodes × 4 L40S GPUs = 16 GPUs Total
# Purpose: Multi-modal baseline (Camera + LiDAR)
# Expected: ~20 hours, PDMS ~82-84
# =============================================================================

echo "=============================================="
echo "EXP-A4: TransFuser @ 100% navtrain"
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
EXPERIMENT_NAME="exp_a4_transfuser_100pct_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${NAVSIM_EXP_ROOT}/${EXPERIMENT_NAME}/notes"
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
echo "  Agent: transfuser_agent"
echo "  Architecture: ResNet34 (Image) + ResNet34 (LiDAR) + Transformer"
echo "  Trainable params: ~45M"
echo "  Input: 3 cameras (1024×256) + LiDAR BEV (256×256)"
echo "  Auxiliary: 3D detection + BEV segmentation"
echo "  Data: 100% navtrain (~103K scenarios)"
echo "  Batch size: 16 per GPU × 16 GPUs = 256 effective"
echo "  Epochs: 20"
echo "  Expected time: ~20 hours"
echo ""

# CRITICAL: Use srun for proper multi-node DDP
srun python navsim/planning/script/run_training.py \
    agent=transfuser_agent \
    experiment_name="${EXPERIMENT_NAME}" \
    train_test_split=navtrain \
    cache_path=null \
    trainer.params.max_epochs=20 \
    trainer.params.accelerator=gpu \
    trainer.params.num_nodes=4 \
    dataloader.params.batch_size=16 \
    dataloader.params.num_workers=2 \
    dataloader.params.pin_memory=true \

echo ""
echo "=============================================="
echo "Training complete at $(date)"
echo "Results saved to: ${NAVSIM_EXP_ROOT}/${EXPERIMENT_NAME}"
echo "Checkpoint: ${NAVSIM_EXP_ROOT}/${EXPERIMENT_NAME}/checkpoints/last.ckpt"
echo "=============================================="
