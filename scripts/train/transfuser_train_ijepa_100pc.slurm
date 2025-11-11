#!/bin/bash
# =============================================================================
# SLURM CONFIGURATION
#
# --> UNCOMMENT THE RESOURCE PROFILE YOU WANT TO USE <--
# =============================================================================

# --- Profile: L40S (4 GPUs/node) ---
#SBATCH --partition=l40s_public
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4

# --- Profile: H200 (8 GPUs/node) ---
##SBATCH --partition=h200_public
##SBATCH --ntasks-per-node=2
##SBATCH --gres=gpu:2

# =============================================================================
# COMMON JOB SETTINGS (edit as needed)
# =============================================================================
# -- Account --
## SBATCH --account=torch_pr_68_tandon_advanced
#SBATCH --account=torch_pr_68_general

# -- Job Resources --
#SBATCH --nodes=4
#SBATCH --cpus-per-task=4       # Number of CPU cores per task.
#SBATCH --mem=384GB              # Total memory per node.

# -- Job Details --
#SBATCH --job-name=navsim_train
#SBATCH --time=12:00:00          # Job time limit (HH:MM:SS).
#SBATCH --requeue                # Requeue job if it fails due to node failure.

# -- Logging --
#SBATCH --output=/scratch/ah7072/experiments/logs/output/train_%j.out
#SBATCH --error=/scratch/ah7072/experiments/logs/error/train_%j.err

# =============================================================================
# NAVSIM DENSE TRAJECTORY PREDICTION TRAINING SCRIPT
#
# Flexible configuration for various agents and data subsets.
# To run: sbatch --export=AGENT=...,DATA_PERCENT=... this_script.slurm
#
# See EXPERIMENT CONFIGURATION section for tunable variables.
# =============================================================================

# -----------------------------------------------------------------------------
# EXPERIMENT CONFIGURATION (edit or export these variables)
# -----------------------------------------------------------------------------
# Agent configuration (must match a config file in navsim/planning/script/config/common/agent)
export AGENT=${AGENT:-"transfuser_agent"}

# Backbone variant (e.g., "ijepa", "resnet34")
export BACKBONE=${BACKBONE:-"ijepa"}

# Data percentage (e.g., 100, 50, 25, 10)
export DATA_PERCENT=${DATA_PERCENT:-100}

# Training parameters
export EPOCHS=${EPOCHS:-20}
export BATCH_SIZE=${BATCH_SIZE:-16}
export NUM_WORKERS=${NUM_WORKERS:-2}
# -----------------------------------------------------------------------------

# Construct experiment name
export EXPERIMENT_NAME="exp_${AGENT}_${BACKBONE}_${DATA_PERCENT}pct_$(date +%Y%m%d_%H%M%S)"

echo "=============================================="
echo "Starting NAVSIM Training Job"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Agent: ${AGENT}"
echo "Backbone: ${BACKBONE}"
echo "Data Subset: ${DATA_PERCENT}%"
echo "Epochs: ${EPOCHS}"
echo "Batch Size: ${BATCH_SIZE}"
echo "----------------------------------------------"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "GPUs per node: $(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "=============================================="

# Environment setup
export HYDRA_FULL_ERROR=1
export NAVSIM_DEVKIT_ROOT="/scratch/ah7072/navsim"
export OPENSCENE_DATA_ROOT="/scratch/ah7072/data"
export NUPLAN_MAPS_ROOT="/scratch/ah7072/data/maps"
export NAVSIM_EXP_ROOT="/scratch/ah7072/experiments"
export WANDB_MODE="online" # Set to "disabled" for no logging

# DDP environment variables
export MASTER_PORT=${MASTER_PORT:-12360}
export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)

echo "DDP setup:"
echo "  Master node: $MASTER_ADDR"
echo "  Master port: $MASTER_PORT"
echo ""

# Create experiment directories
mkdir -p "${NAVSIM_EXP_ROOT}/${EXPERIMENT_NAME}/notes"
mkdir -p "${NAVSIM_EXP_ROOT}/logs/output"
mkdir -p "${NAVSIM_EXP_ROOT}/logs/error"

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

# Verify environment
echo "Environment check:"
echo "  Python: $(which python)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not Found')"
echo "  CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'Not Found')"
echo ""

# Determine train/test split based on data percentage
TRAIN_SPLIT="navtrain"
echo "Using data split: ${TRAIN_SPLIT}"

# CRITICAL: Use srun for proper DDP initialization
srun python navsim/planning/script/run_training.py \
    agent=${AGENT} \
    experiment_name="${EXPERIMENT_NAME}" \
    train_test_split=${TRAIN_SPLIT} \
    cache_path=null \
    dataloader.params.batch_size=${BATCH_SIZE} \
    dataloader.params.num_workers=${NUM_WORKERS} \
    trainer.params.max_epochs=${EPOCHS} \
    trainer.params.strategy=ddp_find_unused_parameters_true \
    trainer.params.num_nodes=${SLURM_NNODES} \
    trainer.params.accelerator=gpu \
    agent.config.backbone="${BACKBONE}"

echo ""
echo "=============================================="
echo "Training complete at $(date)"
echo "Results saved to: ${NAVSIM_EXP_ROOT}/${EXPERIMENT_NAME}"
echo "=============================================="

