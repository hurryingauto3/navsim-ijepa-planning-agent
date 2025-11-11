#!/bin/bash
#SBATCH --job-name=ijepa_planning_agent_h200
#SBATCH --partition=h200_public
#SBATCH --account=torch_pr_68_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:8
#SBATCH --mem=384GB
#SBATCH --time=15:00:00
#SBATCH --output=/scratch/ah7072/experiments/logs/output/train_ijepa_planning_agent_h200_%j.out
#SBATCH --error=/scratch/ah7072/experiments/logs/error/train_ijepa_planning_agent_h200_%j.err
#SBATCH --requeue

# =============================================================================
# ijepa_planning_agent @ 100% Training: 1 Node × 8 H200 GPUs
# Purpose: Single-node baseline on H200 hardware
# Expected: Faster training time compared to L40S
# =============================================================================

echo "=============================================="
echo "EXP-A4: ijepa_planning_agent @ 100% navtrain on H200"
echo "Configuration: 1 node × 8 H200 GPUs"
echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "=============================================="

# Environment setup
export NAVSIM_DEVKIT_ROOT="/scratch/ah7072/navsim"
export OPENSCENE_DATA_ROOT="/scratch/ah7072/data/openscene"
export NUPLAN_MAPS_ROOT="/scratch/ah7072/data/maps"
export NAVSIM_EXP_ROOT="/scratch/ah7072/experiments"
export HYDRA_FULL_ERROR=1
export DP_PREDS="none"

# Multi-node DDP environment variables (still needed for single-node multi-GPU)
export MASTER_PORT=12360
export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)

echo "DDP setup:"
echo "  Master node: $MASTER_ADDR"
echo "  Master port: $MASTER_PORT"
echo "  Total tasks: $SLURM_NTASKS"
echo "  Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "  GPUs per node: 8"
echo "  Total GPUs: 8"
echo ""

# Training command
srun python -u /path/to/your/training_script.py \
    --config-path=/path/to/your/config \
    --config-name=your_config_name
