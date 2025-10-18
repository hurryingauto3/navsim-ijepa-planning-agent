#!/bin/bash
#SBATCH --job-name=hydra_4node_16gpu
#SBATCH --partition=l40s_public
#SBATCH --account=users
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=256GB
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/ah7072/navsim_workspace/exp/logs/train_%j.out
#SBATCH --error=/scratch/ah7072/navsim_workspace/exp/logs/train_%j.err

# =============================================================================
# Multi-Node DDP Training: 4 Nodes × 4 L40S GPUs = 16 GPUs Total
# Maximum parallelism with L40S cluster!
# =============================================================================

echo "=============================================="
echo "Hydra-MDP Multi-Node DDP Training"
echo "Configuration: 4 nodes × 4 L40S GPUs = 16 GPUs"
echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Node rank: $SLURM_NODEID"
echo "=============================================="

# Environment setup
export NAVSIM_DEVKIT_ROOT="/scratch/ah7072/GTRS"
export OPENSCENE_DATA_ROOT="/scratch/ah7072/data/openscene"
export NUPLAN_MAPS_ROOT="/scratch/ah7072/data/maps"
export NAVSIM_EXP_ROOT="/scratch/ah7072/experiments"
export DP_PREDS="none"

# Multi-node DDP environment variables
export MASTER_PORT=12356
export WORLD_SIZE=$((SLURM_NNODES * 4))  # 4 nodes × 4 GPUs = 16 total
export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)

# Each task gets one GPU (SLURM handles GPU assignment)
# Do NOT set CUDA_VISIBLE_DEVICES - let SLURM manage it

echo "Multi-node DDP setup:"
echo "  Master node: $MASTER_ADDR"
echo "  Master port: $MASTER_PORT"
echo "  World size: $WORLD_SIZE"
echo "  Node ID: $SLURM_NODEID"
echo "  Local GPUs: 4"
echo ""

# Create experiment directory (only on master node)
if [ "$SLURM_NODEID" -eq 0 ]; then
    EXPERIMENT_NAME="hydra_w_ep_4node_16gpu_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "${NAVSIM_EXP_ROOT}/${EXPERIMENT_NAME}/notes"
    echo "$EXPERIMENT_NAME" > /tmp/experiment_name_${SLURM_JOB_ID}.txt
fi

# Wait for master node to create experiment name
sleep 5
EXPERIMENT_NAME=$(cat /tmp/experiment_name_${SLURM_JOB_ID}.txt)

cd "${NAVSIM_DEVKIT_ROOT}"

# Load conda environment
module purge
module load anaconda3/2025.06
source activate navsim

echo "Environment check on node $SLURM_NODEID:"
echo "  Python: $(which python)"
echo "  GPUs detected: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo ""

# MAXIMUM PARALLELISM:
# - 4 nodes × 4 GPUs = 16 GPUs total
# - batch_size=22: per GPU (22 × 16 = 352 effective batch size!)
# - Expected: ~5 hours for 40 epochs (2× faster than 8 GPUs)

python navsim/planning/script/run_training_dense.py \
    agent=hydra_mdp_v8192_w_ep \
    experiment_name="${EXPERIMENT_NAME}" \
    train_test_split=navtrain \
    cache_path=null \
    ~trainer.params.strategy \
    trainer.params.max_epochs=1 \
    trainer.params.accelerator=gpu \
    trainer.params.num_nodes=4 \
    trainer.params.precision=16-mixed \
    dataloader.params.batch_size=22 \
    dataloader.params.num_workers=8 \
    dataloader.params.pin_memory=true

echo ""
echo "=============================================="
echo "Training complete at $(date) on node $SLURM_NODEID"
echo "=============================================="
