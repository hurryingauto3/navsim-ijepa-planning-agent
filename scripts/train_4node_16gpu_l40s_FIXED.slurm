#!/bin/bash
#SBATCH --job-name=hydra_4node_16gpu
#SBATCH --partition=l40s_public
#SBATCH --account=torch_pr_68_tandon_advanced
#SBATCH --nodes=4
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=400GB
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/ah7072/navsim_workspace/exp/logs/train_%j.out
#SBATCH --error=/scratch/ah7072/navsim_workspace/exp/logs/train_%j.err
#SBATCH --requeue

# =============================================================================
# Multi-Node DDP Training: 4 Nodes × 4 L40S GPUs = 16 GPUs Total
# CRITICAL: Uses srun to launch processes correctly across nodes
# =============================================================================

echo "=============================================="
echo "Hydra-MDP Multi-Node DDP Training (FIXED)"
echo "Configuration: 4 nodes × 4 L40S GPUs = 16 GPUs"
echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "=============================================="

# Environment setup
export NAVSIM_DEVKIT_ROOT="/scratch/ah7072/GTRS"
export OPENSCENE_DATA_ROOT="/scratch/ah7072/data/openscene"
export NUPLAN_MAPS_ROOT="/scratch/ah7072/data/maps"
export NAVSIM_EXP_ROOT="/scratch/ah7072/experiments"
export DP_PREDS="none"

# Multi-node DDP environment variables
export MASTER_PORT=12356
export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)

echo "Multi-node DDP setup:"
echo "  Master node: $MASTER_ADDR"
echo "  Master port: $MASTER_PORT"
echo "  Total tasks: $SLURM_NTASKS"
echo "  Tasks per node: $SLURM_NTASKS_PER_NODE"
echo ""

# Create experiment directory
EXPERIMENT_NAME="hydra_w_ep_4node_16gpu_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${NAVSIM_EXP_ROOT}/${EXPERIMENT_NAME}/notes"

cd "${NAVSIM_DEVKIT_ROOT}"

# Load conda environment: prefer Miniconda in /scratch/$USER if available
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

echo "Environment check:"
echo "  Python: $(which python)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo ""

# CRITICAL: Use srun to launch across all nodes
# - srun ensures proper task distribution
# - Each task gets one GPU automatically via --gres
# - PyTorch Lightning detects SLURM environment and handles DDP

srun python navsim/planning/script/run_training_dense.py \
    agent=hydra_mdp_v8192_w_ep \
    experiment_name="${EXPERIMENT_NAME}" \
    train_test_split=navtrain \
    cache_path=null \
    ~trainer.params.strategy \
    trainer.params.max_epochs=40 \
    trainer.params.accelerator=gpu \
    trainer.params.num_nodes=4 \
    trainer.params.precision=16-mixed \
    dataloader.params.batch_size=22 \
    dataloader.params.num_workers=2 \
    dataloader.params.pin_memory=true

echo ""
echo "=============================================="
echo "Training complete at $(date)"
echo "=============================================="
