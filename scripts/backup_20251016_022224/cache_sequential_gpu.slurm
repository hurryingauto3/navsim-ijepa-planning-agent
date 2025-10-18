#!/bin/bash
#SBATCH --job-name=cache_sequential
#SBATCH --output=/scratch/ah7072/navsim_workspace/exp/logs/train_%j.out
#SBATCH --error=/scratch/ah7072/navsim_workspace/exp/logs/train_%j.err
#SBATCH --partition=h200_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=196GB
#SBATCH --time=24:00:00

# NAVSIM Sequential Feature Caching (NO Ray workers - avoid OOM)
# Single process with GPU - simplest and most memory-efficient

echo "=== NAVSIM Sequential Feature Caching ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

# Load environment
module purge
module load anaconda3/2025.06
source activate navsim

# Set environment variables
export NAVSIM_DEVKIT_ROOT="$HOME/GTRS"
export OPENSCENE_DATA_ROOT="/scratch/$USER/openscene"
export NUPLAN_MAPS_ROOT="/scratch/$USER/maps"
export NAVSIM_EXP_ROOT="/scratch/$USER/experiments"

# PyTorch settings
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

cd $NAVSIM_DEVKIT_ROOT

echo "=== Starting Sequential Caching (no Ray) ==="
echo "Using single_machine_thread_pool worker (CPU-based, no model duplication)"

# Run caching with single_machine_thread_pool (safest option)
python navsim/planning/script/run_dataset_caching_gpu.py \
    agent=hydra_mdp_v8192_w_ep \
    train_test_split=trainval \
    experiment_name=training_cache \
    cache_path=/scratch/$USER/experiments/training_cache \
    force_cache_computation=true \
    worker=single_machine_thread_pool \
    worker.max_workers=4

echo "=== Caching Complete ==="
echo "Date: $(date)"
