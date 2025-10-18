#!/bin/bash
#SBATCH --job-name=cache_ray_limited
#SBATCH --output=/scratch/ah7072/navsim_workspace/exp/logs/train_%j.out
#SBATCH --error=/scratch/ah7072/navsim_workspace/exp/logs/train_%j.err
#SBATCH --partition=h200_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=196GB
#SBATCH --time=24:00:00

# NAVSIM Ray Feature Caching with Limited Workers
# Limits Ray workers to prevent OOM

echo "=== NAVSIM Ray Caching (Limited Workers) ==="
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

# Ray settings - CRITICAL: limit memory and workers
export RAY_memory_monitor_refresh_ms=0
export RAY_OBJECT_STORE_MEMORY=30000000000  # 30GB
export RAY_memory=160000000000              # 160GB total

# PyTorch settings
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

cd $NAVSIM_DEVKIT_ROOT

echo "=== Starting Ray Caching with 4 workers (limited to avoid OOM) ==="

# Run with LIMITED Ray workers
python navsim/planning/script/run_dataset_caching_gpu.py \
    agent=hydra_mdp_v8192_w_ep \
    train_test_split=trainval \
    experiment_name=training_cache_ray \
    cache_path=/scratch/$USER/experiments/training_cache \
    force_cache_computation=true \
    worker=ray_distributed \
    worker.threads_per_node=4 \
    worker.use_distributed=false

echo "=== Caching Complete ==="
echo "Date: $(date)"
