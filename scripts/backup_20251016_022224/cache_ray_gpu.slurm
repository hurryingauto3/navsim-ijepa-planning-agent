#!/bin/bash
#SBATCH --job-name=cache_ray_gpu
#SBATCH --output=/scratch/ah7072/navsim_workspace/exp/logs/train_%j.out
#SBATCH --error=/scratch/ah7072/navsim_workspace/exp/logs/train_%j.err
#SBATCH --partition=h200_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=196GB
#SBATCH --time=12:00:00

# NAVSIM Ray+GPU Feature Caching
# Uses 1 GPU with Ray workers to cache training features efficiently

echo "=== NAVSIM Ray+GPU Feature Caching ==="
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

# Ray configuration - reduce workers to avoid OOM
export RAY_memory_monitor_refresh_ms=0
export RAY_OBJECT_STORE_MEMORY=50000000000  # 50GB for Ray object store

# PyTorch settings
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

cd $NAVSIM_DEVKIT_ROOT

echo "=== Starting Ray+GPU Caching ==="
echo "Using modified caching script with GPU placement"

# Run caching with ray_distributed worker
python navsim/planning/script/run_dataset_caching_gpu.py \
    agent=hydra_mdp_v8192_w_ep \
    train_test_split=trainval \
    experiment_name=training_cache \
    cache_path=/scratch/$USER/experiments/training_cache \
    force_cache_computation=true \
    worker=ray_distributed \
    worker.threads_per_node=4 \
    worker.use_distributed=false

echo "=== Caching Complete ==="
echo "Date: $(date)"
