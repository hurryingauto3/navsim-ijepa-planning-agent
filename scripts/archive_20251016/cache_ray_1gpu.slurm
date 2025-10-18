#!/bin/bash
#SBATCH --job-name=cache_ray_1gpu
#SBATCH --output=/scratch/ah7072/navsim_workspace/exp/logs/train_%j.out
#SBATCH --error=/scratch/ah7072/navsim_workspace/exp/logs/train_%j.err
#SBATCH --partition=h200_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=196GB
#SBATCH --time=24:00:00

# NAVSIM Feature Caching with Ray + 1 GPU
# Modified caching script with shared agent across workers

echo "=== NAVSIM Ray + 1 GPU Feature Caching ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: 196GB"

# Load environment
module purge
module load anaconda3/2025.06
source activate navsim

# Set environment variables
export NAVSIM_DEVKIT_ROOT="/scratch/$USER/GTRS"
export OPENSCENE_DATA_ROOT="/scratch/$USER/data/openscene"
export NUPLAN_MAPS_ROOT="/scratch/$USER/data/maps"
export NAVSIM_EXP_ROOT="/scratch/$USER/experiments"

# Ray configuration - optimized for single GPU with limited workers
export RAY_memory_monitor_refresh_ms=0
export RAY_OBJECT_STORE_MEMORY=40000000000    # 40GB for Ray object store
export RAY_memory=150000000000                # 150GB total RAM for Ray

# PyTorch settings - optimize CUDA memory
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=4

cd $NAVSIM_DEVKIT_ROOT

echo ""
echo "=== Configuration ==="
echo "Worker Type: ray_distributed"
echo "Workers per node: 4 (limited to prevent OOM)"
echo "Strategy: Each Ray worker gets ONE agent instance (reused for all batches)"
echo "GPU: Shared across all workers via CUDA"
echo ""

echo "=== Starting Ray Caching ==="
date

# Run caching with Ray - LIMITED to 4 workers to prevent OOM
python navsim/planning/script/run_dataset_caching.py \
    agent=hydra_mdp_v8192_w_ep \
    train_test_split=trainval \
    experiment_name=training_cache_ray_1gpu \
    cache_path=/scratch/$USER/experiments/training_cache \
    force_cache_computation=true \
    worker=ray_distributed \
    worker.threads_per_node=4 \
    worker.use_distributed=false \
    worker.log_to_driver=true

echo ""
echo "=== Caching Complete ==="
echo "Cache location: /scratch/$USER/experiments/training_cache"
echo "Date: $(date)"
echo ""
echo "Next step: Train with cache using:"
echo "  cache_path=/scratch/$USER/experiments/training_cache"
