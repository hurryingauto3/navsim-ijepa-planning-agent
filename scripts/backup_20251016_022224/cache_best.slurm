#!/bin/bash
#SBATCH --job-name=cache_threadpool
#SBATCH --output=/scratch/ah7072/navsim_workspace/exp/logs/train_%j.out
#SBATCH --error=/scratch/ah7072/navsim_workspace/exp/logs/train_%j.err
#SBATCH --partition=h200_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=196GB
#SBATCH --time=24:00:00

# NAVSIM ThreadPool Feature Caching (BEST for single node)
# Uses ThreadPoolExecutor - all threads share same GPU model (no duplication!)

echo "=== NAVSIM ThreadPool Feature Caching (Recommended) ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "CPUs: $SLURM_CPUS_PER_TASK"

# Load environment
module purge
module load anaconda3/2025.06
source activate navsim

# Set environment variables
export NAVSIM_DEVKIT_ROOT="$HOME/GTRS"
export OPENSCENE_DATA_ROOT="/scratch/$USER/openscene"
export NUPLAN_MAPS_ROOT="/scratch/$USER/maps"
export NAVSIM_EXP_ROOT="/scratch/$USER/experiments"

# PyTorch settings - enable memory sharing
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=4

cd $NAVSIM_DEVKIT_ROOT

echo "=== Starting ThreadPool Caching ==="
echo "Worker: single_machine_thread_pool with 8 workers"
echo "Model loaded once on GPU, shared across all threads"

# Run caching with ThreadPoolExecutor (all threads share model in GPU memory)
python navsim/planning/script/run_dataset_caching_gpu.py \
    agent=hydra_mdp_v8192_w_ep \
    train_test_split=trainval \
    experiment_name=training_cache_threadpool \
    cache_path=/scratch/$USER/experiments/training_cache \
    force_cache_computation=true \
    worker=single_machine_thread_pool \
    worker.max_workers=8

echo "=== Caching Complete ==="
echo "Cache location: /scratch/$USER/experiments/training_cache"
echo "Date: $(date)"
