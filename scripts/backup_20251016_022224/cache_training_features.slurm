#!/bin/bash
#SBATCH --job-name=cache_train_feat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=384GB
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/ah7072/navsim_workspace/exp/logs/cache_train_%j.out
#SBATCH --error=/scratch/ah7072/navsim_workspace/exp/logs/cache_train_%j.err
#SBATCH --mail-user=ah7072@nyu.edu
#SBATCH --mail-type=END,FAIL

# =============================================================================
# Fast parallel training feature cache creation
# Uses 128 CPUs for maximum parallelism (no GPU needed)
# This is DIFFERENT from metric cache - this caches preprocessed features
# =============================================================================

echo "=============================================="
echo "Creating Training Feature Cache (Parallel)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Node: $SLURMD_NODENAME"
echo "This is a one-time operation"
echo "=============================================="

# Environment setup
export NAVSIM_DEVKIT_ROOT="/scratch/ah7072/navsim_workspace/navsim"
export OPENSCENE_DATA_ROOT="/scratch/ah7072/navsim_workspace/dataset"
export NUPLAN_MAPS_ROOT="/scratch/ah7072/navsim_workspace/dataset/maps"
export NAVSIM_EXP_ROOT="/scratch/ah7072/navsim_workspace/exp"

cd "${NAVSIM_DEVKIT_ROOT}"

# Load conda environment
module load anaconda3/2025.06
eval "$(conda shell.bash hook)"
conda activate navsim

echo "Using Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Run with maximum parallelism for cache creation
# force_cache_computation=true will create the cache
# High num_workers for parallel feature extraction
python navsim/planning/script/run_training_dense.py \
    agent=hydra_mdp_v8192_w_ep \
    experiment_name="cache_creation" \
    cache_path="${NAVSIM_EXP_ROOT}/training_cache" \
    train_test_split=navtrain \
    force_cache_computation=true \
    ~trainer.params.strategy \
    trainer.params.max_epochs=1 \
    trainer.params.limit_train_batches=1 \
    trainer.params.limit_val_batches=1 \
    trainer.params.accelerator=cpu \
    dataloader.params.batch_size=1 \
    dataloader.params.num_workers=${SLURM_CPUS_PER_TASK} \
    dataloader.params.pin_memory=false \
    dataloader.params.prefetch_factor=2

echo ""
echo "=============================================="
echo "Training cache created!"
echo "Cache location: ${NAVSIM_EXP_ROOT}/training_cache"
echo "Now training will be much faster!"
echo "=============================================="
