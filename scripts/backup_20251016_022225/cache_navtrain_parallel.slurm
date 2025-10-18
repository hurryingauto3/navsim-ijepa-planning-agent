#!/bin/bash
#SBATCH --job-name=cache_navtrain
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=256GB
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/ah7072/navsim_workspace/exp/logs/cache_navtrain_%j.out
#SBATCH --error=/scratch/ah7072/navsim_workspace/exp/logs/cache_navtrain_%j.err
#SBATCH --mail-user=ah7072@nyu.edu
#SBATCH --mail-type=END,FAIL

# =============================================================================
# Fast parallel metric caching for navtrain split
# Uses 128 CPUs for maximum parallelism (no GPU needed)
# =============================================================================

echo "=============================================="
echo "Parallel Metric Caching: navtrain"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Node: $SLURMD_NODENAME"
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

# Run with maximum parallelism
# Use exact GTRS parameter names: train_test_split (not split)
# worker and max_number_of_workers match default_common.yaml defaults
python navsim/planning/script/run_metric_caching.py \
    train_test_split=navtrain \
    metric_cache_path="${NAVSIM_EXP_ROOT}/metric_cache" \
    worker=ray_distributed_no_torch \
    max_number_of_workers=${SLURM_CPUS_PER_TASK}

echo ""
echo "=============================================="
echo "Metric caching complete!"
echo "Cache saved to: ${NAVSIM_EXP_ROOT}/metric_cache/navtrain"
echo "=============================================="
