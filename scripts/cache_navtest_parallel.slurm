#!/bin/bash
#SBATCH --job-name=cache_navtest_parallel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=256GB
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/ah7072/experiments/logs/caching/cache_navtest_%j.out
#SBATCH --error=/scratch/ah7072/experiments/logs/caching/cache_navtest_%j.err
#SBATCH --mail-user=ah7072@nyu.edu
#SBATCH --mail-type=END,FAIL

# =============================================================================
# Fast parallel metric caching for navtest split
# Uses 128 CPUs for maximum parallelism (no GPU needed)
# =============================================================================
# Activate Conda Environment
# Prefer local Miniconda in /scratch/$USER, fallback to system module
export CONDA_ROOT="/scratch/$USER/miniconda3"

if [ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]; then
    # Use conda installed in scratch
    source "${CONDA_ROOT}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ROOT}/envs/navsim"
    echo "✓ Using local Miniconda environment"
else
    # Fallback to system module
    module purge || true
    module load anaconda3/2025.06 || true
    if command -v conda &> /dev/null; then
        source $(conda info --base)/etc/profile.d/conda.sh || true
        conda activate navsim || source activate navsim || true
        echo "✓ Using system Anaconda environment"
    else
        echo "ERROR: conda not found; evaluation will fail"
        exit 1
    fi
fi

# Set Python path and error reporting
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1

# Environment setup
export NAVSIM_DEVKIT_ROOT="/scratch/ah7072/navsim"
export OPENSCENE_DATA_ROOT="/scratch/ah7072/data/openscene"
export NUPLAN_MAPS_ROOT="/scratch/ah7072/data/maps"
export NAVSIM_EXP_ROOT="/scratch/ah7072/experiments"
export EVAL_SPLIT="navtest"

echo "=============================================="
echo "Parallel Metric Caching: ${EVAL_SPLIT} Split"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Node: $SLURM_NODE_NAME"
echo "=============================================="

cd "${NAVSIM_DEVKIT_ROOT}"

# Remove duplicate conda activation - already done above

echo "Using Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Run with maximum parallelism
# Use exact GTRS parameter names: train_test_split (not split)
# worker and max_number_of_workers match default_common.yaml defaults
python navsim/planning/script/run_metric_caching.py \
    train_test_split="${EVAL_SPLIT}" \
    metric_cache_path="${NAVSIM_EXP_ROOT}/metric_cache" \
    worker=ray_distributed_no_torch \
    max_number_of_workers="${SLURM_CPUS_PER_TASK}"

echo ""
echo "=============================================="
echo "Metric caching complete!"
echo "Cache saved to: ${NAVSIM_EXP_ROOT}/metric_cache"
echo "=============================================="
