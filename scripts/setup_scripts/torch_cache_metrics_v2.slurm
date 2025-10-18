#!/bin/bash
#SBATCH --job-name=cache_metrics
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80GB
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/ah7072/navsim_workspace/exp/logs/cache_metrics_%j.out
#SBATCH --error=/scratch/ah7072/navsim_workspace/exp/logs/cache_metrics_%j.err

# =============================================================================
# Metric Caching Script for NAVSIM (Torch HPC)
# 
# This pre-computes expensive operations like map access and coordinate
# transformations for faster evaluation. Run this ONCE before evaluation.
# 
# Uses Ubuntu 24.04 container with bind-mounted conda environment
# =============================================================================

# Environment setup
export NAVSIM_DEVKIT_ROOT="/scratch/ah7072/navsim_workspace/navsim"
export OPENSCENE_DATA_ROOT="/scratch/ah7072/navsim_workspace/dataset"
export NUPLAN_MAPS_ROOT="/scratch/ah7072/navsim_workspace/dataset/maps"
export NAVSIM_EXP_ROOT="/scratch/ah7072/navsim_workspace/exp"
export APPTAINER_BINDPATH=/scratch,/state/partition1,/mnt,/share/apps,/home/$USER

# Parse split argument (default: navtest)
SPLIT="${1:-navtest}"

echo "=============================================="
echo "Metric Caching for Split: $SPLIT"
echo "=============================================="
echo "Cache will be saved to: ${NAVSIM_EXP_ROOT}/metric_cache"
echo ""

# Create cache directory
CACHE_PATH="${NAVSIM_EXP_ROOT}/metric_cache"
mkdir -p "${CACHE_PATH}"

# Navigate to repo
cd "${NAVSIM_DEVKIT_ROOT}"

# Use Ubuntu 24.04 container with conda environment bind-mounted
CONTAINER_IMAGE="/share/apps/images/ubuntu-24.04.3.sif"

# Run metric caching with conda environment from your home directory
/share/apps/apptainer/bin/singularity exec --nv \
  --bind /home/$USER/.conda:/conda_env:ro \
  ${CONTAINER_IMAGE} \
  bash -c "
    source /conda_env/etc/profile.d/conda.sh && \
    conda activate navsim && \
    python navsim/planning/script/run_metric_caching.py \
      train_test_split=${SPLIT} \
      metric_cache_path=${CACHE_PATH}
  "

echo ""
echo "=============================================="
echo "Metric caching complete for split: ${SPLIT}"
echo "Cache location: ${CACHE_PATH}/${SPLIT}"
echo "=============================================="
