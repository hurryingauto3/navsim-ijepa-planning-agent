#!/bin/bash
#SBATCH --job-name=download_maps
#SBATCH --nodes=1
#SBATCH --account=pr_141_general
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=01:00:00
#SBATCH --output=/vast/ah7072/data/logs/download_maps_%j.out
#SBATCH --error=/vast/ah7072/data/logs/download_maps_%j.err

# NAVSIM Maps Download - Torch Cluster
# Downloads nuPlan maps (~9GB, required for all training)

set -e  # Exit on any error
set -x  # Print commands for debugging

echo "=========================================="
echo "NAVSIM Maps Download"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "=========================================="

# Define paths
DATA_ROOT="/vast/ah7072/data"
MAPS_DIR="${DATA_ROOT}/maps"

# Create directory
mkdir -p ${MAPS_DIR}
cd ${MAPS_DIR}

echo "Downloading nuPlan maps..."
wget https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-maps-v1.1.zip

echo "Extracting maps..."
unzip -q nuplan-maps-v1.1.zip

echo "Cleaning up..."
rm nuplan-maps-v1.1.zip
mv nuplan-maps-v1.0 maps

echo "=========================================="
echo "Maps download complete!"
echo "Location: ${MAPS_DIR}/maps"
du -sh ${MAPS_DIR}
echo "End time: $(date)"
echo "=========================================="
