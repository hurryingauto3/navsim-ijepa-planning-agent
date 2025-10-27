#!/bin/bash
#SBATCH --job-name=download_navtest
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=24:00:00   
#SBATCH --output=/scratch/ah7072/data/logs/download_navtest_%j.out
#SBATCH --error=/scratch/ah7072/data/logs/download_navtest_%j.err


set -e  # Exit on any error
set -x  # Print commands for debugging

echo "=========================================="
echo "NAVSIM navtest Dataset Download"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "=========================================="

# Define paths
DATA_ROOT="/scratch/ah7072/data"
OPENSCENE_DIR="${DATA_ROOT}/openscene"

# Create directory
mkdir -p ${OPENSCENE_DIR}
cd ${OPENSCENE_DIR}

echo "Step 0/3: Downloading navhard two-stage dataset..."
wget https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/navsim-v2/navsim_v2.2_navhard_two_stage_curr_sensors.tar.gz
wget https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/navsim-v2/navsim_v2.2_navhard_two_stage_hist_sensors.tar.gz
wget https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/navsim-v2/navsim_v2.2_navhard_two_stage_scene_pickles.tar.gz

echo "Extracting navhard two-stage dataset..."
tar -xzvf navsim_v2.2_navhard_two_stage_curr_sensors.tar.gz
tar -xzvf navsim_v2.2_navhard_two_stage_hist_sensors.tar.gz
tar -xzvf navsim_v2.2_navhard_two_stage_scene_pickles.tar.gz

echo "Cleaning up tar files..."
rm navsim_v2.2_navhard_two_stage_curr_sensors.tar.gz
rm navsim_v2.2_navhard_two_stage_hist_sensors.tar.gz
rm navsim_v2.2_navhard_two_stage_scene_pickles.tar.gz

echo "=========================================="
echo "navhard two-stage dataset download complete!"
echo "Location: ${OPENSCENE_DIR}"
echo "Final storage usage:"
du -sh ${OPENSCENE_DIR}
echo "Final file count:"
find ${OPENSCENE_DIR} -type f | wc -l
echo "End time: $(date)"
echo "=========================================="

