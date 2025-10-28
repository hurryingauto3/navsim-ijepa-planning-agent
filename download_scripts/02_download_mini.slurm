#!/bin/bash
#SBATCH --job-name=download_mini
#SBATCH --nodes=1
#SBATCH --account=pr_141_general
#SBATCH --cpus-per-task=4
#SBATCH --mem=128GB
#SBATCH --time=24:00:00   
#SBATCH --output=/vast/ah7072/data/logs/download_mini_%j.out
#SBATCH --error=/vast/ah7072/data/logs/download_mini_%j.err

# NAVSIM Mini Dataset Download - Torch Cluster
# Downloads navmini (~152GB, ~1M files)
# Use this to test your workflow before downloading the full navtrain

set -e  # Exit on any error
set -x  # Print commands for debugging

echo "=========================================="
echo "NAVSIM Mini Dataset Download"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "=========================================="

# Define paths
DATA_ROOT="/vast/ah7072/data"
OPENSCENE_DIR="${DATA_ROOT}/openscene"

# Create directory
mkdir -p ${OPENSCENE_DIR}
cd ${OPENSCENE_DIR}

echo "Step 1/3: Downloading metadata..."
wget https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_metadata_mini.tgz
tar -xzf openscene_metadata_mini.tgz
rm openscene_metadata_mini.tgz
mv openscene-v1.1/meta_datas mini_navsim_logs
rm -r openscene-v1.1

echo "Step 2/3: Downloading camera images (32 chunks)..."
for split in {0..31}; do
    echo "  Downloading camera chunk ${split}/31..."
    wget https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_sensor_mini_camera/openscene_sensor_mini_camera_${split}.tgz
    echo "  Extracting camera chunk ${split}/31..."
    tar -xzf openscene_sensor_mini_camera_${split}.tgz
    rm openscene_sensor_mini_camera_${split}.tgz
done

echo "Step 3/3: Downloading LiDAR data (32 chunks)..."
for split in {0..31}; do
    echo "  Downloading LiDAR chunk ${split}/31..."
    wget https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_sensor_mini_lidar/openscene_sensor_mini_lidar_${split}.tgz
    echo "  Extracting LiDAR chunk ${split}/31..."
    tar -xzf openscene_sensor_mini_lidar_${split}.tgz
    rm openscene_sensor_mini_lidar_${split}.tgz
done

echo "Finalizing structure..."
mv openscene-v1.1/sensor_blobs mini_sensor_blobs
rm -r openscene-v1.1

echo "=========================================="
echo "Mini dataset download complete!"
echo "Location: ${OPENSCENE_DIR}"
echo "Storage usage:"
du -sh ${OPENSCENE_DIR}
echo "File count:"
find ${OPENSCENE_DIR} -type f | wc -l
echo "End time: $(date)"
echo "=========================================="
