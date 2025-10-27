#!/bin/bash
#SBATCH --job-name=download_navtest
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=24:00:00   
#SBATCH --output=/scratch/ah7072/data/logs/download_navtest_%j.out
#SBATCH --error=/scratch/ah7072/data/logs/download_navtest_%j.err

# NAVSIM navtest Dataset Download - Torch Cluster
# Downloads full navtest split (~459GB, ~3M files)
# This is the main testing dataset for your thesis

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

echo "Step 1/3: Downloading metadata..."
wget https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_metadata_test.tgz
tar -xzf openscene_metadata_test.tgz
rm openscene_metadata_test.tgz

echo "Step 2/3: Downloading current frame images (4 chunks)..."
for split in {0..31}; do
    wget https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_sensor_test_camera/openscene_sensor_test_camera_${split}.tgz
    echo "Extracting file openscene_sensor_test_camera_${split}.tgz"
    tar -xzf openscene_sensor_test_camera_${split}.tgz
    rm openscene_sensor_test_camera_${split}.tgz
done

echo "Step 3/3: Downloading history frame images (4 chunks)..."
for split in {0..31}; do
    wget https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_sensor_test_lidar/openscene_sensor_test_lidar_${split}.tgz
    echo "Extracting file openscene_sensor_test_lidar_${split}.tgz"
    tar -xzf openscene_sensor_test_lidar_${split}.tgz
    rm openscene_sensor_test_lidar_${split}.tgz
done

mv openscene-v1.1/meta_datas test_navsim_logs
mv openscene-v1.1/sensor_blobs test_sensor_blobs
rm -r openscene-v1.1

echo "=========================================="
echo "navtest dataset download complete!"
echo "Location: ${OPENSCENE_DIR}"
echo "Final storage usage:"
du -sh ${OPENSCENE_DIR}
echo "Final file count:"
find ${OPENSCENE_DIR} -type f | wc -l
echo "End time: $(date)"
echo "=========================================="

echo ""
echo "Next steps:"
echo "1. Verify data integrity"
echo "2. Set environment variables in ~/.bashrc:"
echo "   export OPENSCENE_DATA_ROOT=/scratch/ah7072/data/openscene"
echo "   export NUPLAN_MAPS_ROOT=/scratch/ah7072/data/maps"
echo "3. Run training: sbatch your_training_script.slurm"
