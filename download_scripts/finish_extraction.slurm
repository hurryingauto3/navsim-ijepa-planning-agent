#!/bin/bash
#SBATCH --job-name=extract_missing_scenes
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=3:00:00
#SBATCH --output=/scratch/ah7072/data/logs/extract_scenes_%j.out

set -e
set -x

echo "==========================================="
echo "Extracting Missing Navtrain Scenes"
echo "Start: $(date)"
echo "==========================================="

cd /scratch/ah7072/data/openscene

# Track initial state
echo "BEFORE EXTRACTION:"
echo "Scenes: $(ls -d trainval_sensor_blobs/trainval/2021* | wc -l)"
echo "Size: $(du -sh trainval_sensor_blobs/trainval/ | cut -f1)"
echo ""

# Extract history archives (which contain additional scenes)
for i in 1 2 3 4; do
    echo "===== Extracting navtrain_history_${i}.tgz ====="
    tar -xzf navtrain_history_${i}.tgz
    
    echo "Merging history_split_${i} scenes into trainval..."
    rsync -av --info=progress2 history_split_${i}/* trainval_sensor_blobs/trainval/
    
    rm -rf history_split_${i}
    
    echo "Progress ${i}/4:"
    echo "  Scenes now: $(ls -d trainval_sensor_blobs/trainval/2021* | wc -l)"
    echo "  Size now: $(du -sh trainval_sensor_blobs/trainval/ | cut -f1)"
    echo ""
done

echo "==========================================="
echo "AFTER EXTRACTION:"
echo "==========================================="
echo "Total scenes: $(ls -d trainval_sensor_blobs/trainval/2021* | wc -l)"
echo "Total size: $(du -sh trainval_sensor_blobs/trainval/ | cut -f1)"
echo "Total files: $(find trainval_sensor_blobs/trainval/ -type f | wc -l)"
echo ""
echo "Expected: ~1800-2000 scenes, ~580-600GB, ~2M files"
echo "End: $(date)"
echo "==========================================="