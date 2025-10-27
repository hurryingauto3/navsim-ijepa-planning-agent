#!/bin/bash
#SBATCH --job-name=download_navtrain
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/ah7072/data/logs/download_navtrain_%j.out
#SBATCH --error=/scratch/ah7072/data/logs/download_navtrain_%j.err

# NAVSIM Navtrain Dataset Download - Improved & Robust
# - Verifies MD5 checksums for all downloaded archives.
# - Automatically re-downloads corrupted files.
# - Skips downloads if valid files already exist.
# - Preserves downloaded .tgz archives for future checks.

set -e  # Exit on any error
set -x  # Print commands for debugging

echo "=========================================="
echo "NAVSIM Navtrain Dataset Download (Robust)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "=========================================="

# Define paths
DATA_ROOT="/scratch/ah7072/data"
OPENSCENE_DIR="${DATA_ROOT}/openscene"
DOWNLOAD_URL_BASE="https://s3.eu-central-1.amazonaws.com/avg-projects-2/navsim"

# --- Arrays for file chunks and checksums ---
CHUNKS_CURRENT=(
    "navtrain_current_1.tgz"
    "navtrain_current_2.tgz"
    "navtrain_current_3.tgz"
    "navtrain_current_4.tgz"
)
MD5_CURRENT=(
    "6f92f38d5f03ed852da7872a7122bdd2"
    "7a72f0a758b5df6cbe4c565920a4869f"
    "b083fce1428308abb5682a1a150cc1af"
    "68354ac2c993aa1ebbfac59547fdb840"
)
CHUNKS_HISTORY=(
    "navtrain_history_1.tgz"
    "navtrain_history_2.tgz"
    "navtrain_history_3.tgz"
    "navtrain_history_4.tgz"
)
MD5_HISTORY=(
    "dc46ed34d92d5ab9cc1464d67b72fbf6"
    "fab177bdb79c0c9536da1566d13e5995"
    "71ed9a2387edc3849921861d7873c7f0"
    "2cc13aced2f458e50fe4cc2f26d18e07"
)

# --- Helper function for download and verification ---
download_and_verify() {
    local chunk_name=$1
    local expected_md5=$2
    local split_dir_name=$3

    # Check if file exists and is valid
    local is_valid=false
    if [ -f "$chunk_name" ]; then
        echo "Verifying existing file: $chunk_name"
        if echo "$expected_md5  $chunk_name" | md5sum -c -; then
            echo "Checksum OK for $chunk_name."
            is_valid=true
        else
            echo "Checksum FAILED for $chunk_name. Deleting corrupted file."
            rm "$chunk_name"
        fi
    fi

    # Download if not valid/present
    if [ "$is_valid" = false ]; then
        echo "Downloading $chunk_name..."
        wget "${DOWNLOAD_URL_BASE}/${chunk_name}"
        echo "Verifying downloaded file: $chunk_name"
        if ! echo "$expected_md5  $chunk_name" | md5sum -c -; then
            echo "ERROR: Downloaded file $chunk_name is corrupt. Please check network or re-run."
            exit 1
        fi
    fi

    echo "Extracting file $chunk_name..."
    tar -xzf "$chunk_name"
    
    echo "Moving files to final location..."
    rsync -rv --remove-source-files ${split_dir_name}/* trainval_sensor_blobs/trainval/
    rm -r "${split_dir_name}"
}


# --- Main script execution ---
mkdir -p "${OPENSCENE_DIR}"
cd "${OPENSCENE_DIR}"

echo "Step 1/3: Downloading metadata..."
if [ ! -d "trainval_navsim_logs" ]; then
    wget https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_metadata_trainval.tgz
    tar -xzf openscene_metadata_trainval.tgz
    rm openscene_metadata_trainval.tgz
    mv openscene-v1.1/meta_datas trainval_navsim_logs
    rm -r openscene-v1.1
else
    echo "Metadata directory 'trainval_navsim_logs' already exists. Skipping download."
fi

echo "Step 2/3: Downloading and processing current frame images..."
mkdir -p trainval_sensor_blobs/trainval
for i in "${!CHUNKS_CURRENT[@]}"; do
    CHUNK="${CHUNKS_CURRENT[$i]}"
    MD5="${MD5_CURRENT[$i]}"
    SPLIT_DIR="current_split_$((i+1))"
    download_and_verify "$CHUNK" "$MD5" "$SPLIT_DIR"
    echo "  Current storage usage:"
    du -sh "${OPENSCENE_DIR}"
done

echo "Step 3/3: Downloading and processing history frame images..."
for i in "${!CHUNKS_HISTORY[@]}"; do
    CHUNK="${CHUNKS_HISTORY[$i]}"
    MD5="${MD5_HISTORY[$i]}"
    SPLIT_DIR="history_split_$((i+1))"
    download_and_verify "$CHUNK" "$MD5" "$SPLIT_DIR"
    echo "  Current storage usage:"
    du -sh "${OPENSCENE_DIR}"
done


echo "=========================================="
echo "Navtrain dataset download complete!"
echo "Location: ${OPENSCENE_DIR}"
echo "Final storage usage:"
du -sh "${OPENSCENE_DIR}"
echo "Final file count:"
find "${OPENSCENE_DIR}" -type f | wc -l
echo "End time: $(date)"
echo "=========================================="

echo ""
echo "Next steps:"
echo "1. The script has already verified data integrity with MD5 checksums."
echo "2. Set environment variables in ~/.bashrc if not already done:"
echo "   export OPENSCENE_DATA_ROOT=/scratch/ah7072/data/openscene"
echo "   export NUPLAN_MAPS_ROOT=/scratch/ah7072/data/maps"
echo "3. Run training: sbatch your_training_script.slurm"
