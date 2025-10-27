#!/bin/bash
#SBATCH --job-name=download_navtrain
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --output=/vast/ah7072/data/logs/download_navtrain_%j.out
#SBATCH --error=/vast/ah7072/data/logs/download_navtrain_%j.err

set -euo pipefail
# ==== CONFIG ====
# Choose a base directory. Prefer /vast/$USER/data when available (user requested saving under vast/data).
# Fall back to /scratch/$USER/data if /vast isn't present or writable.
USER_NAME="${USER:-$(whoami)}"
if [ -d "/vast" ] && [ -w "/vast" ]; then
  BASE_DIR="/vast/${USER_NAME}/data"
else
  BASE_DIR="/scratch/${USER_NAME}/data"
fi
OUT_DIR="${BASE_DIR}/navtrain"
WGET="wget -q --show-progress --tries=3 --timeout=30"

# Concurrency tuning (sane defaults). We detect CPUs and limit jobs to avoid overwhelming I/O.
NPROC=$(nproc --all 2>/dev/null || echo 4)
# Use fewer concurrent downloads than cores; typical rule: cores/8 clamped between 2 and 16
MAX_DOWNLOAD_JOBS=$(( NPROC/8 ))
if [ "${MAX_DOWNLOAD_JOBS}" -lt 2 ]; then MAX_DOWNLOAD_JOBS=2; fi
if [ "${MAX_DOWNLOAD_JOBS}" -gt 16 ]; then MAX_DOWNLOAD_JOBS=16; fi
# Extraction can be slightly more parallel but keep conservative default
MAX_EXTRACT_JOBS=$(( MAX_DOWNLOAD_JOBS/2 ))
if [ "${MAX_EXTRACT_JOBS}" -lt 1 ]; then MAX_EXTRACT_JOBS=1; fi

# Check for helpers
PIGZ_PATH="$(which pigz 2>/dev/null || true)"
XARGS_PATH="$(which xargs 2>/dev/null || true)"

mkdir -p "${OUT_DIR}"
cd "${OUT_DIR}"
echo "[INFO] Working dir: $(pwd)"
echo "[INFO] Job: ${SLURM_JOB_NAME:-no-slurm}  ID: ${SLURM_JOB_ID:-n/a}"
echo "[INFO] Chosen BASE_DIR=${BASE_DIR}  nproc=${NPROC}  max_download_jobs=${MAX_DOWNLOAD_JOBS}  max_extract_jobs=${MAX_EXTRACT_JOBS}"
if [ -n "${PIGZ_PATH}" ]; then
  echo "[INFO] pigz found at ${PIGZ_PATH} - will use multi-threaded decompression where possible"
else
  echo "[INFO] pigz not found - using single-threaded tar decompression"
fi
# ---- 1) OpenScene metadata (NavSim logs) ----
META_TGZ="openscene_metadata_trainval.tgz"
META_URL="https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_metadata_trainval.tgz"
if [ ! -d "trainval_navsim_logs" ]; then
  if [ ! -f "${META_TGZ}" ]; then
    echo "[INFO] Downloading metadata tarball..."
    ${WGET} -O "${META_TGZ}" "${META_URL}"
  else
    echo "[INFO] Found existing ${META_TGZ}, reusing."
  fi
  echo "[INFO] Extracting metadata..."
  # Use pigz if available for faster decompression
  if [ -n "${PIGZ_PATH}" ]; then
    tar --use-compress-program="${PIGZ_PATH} -d -p ${MAX_EXTRACT_JOBS}" -xvf "${META_TGZ}"
  else
    tar -xzf "${META_TGZ}"
  fi
  rm -f "${META_TGZ}"
  # Move meta_datas into final location, then remove wrapper folder
  if [ -d "openscene-v1.1/meta_datas" ]; then
    mv openscene-v1.1/meta_datas trainval_navsim_logs
    rm -rf openscene-v1.1
  fi
else
  echo "[INFO] trainval_navsim_logs already present, skipping metadata step."
fi
# ---- 2) Sensor blobs output dir ----
# Prepare final folders
mkdir -p trainval_sensor_blobs/trainval

# Helper: single fetch+extract+merge job used when running sequentially or by worker
fetch_extract_and_merge_single () {
  local label="$1"   # "current" or "history"
  local idx="$2"
  local url="$3"
  local tgz="${label}_split_${idx}.tgz"
  local folder="${label}_split_${idx}"
  echo "[INFO] Processing ${label} split ${idx}"

  if [ ! -f "${tgz}" ]; then
    echo "[INFO]   Downloading ${tgz}"
    ${WGET} -O "${tgz}" "${url}"
  else
    echo "[INFO]   Found existing ${tgz}, reusing."
  fi

  echo "[INFO]   Extracting ${tgz}"
  if [ -n "${PIGZ_PATH}" ]; then
    # Use pigz for multi-threaded decompression
    tar --use-compress-program="${PIGZ_PATH} -d -p ${MAX_EXTRACT_JOBS}" -xvf "${tgz}"
  else
    tar -xzf "${tgz}"
  fi
  rm -f "${tgz}"

  # Rsync into final folder
  if [ -d "${folder}" ]; then
    echo "[INFO]   Merging ${folder} -> trainval_sensor_blobs/trainval"
    rsync -a "${folder}/" "trainval_sensor_blobs/trainval/"
    rm -rf "${folder}"
  else
    echo "[WARN]   Expected folder ${folder} not found after extract."
  fi
}

# Helper: run a list of commands in parallel using xargs if available, else background jobs with a simple semaphore
run_parallel_cmds() {
  local cmds_file="$1"
  local jobs="$2"
  if [ -n "${XARGS_PATH}" ]; then
    cat "${cmds_file}" | ${XARGS_PATH} -I CMD -P "${jobs}" bash -c CMD
  else
    # fallback: spawn background jobs, throttling via jobs -r
    while read -r cmd; do
      bash -c "$cmd" &
      # throttle
      while [ "$(jobs -r | wc -l)" -ge "${jobs}" ]; do
        sleep 1
      done
    done < "${cmds_file}"
    wait
  fi
}
# ---- 3) Download + merge CURRENT splits (1..4) ----
echo "[INFO] Preparing download list for CURRENT splits"
DOWNLOAD_CMDS="$(mktemp)"
for split in 1 2 3 4; do
  URL="https://s3.eu-central-1.amazonaws.com/avg-projects-2/navsim/navtrain_current_${split}.tgz"
  tgz="current_split_${split}.tgz"
  # skip if already merged folder present
  if [ -d "current_split_${split}" ] || [ -d "trainval_sensor_blobs/trainval" ] && ls trainval_sensor_blobs/trainval | grep -q "current" 2>/dev/null; then
    echo "[INFO] current split ${split} seems already present; fetch/merge may be skipped later."
  fi
  echo "fetch_extract_and_merge_single 'current' ${split} '${URL}'" >> "${DOWNLOAD_CMDS}"
done

echo "[INFO] Preparing download list for HISTORY splits"
for split in 1 2 3 4; do
  URL="https://s3.eu-central-1.amazonaws.com/avg-projects-2/navsim/navtrain_history_${split}.tgz"
  echo "fetch_extract_and_merge_single 'history' ${split} '${URL}'" >> "${DOWNLOAD_CMDS}"
done

echo "[INFO] Running download+extract+merge using up to ${MAX_DOWNLOAD_JOBS} parallel jobs"
run_parallel_cmds "${DOWNLOAD_CMDS}" "${MAX_DOWNLOAD_JOBS}"
rm -f "${DOWNLOAD_CMDS}"
# ---- 4) Download + merge HISTORY splits (1..4) ----
for split in 1 2 3 4; do
  URL="https://s3.eu-central-1.amazonaws.com/avg-projects-2/navsim/navtrain_history_${split}.tgz"
  echo "[INFO] Extracting file navtrain_history_${split}.tgz"
  fetch_and_merge "history" "${split}" "${URL}"
done
# ---- 5) Report sizes ----
echo "[INFO] Final layout:"
du -sh trainval_navsim_logs || true
du -sh trainval_sensor_blobs || true
echo "[INFO] Done."










