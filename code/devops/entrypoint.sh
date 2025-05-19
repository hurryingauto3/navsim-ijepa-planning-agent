#!/bin/bash
set -e

DATASET_DIR="/navsim_workspace/dataset"

if [ ! -d "$DATASET_DIR" ] || [ -z "$(ls -A "$DATASET_DIR")" ]; then
    echo "[INFO] Dataset not found or empty at $DATASET_DIR. Downloading..."
    mkdir -p "$DATASET_DIR"
    cd "$DATASET_DIR"
    /navsim_workspace/navsim/download/download_maps.sh
    /navsim_workspace/navsim/download/download_mini.sh
    /navsim_workspace/navsim/download/download_private_test_e2e.sh
    /navsim_workspace/navsim/download/download_warmup_synthetic_scenes.sh
    /navsim_workspace/navsim/download/download_test.sh
    /navsim_workspace/navsim/download/download_navtrain.sh
    # /navsim_workspace/navsim/download/download_trainval.sh
else
    echo "[INFO] Dataset already exists at $DATASET_DIR. Skipping download."
fi

echo "[INFO] Launching Jupyter Notebook on :8888 and VS Code Server on :8080..."

# Start Jupyter Notebook in the background
jupyter notebook --notebook-dir=/navsim_workspace --ip=0.0.0.0 --allow-root --no-browser &

# Start VS Code Server (foreground)
exec code-server /navsim_workspace