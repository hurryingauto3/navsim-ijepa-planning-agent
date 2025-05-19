#!/bin/bash

# --- Configuration ---
# Agent name in Hydra config
AGENT_CONFIG_NAME="ijepa_planning_agent"
# Dataset split
SPLIT_NAME="trainval"
# Output directory for the cached data
CACHE_DIR="${NAVSIM_EXP_ROOT}/cache_${AGENT_CONFIG_NAME}_${SPLIT_NAME}"
# Number of Ray workers (adjust based on RAM, start low)
NUM_WORKERS=24
# Experiment name for logging
EXP_NAME="cache_${AGENT_CONFIG_NAME}_${SPLIT_NAME}_10pct"

# --- Ensure Directories Exist ---
mkdir -p "$(dirname "$CACHE_DIR")"

# --- Command ---
echo "Starting Dataset Caching for ${SPLIT_NAME}"
echo "Cache will be saved to: ${CACHE_DIR}"

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_dataset_caching.py \
    experiment_name=$EXP_NAME \
    agent=$AGENT_CONFIG_NAME \
    train_test_split=$SPLIT_NAME \
    cache_path=$CACHE_DIR \
    worker=ray_distributed \
    worker.threads_per_node=$NUM_WORKERS 

echo "Dataset Caching finished."