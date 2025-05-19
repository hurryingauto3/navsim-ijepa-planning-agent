#!/bin/bash

# --- Configuration ---
# Dataset split for metrics
SPLIT_NAME="test"
# Approx 10% of test scenes (~40000 * 0.1 - ADJUST IF NEEDED)
MAX_SCENES_LIMIT=4000
# Output directory for the metric cache
METRIC_CACHE_DIR="${NAVSIM_EXP_ROOT}/metric_cache_${SPLIT_NAME}_10pct"
# Experiment name for logging
EXP_NAME="metric_cache_${SPLIT_NAME}_10pct"

# --- Ensure Directories Exist ---
mkdir -p "$(dirname "$METRIC_CACHE_DIR")"

# --- Command ---
echo "Starting Metric Caching for ${SPLIT_NAME} (10% ~ ${MAX_SCENES_LIMIT} scenes)..."
echo "Metric cache will be saved to: ${METRIC_CACHE_DIR}"

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
    experiment_name=$EXP_NAME \
    train_test_split=$SPLIT_NAME \
    train_test_split.scene_filter.max_scenes=$MAX_SCENES_LIMIT \
    metric_cache_path=$METRIC_CACHE_DIR \
    # --- Add overrides for specific metrics or simulation if needed ---
    # e.g., +simulation_cfg=pdm_diffusion_low_rep # If caching planner metrics
    # force_metric_cache_computation=True # If available and needed

echo "Metric Caching finished."