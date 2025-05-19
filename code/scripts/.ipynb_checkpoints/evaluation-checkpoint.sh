#!/bin/bash

# --- Configuration ---
AGENT_CONFIG_NAME="ijepa_planning_agent"
SPLIT_NAME="test"
# Approx 10% of test scenes (~40000 * 0.1 - ADJUST IF NEEDED)
MAX_SCENES_LIMIT=4000
# Path to the trained MLP weights file (.pth)
# This should be the file containing ONLY the mlp.state_dict()
# If NAVSIM_TRAINED_MODEL_PATH was set by training script, use it. Otherwise, set manually.
# Example: Assume training saved MLP weights to a .pth file inside the checkpoint dir
TRAINED_MLP_WEIGHTS="${NAVSIM_TRAINED_MODEL_PATH}/mlp_weights.pth" # ADJUST THIS PATH based on how training saves weights
# Check if the variable is set, otherwise prompt
if [ -z "$NAVSIM_TRAINED_MODEL_PATH" ] || [ ! -f "$TRAINED_MLP_WEIGHTS" ]; then
    echo "NAVSIM_TRAINED_MODEL_PATH not set or weights file not found at expected location."
    echo "Please set the TRAINED_MLP_WEIGHTS variable in the script manually."
    # Example Manual Path:
    # TRAINED_MLP_WEIGHTS="/path/to/your/exp/train_ijepa.../checkpoints/mlp_weights_epoch_X.pth"
    read -p "Enter path to trained MLP weights (.pth file): " TRAINED_MLP_WEIGHTS
    if [ ! -f "$TRAINED_MLP_WEIGHTS" ]; then
        echo "ERROR: File not found: $TRAINED_MLP_WEIGHTS"
        exit 1
    fi
fi

# Optional: Path to pre-computed metric cache (must match the one generated above)
# METRIC_CACHE_DIR="${NAVSIM_EXP_ROOT}/metric_cache_${SPLIT_NAME}_10pct"
# Add metric_cache_path=$METRIC_CACHE_DIR to command if using cached metrics.

# Experiment name for evaluation results
EXP_NAME="eval_${AGENT_CONFIG_NAME}_${SPLIT_NAME}_10pct_single_stage"

# --- Command ---
echo "Starting Single-Stage Evaluation for ${SPLIT_NAME} (10% ~ ${MAX_SCENES_LIMIT} scenes)..."
echo "Using agent config: ${AGENT_CONFIG_NAME}"
echo "Using MLP weights: ${TRAINED_MLP_WEIGHTS}"

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_one_stage.py \
    experiment_name=$EXP_NAME \
    agent=$AGENT_CONFIG_NAME \
    train_test_split=$SPLIT_NAME \
    train_test_split.scene_filter.max_scenes=$MAX_SCENES_LIMIT \
    # --- Provide path to trained MLP weights ---
    agent.mlp_weights_path=$TRAINED_MLP_WEIGHTS \
    # --- Optional: Use metric cache ---
    # metric_cache_path=$METRIC_CACHE_DIR \
    # --- Other options ---
    traffic_agents_policy=non_reactive # Or reactive

echo "Evaluation finished. Results in ${NAVSIM_EXP_ROOT}/${EXP_NAME}/..."