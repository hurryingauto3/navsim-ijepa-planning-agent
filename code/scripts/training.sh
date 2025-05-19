#!/bin/bash

# --- Configuration ---
AGENT_CONFIG_NAME="ijepa_planning_agent"
SPLIT_NAME="trainval"
# Use the SAME cache directory created in the previous step
CACHE_DIR="${NAVSIM_EXP_ROOT}/cache_${AGENT_CONFIG_NAME}_${SPLIT_NAME}"
# Number of training epochs
MAX_EPOCHS=30 # Or your desired number
# Experiment name for training logs and checkpoints
EXP_NAME="train_${AGENT_CONFIG_NAME}_${SPLIT_NAME}"

# --- Ensure Cache Directory Exists ---
if [ ! -d "$CACHE_DIR" ]; then
    echo "ERROR: Cache directory $CACHE_DIR does not exist. Run dataset caching first."
    exit 1
fi

# --- Command ---
echo "Starting Training for ${SPLIT_NAME} .."
echo "Using cache from: ${CACHE_DIR}"

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
    experiment_name=$EXP_NAME \
    agent=$AGENT_CONFIG_NAME \
    train_test_split=$SPLIT_NAME \
    trainer.params.max_epochs=$MAX_EPOCHS \
    cache_path=$CACHE_DIR \
    use_cache_without_dataset=True \
    force_cache_computation=False \
    # --- Use Cache ---
    # --- Limit Training/Validation Batches ---
    # --- Other Trainer Params (Adjust as needed) ---
    # dataloader.params.batch_size=... # Ensure batch size is appropriate

# Note: Worker config is less relevant here if use_cache_without_dataset=True,
# as the main load is reading cache files, not running builders.
# Default dataloader workers should be sufficient.

echo "Training finished. Checkpoints in ${NAVSIM_EXP_ROOT}/${EXP_NAME}/..."

# --- Find the best checkpoint (example using 'last.ckpt') ---
# You might need more robust logic to find the best checkpoint based on validation metrics
MODEL_CKPT=$(find "${NAVSIM_EXP_ROOT}/${EXP_NAME}" -name "last.ckpt" | head -n 1)

echo "--------------------------------------------------"
echo "Training Complete!"
echo "Cache used: ${CACHE_DIR}"
echo "Checkpoints and logs saved in: ${NAVSIM_EXP_ROOT}/${EXP_NAME}"
if [ -n "$MODEL_CKPT" ]; then
    echo "Path to last saved model checkpoint:"
    echo $MODEL_CKPT
    # Save this path for the evaluation step
    export NAVSIM_TRAINED_MODEL_PATH=$MODEL_CKPT
else
    echo "WARNING: Could not find last.ckpt automatically."
fi
echo "--------------------------------------------------"