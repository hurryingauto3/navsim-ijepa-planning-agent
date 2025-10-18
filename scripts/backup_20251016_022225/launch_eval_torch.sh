#!/bin/bash
# =============================================================================
# Quick evaluation launcher for all three trained Hydra-MDP models
# Run this after training completes
# =============================================================================

set -e

# Check we're on Torch
if [[ $(hostname) != *"cs649"* ]]; then
    echo "Error: This script must be run on Torch (cs649)"
    exit 1
fi

echo "=============================================="
echo "Hydra-MDP Evaluation - Launch Script"
echo "=============================================="
echo ""

# Find the most recent checkpoints for each variant
echo "Searching for trained model checkpoints..."
echo ""

PDM_CKPT=$(find /scratch/ah7072/navsim_workspace/exp -name "last.ckpt" -path "*/hydra_pdm_v8192*/checkpoints/*" | sort -r | head -1)
W_CKPT=$(find /scratch/ah7072/navsim_workspace/exp -name "last.ckpt" -path "*/hydra_w_v8192_2*/checkpoints/*" | sort -r | head -1)
W_EP_CKPT=$(find /scratch/ah7072/navsim_workspace/exp -name "last.ckpt" -path "*/hydra_w_ep_v8192*/checkpoints/*" | sort -r | head -1)

# Check if checkpoints exist
if [[ -z "$PDM_CKPT" ]]; then
    echo "Warning: No checkpoint found for PDM-only model"
else
    echo "PDM-only checkpoint: $PDM_CKPT"
fi

if [[ -z "$W_CKPT" ]]; then
    echo "Warning: No checkpoint found for Weighted model"
else
    echo "Weighted checkpoint: $W_CKPT"
fi

if [[ -z "$W_EP_CKPT" ]]; then
    echo "Warning: No checkpoint found for Weighted+EP model"
else
    echo "Weighted+EP checkpoint: $W_EP_CKPT"
fi

echo ""
echo "=============================================="
echo "Submitting evaluation jobs..."
echo "=============================================="
echo ""

cd /scratch/$USER/GTRS/scripts

# Submit evaluation jobs
if [[ -n "$PDM_CKPT" ]]; then
    echo "Submitting PDM-only evaluation..."
    EVAL_JOB1=$(sbatch --export=CHECKPOINT_PATH="$PDM_CKPT",EXPERIMENT_NAME="pdm",CONFIG_NAME="hydra_mdp_v8192_pdm" torch_eval.slurm | awk '{print $4}')
    echo "  → Job ID: $EVAL_JOB1"
fi

if [[ -n "$W_CKPT" ]]; then
    echo "Submitting Weighted evaluation..."
    EVAL_JOB2=$(sbatch --export=CHECKPOINT_PATH="$W_CKPT",EXPERIMENT_NAME="w",CONFIG_NAME="hydra_mdp_v8192_w" torch_eval.slurm | awk '{print $4}')
    echo "  → Job ID: $EVAL_JOB2"
fi

if [[ -n "$W_EP_CKPT" ]]; then
    echo "Submitting Weighted+EP evaluation..."
    EVAL_JOB3=$(sbatch --export=CHECKPOINT_PATH="$W_EP_CKPT",EXPERIMENT_NAME="w_ep",CONFIG_NAME="hydra_mdp_v8192_w_ep" torch_eval.slurm | awk '{print $4}')
    echo "  → Job ID: $EVAL_JOB3"
fi

echo ""
echo "=============================================="
echo "All evaluation jobs submitted!"
echo "=============================================="
echo ""
echo "Monitor with: squeue -u $USER"
echo "Results will be in: /scratch/$USER/experiments/eval_*/"
echo ""
