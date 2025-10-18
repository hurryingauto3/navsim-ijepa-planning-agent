#!/bin/bash
# =============================================================================
# Master Setup Script for Hydra-MDP Prerequisites
# 
# Downloads and caches everything needed before training:
# 1. PDM ground truth scores
# 2. Pretrained vision backbones
# 3. Metric caches for evaluation splits
# =============================================================================

set -e

echo "=========================================="
echo "Hydra-MDP Complete Setup Script"
echo "=========================================="
echo ""
echo "This will download and cache all prerequisites:"
echo "  1. PDM ground truth scores (for training)"
echo "  2. Pretrained DD3D backbone (for training)"
echo "  3. Metric caches (for evaluation)"
echo ""
echo "=========================================="
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if we're on Torch
if [[ $(hostname) != *"cs649"* ]]; then
    echo "Warning: Not on Torch (cs649)"
    echo "Some operations may fail or be slow"
    echo ""
fi

# Step 1: Download PDM ground truths
echo "=========================================="
echo "Step 1/3: Downloading PDM Ground Truths"
echo "=========================================="
echo ""
bash "${SCRIPT_DIR}/download_pdm_gt.sh"
echo ""

# Step 2: Download pretrained backbones
echo "=========================================="
echo "Step 2/3: Downloading Pretrained Backbones"
echo "=========================================="
echo ""
bash "${SCRIPT_DIR}/download_backbones.sh"
echo ""

# Step 3: Cache metrics (submit as jobs)
echo "=========================================="
echo "Step 3/3: Submitting Metric Caching Jobs"
echo "=========================================="
echo ""

if [[ $(hostname) == *"cs649"* ]]; then
    # We're on Torch, submit Slurm jobs
    echo "Submitting metric caching jobs for evaluation splits..."
    echo ""
    
    # Create logs directory
    mkdir -p /scratch/ah7072/navsim_workspace/exp/logs
    
    # Cache navmini (for quick testing)
    echo "  → Submitting cache job for navmini split..."
    JOB1=$(sbatch "${SCRIPT_DIR}/torch_cache_metrics.slurm" navmini | awk '{print $4}')
    echo "    Job ID: $JOB1"
    
    # Cache navtest (for final evaluation)
    echo "  → Submitting cache job for navtest split..."
    JOB2=$(sbatch "${SCRIPT_DIR}/torch_cache_metrics.slurm" navtest | awk '{print $4}')
    echo "    Job ID: $JOB2"
    
    echo ""
    echo "Metric caching jobs submitted!"
    echo "Monitor with: squeue -u $USER"
    echo "Check logs in: /scratch/ah7072/navsim_workspace/exp/logs/"
else
    echo "Not on Torch - skipping metric caching job submission"
    echo "You'll need to run metric caching manually later"
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Downloaded files:"
echo "  • PDM GT: ${OPENSCENE_DATA_ROOT}/traj_pdm_v2/"
echo "  • Backbone: ${OPENSCENE_DATA_ROOT}/models/dd3d_det_final.pth"
echo ""
if [[ $(hostname) == *"cs649"* ]]; then
    echo "Metric caching jobs are running in background."
    echo "Wait for them to complete before running evaluations."
    echo ""
    echo "Check status: squeue -u $USER"
    echo "Cache location: ${NAVSIM_EXP_ROOT}/metric_cache/"
    echo ""
fi
echo "You're now ready to:"
echo "  1. Run smoke test: sbatch scripts/torch_smoke_test.slurm"
echo "  2. Start training: ./scripts/launch_all_torch.sh"
echo ""
