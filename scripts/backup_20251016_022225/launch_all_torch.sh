#!/bin/bash
# =============================================================================
# Master script to launch all three Hydra-MDP ablation experiments on Torch
# Run from Greene: ssh greene.hpc.nyu.edu, then ssh cs649, then run this script
# =============================================================================

set -e  # Exit on error

# Check we're on Torch
if [[ $(hostname) != *"cs649"* ]]; then
    echo "Error: This script must be run on Torch (cs649)"
    echo "From Greene, run: ssh cs649"
    exit 1
fi

# Create logs directory
mkdir -p /scratch/ah7072/navsim_workspace/exp/logs

echo "=============================================="
echo "Hydra-MDP Ablation Study - Launch Script"
echo "=============================================="
echo ""
echo "This will submit 3 experiments to recreate Table 1 results:"
echo ""
echo "1. PDM-only (baseline)           -> Expected Score: 80.2"
echo "2. Weighted Confidence           -> Expected Score: 85.7"
echo "3. Weighted + Continuous EP      -> Expected Score: 86.5"
echo ""
echo "=============================================="
echo ""

# Navigate to scripts directory
cd /scratch/ah7072/GTRS/scripts

# Submit jobs sequentially (can also run in parallel if resources allow)
echo "Submitting Job 1: PDM-only Training..."
JOB1=$(sbatch torch_train_pdm.slurm | awk '{print $4}')
echo "  → Submitted with Job ID: $JOB1"
echo ""

echo "Submitting Job 2: Weighted Confidence Training..."
JOB2=$(sbatch torch_train_w.slurm | awk '{print $4}')
echo "  → Submitted with Job ID: $JOB2"
echo ""

echo "Submitting Job 3: Weighted + Continuous EP Training..."
JOB3=$(sbatch torch_train_w_ep.slurm | awk '{print $4}')
echo "  → Submitted with Job ID: $JOB3"
echo ""

echo "=============================================="
echo "All jobs submitted successfully!"
echo "=============================================="
echo ""
echo "Job IDs:"
echo "  PDM-only:        $JOB1"
echo "  Weighted:        $JOB2"
echo "  Weighted+EP:     $JOB3"
echo ""
echo "Monitor jobs with:"
echo "  squeue -u $USER"
echo ""
echo "Check logs at:"
echo "  /scratch/$USER/experiments/logs/"
echo ""
echo "Experiment outputs will be saved to:"
echo "  /scratch/$USER/experiments/hydra_*_v8192_*/"
echo ""
echo "=============================================="
