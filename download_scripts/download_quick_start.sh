#!/bin/bash
# Quick start script - Downloads maps + mini only
# Use this to test your workflow before committing to full download

set -e

echo "=========================================="
echo "NAVSIM Quick Start - Maps + Mini Only"
echo "=========================================="

# Create logs directory
mkdir -p /scratch/ah7072/data/logs

# Submit maps download
echo "Submitting: Maps download..."
JOB1=$(sbatch --parsable /scratch/ah7072/data/01_download_maps.slurm)
echo "  Job ID: $JOB1"

# Submit mini download (depends on maps)
echo "Submitting: Mini dataset download..."
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 /scratch/ah7072/data/02_download_mini.slurm)
echo "  Job ID: $JOB2 (depends on $JOB1)"

echo ""
echo "=========================================="
echo "Quick start jobs submitted!"
echo "=========================================="
echo ""
echo "This will download:"
echo "  - Maps: ~9GB (~1 hour)"
echo "  - Mini dataset: ~152GB (~3-6 hours)"
echo ""
echo "Total: ~161GB, ~1M files, ~4-7 hours"
echo ""
echo "After completion, you can:"
echo "  1. Test training on navmini split"
echo "  2. If successful, run: sbatch 03_download_navtrain.slurm"
echo ""
echo "Monitor: squeue -u $USER"
echo "=========================================="
