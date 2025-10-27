#!/bin/bash
# Master script to submit all NAVSIM download jobs in sequence
# Submits jobs with dependencies so they run in order

set -e

echo "=========================================="
echo "NAVSIM Data Download - Master Script"
echo "Submitting download jobs to Torch cluster"
echo "=========================================="

# Create logs directory
mkdir -p /scratch/ah7072/data/logs

# Submit maps download (runs immediately)
echo "Submitting: Maps download (Job 1/3)..."
JOB1=$(sbatch --parsable /scratch/ah7072/data/01_download_maps.slurm)
echo "  Job ID: $JOB1"

# Submit mini download (depends on maps completing)
echo "Submitting: Mini dataset download (Job 2/3)..."
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 /scratch/ah7072/data/02_download_mini.slurm)
echo "  Job ID: $JOB2 (depends on $JOB1)"

# Submit navtrain download (depends on mini completing)
echo "Submitting: Navtrain dataset download (Job 3/3)..."
JOB3=$(sbatch --parsable --dependency=afterok:$JOB2 /scratch/ah7072/data/03_download_navtrain.slurm)
echo "  Job ID: $JOB3 (depends on $JOB2)"

echo ""
echo "=========================================="
echo "All jobs submitted successfully!"
echo "=========================================="
echo ""
echo "Job chain:"
echo "  $JOB1: Maps (~9GB, ~1 hour)"
echo "  $JOB2: Mini (~152GB, ~3-6 hours) - starts after maps complete"
echo "  $JOB3: Navtrain (~459GB, ~12-24 hours) - starts after mini completes"
echo ""
echo "Total time estimate: 16-31 hours"
echo "Total storage: ~620GB"
echo "Total files: ~4M"
echo ""
echo "Monitor progress:"
echo "  squeue -u $USER"
echo "  tail -f /scratch/ah7072/data/logs/download_*.out"
echo ""
echo "Check storage usage:"
echo "  watch -n 60 'du -sh /scratch/ah7072/data/*'"
echo ""
echo "=========================================="
