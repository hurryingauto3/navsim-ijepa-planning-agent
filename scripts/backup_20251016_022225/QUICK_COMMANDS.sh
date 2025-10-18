#!/bin/bash
# Quick reference commands for Hydra-MDP experiments on Torch

# ============================================
# Setup (run once)
# ============================================
mkdir -p /scratch/ah7072/navsim_workspace/exp/logs

# ============================================
# Access Torch from Greene
# ============================================
# ssh greene.hpc.nyu.edu
# ssh cs649

# ============================================
# Run smoke test (always do this first!)
# ============================================
cd /scratch/ah7072/GTRS/scripts
sbatch torch_smoke_test.slurm

# ============================================
# Launch individual experiments
# ============================================
# PDM-only baseline (Score 80.2)
sbatch torch_train_pdm.slurm

# Weighted confidence (Score 85.7)
sbatch torch_train_w.slurm

# Weighted + continuous EP (Score 86.5) - BEST
sbatch torch_train_w_ep.slurm

# ============================================
# Launch all three at once
# ============================================
./launch_all_torch.sh

# ============================================
# Monitor jobs
# ============================================
# Check queue
squeue -u ah7072

# Watch specific job logs
tail -f /scratch/ah7072/navsim_workspace/exp/logs/hydra_pdm_*.out
tail -f /scratch/ah7072/navsim_workspace/exp/logs/hydra_w_*.out
tail -f /scratch/ah7072/navsim_workspace/exp/logs/hydra_w_ep_*.out

# Cancel job
scancel <JOB_ID>

# ============================================
# Find checkpoints after training
# ============================================
find /scratch/ah7072/navsim_workspace/exp -name "last.ckpt" -path "*/hydra_*/checkpoints/*"

# ============================================
# Evaluate specific checkpoint
# ============================================
# Example:
sbatch --export=CHECKPOINT_PATH="/scratch/ah7072/navsim_workspace/exp/hydra_w_ep_v8192_20251014/checkpoints/last.ckpt",EXPERIMENT_NAME="w_ep_eval",CONFIG_NAME="hydra_mdp_v8192_w_ep" torch_eval.slurm

# ============================================
# Evaluate all trained models at once
# ============================================
./launch_eval_torch.sh

# ============================================
# Check results
# ============================================
# Find evaluation results
find /scratch/ah7072/navsim_workspace/exp -name "pdm_score.json" -type f

# View PDM scores
cat /scratch/ah7072/navsim_workspace/exp/eval_*/pdm_score.json | grep -E '"score"|"no_collision"|"drivable_area"|"ego_progress"'

# ============================================
# Transfer results back to Greene for analysis
# ============================================
# From Torch, copy to Greene
scp -rp /scratch/ah7072/navsim_workspace/exp/eval_* dtn-1:/scratch/ah7072/hydra_results/

# ============================================
# Useful debugging commands
# ============================================
# Check GPU availability
sinfo -p h200,l40s -O partition,available,nodes,gres

# Check your job details
scontrol show job <JOB_ID>

# View full job output
cat /scratch/ah7072/navsim_workspace/exp/logs/hydra_pdm_<JOB_ID>.out
cat /scratch/ah7072/navsim_workspace/exp/logs/hydra_pdm_<JOB_ID>.err

# Check disk usage
du -sh /scratch/ah7072/navsim_workspace/exp/*
