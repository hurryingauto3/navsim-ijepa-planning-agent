#!/bin/bash
# Sync Greene work to local machine
# Usage: Run this FROM YOUR LOCAL MACHINE, not on Greene
# ./sync_to_local.sh

# Configuration - UPDATE THESE
GREENE_USER="ah7072"
GREENE_HOST="greene.hpc.nyu.edu"
LOCAL_BACKUP_DIR="$HOME/thesis_backup/greene_$(date +%Y%m%d)"

# Create local backup directory
mkdir -p "$LOCAL_BACKUP_DIR"

echo "Syncing from Greene to: $LOCAL_BACKUP_DIR"

# Sync scripts
rsync -avz --progress \
  ${GREENE_USER}@${GREENE_HOST}:/scratch/ah7072/scripts/ \
  "$LOCAL_BACKUP_DIR/scripts/"

# Sync GTRS code changes (exclude data and large files)
rsync -avz --progress \
  --exclude='*.pyc' \
  --exclude='__pycache__' \
  --exclude='*.egg-info' \
  --exclude='.git' \
  --exclude='traj_final' \
  ${GREENE_USER}@${GREENE_HOST}:/scratch/ah7072/GTRS/navsim/ \
  "$LOCAL_BACKUP_DIR/GTRS/navsim/"

# Sync experiment configs and notes (not checkpoints)
rsync -avz --progress \
  --exclude='checkpoints' \
  --exclude='*.ckpt' \
  --exclude='wandb' \
  ${GREENE_USER}@${GREENE_HOST}:/scratch/ah7072/experiments/ \
  "$LOCAL_BACKUP_DIR/experiments/"

# Sync summaries and documentation
rsync -avz --progress \
  ${GREENE_USER}@${GREENE_HOST}:/scratch/ah7072/summaries/ \
  "$LOCAL_BACKUP_DIR/summaries/"

echo "✅ Sync complete! Files saved to: $LOCAL_BACKUP_DIR"
echo "💡 Tip: Add this to a weekly cron job on your local machine"
