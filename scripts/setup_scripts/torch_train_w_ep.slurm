#!/bin/bash
#SBATCH --job-name=hydra_w_ep
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="h200|l40s"
#SBATCH --time=48:00:00
#SBATCH --comment="preemption=yes;requeue=yes"
#SBATCH --output=/scratch/ah7072/navsim_workspace/exp/logs/hydra_w_ep_%j.out
#SBATCH --error=/scratch/ah7072/navsim_workspace/exp/logs/hydra_w_ep_%j.err

# =============================================================================
# Hydra-MDP Weighted Confidence + Continuous EP Training (Table 1: Score 86.5)
# This is the best-performing variant with continuous EP regression
# Expected: NC=98.3, DAC=96.0, EP=78.7, TTC=94.6, C=100, Score=86.5
# =============================================================================

# Environment setup
export NAVSIM_DEVKIT_ROOT="/scratch/ah7072/navsim_workspace/navsim"
export OPENSCENE_DATA_ROOT="/scratch/ah7072/navsim_workspace/dataset"
export NUPLAN_MAPS_ROOT="/scratch/ah7072/navsim_workspace/dataset/maps"
export NAVSIM_EXP_ROOT="/scratch/ah7072/navsim_workspace/exp"
export APPTAINER_BINDPATH=/scratch,/state/partition1,/mnt,/share/apps

# Experiment metadata
EXPERIMENT_NAME="hydra_w_ep_v8192_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${NAVSIM_EXP_ROOT}/${EXPERIMENT_NAME}"
mkdir -p "${LOG_DIR}/notes"

# Log experiment details
cat > "${LOG_DIR}/notes/experiment_info.txt" <<EOF
Experiment: Hydra-MDP Weighted Confidence + Continuous EP Training
Config: hydra_mdp_v8192_w_ep
Target Paper Results: Score 86.5 (NC=98.3, DAC=96.0, EP=78.7, TTC=94.6, C=100)
Training Mode: Multi-target Hydra distillation + weighted confidence + continuous EP
Vocab Size: 16384
Key Features:
  - 5 separate prediction heads (NC, DAC, TTC, C, EP)
  - Weighted confidence inference
  - Continuous EP training (regression instead of classification)
  - Trajectory imitation weight: 1.0
  - Progress weight: 2.0
Started: $(date)
Slurm Job ID: $SLURM_JOB_ID
EOF

# Navigate to repo
cd "${NAVSIM_DEVKIT_ROOT}"

# Activate conda environment
if command -v module &> /dev/null; then
    module load anaconda3/2025.06 2>/dev/null || true
fi

if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate navsim 2>/dev/null || conda activate conda_gtrs 2>/dev/null || {
        echo "ERROR: No conda environment found!"; exit 1
    }
else
    echo "ERROR: conda not found!"; exit 1
fi

# Training command
python navsim/planning/script/run_training_dense.py \
    agent=hydra_mdp_v8192_w_ep \
    experiment_name="${EXPERIMENT_NAME}" \
    experiment_root="${NAVSIM_EXP_ROOT}" \
    trainer.max_epochs=20 \
    trainer.devices=1 \
    trainer.accelerator=gpu \
    data.batch_size=32 \
    data.num_workers=4 \
    +trainer.val_check_interval=0.5 \
    +trainer.log_every_n_steps=50

echo "Training completed at $(date)" >> "${LOG_DIR}/notes/experiment_info.txt"
