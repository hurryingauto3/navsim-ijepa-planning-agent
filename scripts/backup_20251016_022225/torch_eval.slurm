#!/bin/bash
#SBATCH --job-name=eval_hydra
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="h200|l40s"
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/ah7072/navsim_workspace/exp/logs/eval_%j.out
#SBATCH --error=/scratch/ah7072/navsim_workspace/exp/logs/eval_%j.err

# =============================================================================
# Evaluation script for Hydra-MDP trained models
# Usage: sbatch torch_eval.slurm <checkpoint_path> <experiment_name> <config_name>
# Example: sbatch torch_eval.slurm /scratch/$USER/experiments/hydra_pdm_v8192_20251013/checkpoints/last.ckpt hydra_pdm_eval hydra_mdp_v8192_pdm
# =============================================================================

# Environment setup
export NAVSIM_DEVKIT_ROOT="/scratch/ah7072/navsim_workspace/navsim"
export OPENSCENE_DATA_ROOT="/scratch/ah7072/navsim_workspace/dataset"
export NUPLAN_MAPS_ROOT="/scratch/ah7072/navsim_workspace/dataset/maps"
export NAVSIM_EXP_ROOT="/scratch/ah7072/navsim_workspace/exp"
export APPTAINER_BINDPATH=/scratch,/state/partition1,/mnt,/share/apps

# Parse arguments (passed via --export or environment)
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$1}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-$2}"
CONFIG_NAME="${CONFIG_NAME:-$3}"
EVAL_SPLIT="${EVAL_SPLIT:-navtest}"  # Default to navtest for final results

if [[ -z "$CHECKPOINT_PATH" ]] || [[ -z "$EXPERIMENT_NAME" ]] || [[ -z "$CONFIG_NAME" ]]; then
    echo "Error: Missing required arguments"
    echo "Usage: sbatch torch_eval.slurm <checkpoint_path> <experiment_name> <config_name>"
    exit 1
fi

echo "=============================================="
echo "Evaluating Hydra-MDP Model"
echo "=============================================="
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Config: $CONFIG_NAME"
echo "Split: $EVAL_SPLIT"
echo "=============================================="

# Output directory
EVAL_DIR="${NAVSIM_EXP_ROOT}/eval_${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${EVAL_DIR}"

# Navigate to repo
cd "${NAVSIM_DEVKIT_ROOT}"

# Use Apptainer container
CONTAINER_IMAGE="/share/apps/images/pytorch-24.09.sif"

# Run evaluation
/share/apps/apptainer/bin/singularity exec --nv \
  ${CONTAINER_IMAGE} \
  python navsim/planning/script/run_pdm_score.py \
    agent=${CONFIG_NAME} \
    checkpoint_path=${CHECKPOINT_PATH} \
    split=${EVAL_SPLIT} \
    output_dir=${EVAL_DIR}

echo ""
echo "=============================================="
echo "Evaluation complete!"
echo "Results saved to: ${EVAL_DIR}"
echo "=============================================="
