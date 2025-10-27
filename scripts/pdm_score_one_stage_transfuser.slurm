#!/bin/bash
#SBATCH --job-name=eval_transfuser_ijepa_1stage
#SBATCH --partition=l40s_public
#SBATCH --account=torch_pr_68_tandon_advanced
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=400GB
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/ah7072/experiments/logs/output/eval_%j.out
#SBATCH --error=/scratch/ah7072/experiments/logs/error/eval_%j.err
#SBATCH --requeue

# =============================================================================
# TransFuser Extended One-Stage Evaluation
# =============================================================================
# Purpose: Evaluate TransFuser on navhard_one_stage (synthetic observations)
# Requires: GPU for TransFuser inference
# Expected: ~30-60 mins for navhard_one_stage (~6k scenarios)
# =============================================================================

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

# Agent Configuration
export TORCH=""
export AGENT="transfuser_agent"
export CHECKPOINT="/scratch/ah7072/experiments/exp_transfuser_agent_ijepa_100pct_20251026_211211/2025.10.26.21.12.49/lightning_logs/version_358532/checkpoints/epoch=19-step=6660.ckpt"

# Evaluation Settings
export EVAL_SPLIT="navtest"
export RUN_NAME="eval_${AGENT}_${EVAL_SPLIT}_$(date +%Y%m%d_%H%M%S)"

# Performance Tuning (GPU-based)
export NUM_WORKERS=8  # Ray workers for parallelization
export BATCH_SIZE=16   # Inference batch size

# -----------------------------------------------------------------------------
# PATH CONFIGURATION
# -----------------------------------------------------------------------------

export NAVSIM_DEVKIT_ROOT="/scratch/ah7072/navsim"
export OPENSCENE_DATA_ROOT="/scratch/ah7072/data/openscene"
export NUPLAN_MAPS_ROOT="/scratch/ah7072/data/maps"
export NAVSIM_EXP_ROOT="/scratch/ah7072/experiments"

# Output Paths
export OUTPUT_DIR="${NAVSIM_EXP_ROOT}/evaluations/${RUN_NAME}"
export METRIC_CACHE="${NAVSIM_EXP_ROOT}/metric_cache"

mkdir -p "${OUTPUT_DIR}"
mkdir -p /scratch/ah7072/experiments/logs/output
mkdir -p /scratch/ah7072/experiments/logs/error

# -----------------------------------------------------------------------------
# VALIDATION
# -----------------------------------------------------------------------------

echo "=============================================="
echo "TransFuser Extended Evaluation"
echo "=============================================="
echo "Job Information:"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Node: $SLURM_JOB_NODELIST"
echo "  Date: $(date)"
echo ""
echo "Configuration:"
echo "  Agent: ${AGENT}"
echo "  Checkpoint: ${CHECKPOINT}"
echo "  Split: ${EVAL_SPLIT}"
echo "  Output: ${OUTPUT_DIR}"
echo "=============================================="
echo ""

# Validate checkpoint
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    exit 1
fi
echo "✓ Checkpoint validated"

# Check synthetic data paths
if [ ! -d "$SYNTHETIC_SENSOR_PATH" ]; then
    echo "WARNING: Synthetic sensor path not found: $SYNTHETIC_SENSOR_PATH"
    echo "  Two-stage evaluation may fail"
fi

# -----------------------------------------------------------------------------
# ENVIRONMENT SETUP
# -----------------------------------------------------------------------------

cd "${NAVSIM_DEVKIT_ROOT}"

# Load conda
CONDA_ROOT="/scratch/$USER/miniconda3"
if [ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]; then
    source "${CONDA_ROOT}/etc/profile.d/conda.sh"
    conda activate navsim
else
    module purge || true
    module load anaconda3/2025.06 || true
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate navsim
fi

export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1

# Configure for GPU usage
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo ""
echo "Environment Check:"
echo "  Python: $(which python)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "  GPUs: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo ""

# -----------------------------------------------------------------------------
# RUN EVALUATION
# -----------------------------------------------------------------------------

echo "=============================================="
echo "Starting Extended Two-Stage Evaluation"
echo "Time: $(date)"
echo "=============================================="
echo ""

python ${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_one_stage.py \
    train_test_split="${EVAL_SPLIT}" \
    experiment_name="${RUN_NAME}" \
    traffic_agents=non_reactive \
    metric_cache_path="${METRIC_CACHE}" \
    output_dir="${OUTPUT_DIR}" \
    agent="${AGENT}" \
    agent.config.backbone="ijepa" \
    agent.checkpoint_path="'${CHECKPOINT}'" \
    worker="ray_distributed${TORCH}"

EVAL_EXIT_CODE=$?

# -----------------------------------------------------------------------------
# RESULTS
# -----------------------------------------------------------------------------

echo ""
echo "=============================================="
echo "EVALUATION COMPLETE"
echo "=============================================="
echo "Time: $(date)"
echo "Exit code: ${EVAL_EXIT_CODE}"
echo ""

if [ $EVAL_EXIT_CODE -eq 0 ]; then
    echo "✓ Status: SUCCESS"
    echo ""
    echo "Results: ${OUTPUT_DIR}"
    
    # Display results if available
    if [ -f "${OUTPUT_DIR}"/*.csv ]; then
        echo ""
        echo "PDM Score Summary:"
        python -c "
import pandas as pd
import glob
csv_file = glob.glob('${OUTPUT_DIR}/*.csv')[0]
df = pd.read_csv(csv_file)
print(f'  Scenarios: {len(df)}')
print(f'  PDMS: {df[\"pdm_score\"].mean():.4f}')
print(f'  NC: {df[\"no_collision\"].mean():.4f}')
print(f'  DAC: {df[\"drivable_area_compliance\"].mean():.4f}')
print(f'  EP: {df[\"ego_progress\"].mean():.4f}')
print(f'  TTC: {df[\"time_to_collision\"].mean():.4f}')
print(f'  C: {df[\"comfort\"].mean():.4f}')
" 2>/dev/null || echo "  (Results CSV processing failed)"
    fi
else
    echo "✗ Status: FAILED"
    echo ""
    echo "Check logs:"
    echo "  ${OUTPUT_DIR}"
    echo "  /scratch/ah7072/experiments/logs/error/eval_${SLURM_JOB_ID}.err"
fi

echo "=============================================="

exit $EVAL_EXIT_CODE