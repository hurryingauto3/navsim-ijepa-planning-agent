#!/bin/bash
#SBATCH --job-name=pdm_2_transfuser
#SBATCH --partition=l40s_public
#SBATCH --account=torch_pr_68_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:4
#SBATCH --mem=200GB
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/ah7072/experiments/logs/output/eval_%j.out
#SBATCH --error=/scratch/ah7072/experiments/logs/error/eval_%j.err
#SBATCH --requeue

# =============================================================================
# PDM SCORE EVALUATION (CPU-Only with Ray Parallelization)
# =============================================================================
# Note: Constant Velocity agent is rule-based, no GPU needed
# Purpose: Evaluate CV agent on NAVSIM navtest using PDM Score
# Ray parallelizes across CPUs for fast evaluation (~1.5-2 hours)
# Usage:   sbatch scripts/pdm_score.slurm
# Docs:    See GTRS/docs/evaluation.md for metric details
# =============================================================================

# -----------------------------------------------------------------------------
# CONFIGURATION SECTION - Modify these for each evaluation
# -----------------------------------------------------------------------------

# Agent Configuration
export TORCH=""
export AGENT="transfuser_agent"          # Agent config name (from navsim/planning/script/config/common/agent/)
export CHECKPOINT="/scratch/ah7072/experiments/exp_a4_transfuser_100pct_20251021_005545/2025.10.21.00.56.32/lightning_logs/version_324588/checkpoints/epoch=19-step=6660.ckpt"


# Evaluation Settings
export EVAL_SPLIT="navhard_two_stage"                   # Data split: 'navmini' (~100 samples, quick), 'navtest' (full eval)
export RUN_NAME="eval_${AGENT}_${EVAL_SPLIT}_$(date +%Y%m%d_%H%M%S)"  # Name for this evaluation run

# Performance Tuning (CPU-Only)
export BATCH_SIZE=32                          # Batch size for evaluation (CPU inference)
export NUM_WORKERS=8                          # DataLoader workers (adjust based on CPU count)
export PREFETCH_FACTOR=2                      # Number of batches to prefetch per worker
export RAY_CPUS=8                             # Ray worker CPUs (leave 2 for main process)

# Validate CHECKPOINT
if [ "$AGENT" = "constant_velocity_agent" ]; then
    export CHECKPOINT=""
else
    # Validate checkpoint exists
    if [ -z "$CHECKPOINT" ]; then
        echo "ERROR: CHECKPOINT variable is not set"
        echo "Please set CHECKPOINT in the configuration section"
        exit 1
    fi

    if [ ! -f "$CHECKPOINT" ]; then
        echo "ERROR: Checkpoint file not found: $CHECKPOINT"
        exit 1
    fi
    echo "✓ Checkpoint validated: $CHECKPOINT"
    echo ""
fi

# -----------------------------------------------------------------------------
# PATH CONFIGURATION - Standard paths (rarely need changes)
# -----------------------------------------------------------------------------

# Project Paths
export NAVSIM_DEVKIT_ROOT="/scratch/ah7072/navsim"
export OPENSCENE_DATA_ROOT="/scratch/ah7072/data"
export NUPLAN_MAPS_ROOT="/scratch/ah7072/data/maps"
export NAVSIM_EXP_ROOT="/scratch/ah7072/experiments"

# Output Paths
export OUTPUT_DIR="${NAVSIM_EXP_ROOT}/evaluations/${RUN_NAME}"
export METRIC_CACHE="${NAVSIM_EXP_ROOT}/metric_cache"

# Synthetic Data Paths (for NavHard / augmented scenarios)
export SYNTHETIC_SENSOR_PATH="${OPENSCENE_DATA_ROOT}/navhard_two_stage/sensor_blobs"
export SYNTHETIC_SCENES_PATH="${OPENSCENE_DATA_ROOT}/navhard_two_stage/synthetic_scene_pickles"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# -----------------------------------------------------------------------------
# JOB INFO & VALIDATION
# -----------------------------------------------------------------------------

echo "=============================================="
echo "PDM SCORE EVALUATION"
echo "=============================================="
echo "Job Information:"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Node: $SLURM_JOB_NODELIST"
echo "  Date: $(date)"
echo "  Run Name: ${RUN_NAME}"
echo ""
echo "Evaluation Configuration:"
echo "  Agent: ${AGENT}"
echo "  Checkpoint: ${CHECKPOINT}"
echo "  Split: ${EVAL_SPLIT}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  Workers: ${NUM_WORKERS}"
echo ""
echo "Output:"
echo "  Directory: ${OUTPUT_DIR}"
echo "  Metric Cache: ${METRIC_CACHE}"
echo "=============================================="
echo ""

# -----------------------------------------------------------------------------
# ENVIRONMENT SETUP
# -----------------------------------------------------------------------------

cd "${NAVSIM_DEVKIT_ROOT}"

# Activate Conda Environment
# Prefer local Miniconda in /scratch/$USER, fallback to system module
export CONDA_ROOT="/scratch/$USER/miniconda3"

if [ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]; then
    # Use conda installed in scratch
    source "${CONDA_ROOT}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ROOT}/envs/navsim"
    echo "✓ Using local Miniconda environment"
else
    # Fallback to system module
    module purge || true
    module load anaconda3/2025.06 || true
    if command -v conda &> /dev/null; then
        source $(conda info --base)/etc/profile.d/conda.sh || true
        conda activate navsim || source activate navsim || true
        echo "✓ Using system Anaconda environment"
    else
        echo "ERROR: conda not found; evaluation will fail"
        exit 1
    fi
fi

# Set Python path and error reporting
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1

# Ray configuration for CPU parallelization
export RAY_NUM_CPUS=${RAY_CPUS}
export RAY_OBJECT_STORE_MEMORY=$((128 * 1024 * 1024 * 1024 / 2))  # 64GB for object store
export OMP_NUM_THREADS=1  # Prevent thread oversubscription

# Verify environment
echo ""
echo "Environment Check:"
echo "  Python: $(which python)"
echo "  Python Version: $(python --version)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  Ray version: $(python -c 'import ray; print(ray.__version__)' 2>/dev/null || echo 'Not installed')"
echo ""
echo "Resource Allocation:"
echo "  CPUs available: ${SLURM_CPUS_PER_TASK}"
echo "  Ray CPUs: ${RAY_CPUS}"
echo "  Memory: ${SLURM_MEM_PER_NODE}MB"
echo "  GPUs available: ${SLURM_GPUS_ON_NODE}"
echo ""

# -----------------------------------------------------------------------------
# RUN EVALUATION
# -----------------------------------------------------------------------------

echo "=============================================="
echo "Starting PDM Score Evaluation"
echo "Time: $(date)"
echo "=============================================="
echo ""


# Run evaluation (two-stage non-reactive for navhard_two_stage)
python ${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score.py \
    train_test_split=${EVAL_SPLIT} \
    experiment_name=${RUN_NAME} \
    metric_cache_path="${METRIC_CACHE}" \
    synthetic_sensor_path=${SYNTHETIC_SENSOR_PATH} \
    synthetic_scenes_path=${SYNTHETIC_SCENES_PATH} \
    output_dir="${OUTPUT_DIR}" \
    agent=${AGENT} \
    agent.checkpoint_path="'${CHECKPOINT}'" \
    worker="ray_distributed${TORCH}"


EVAL_EXIT_CODE=$?

# -----------------------------------------------------------------------------
# RESULTS & SUMMARY
# -----------------------------------------------------------------------------

echo ""
echo "=============================================="
echo "EVALUATION COMPLETE"
echo "=============================================="
echo "=============================================="
echo "Time: $(date)"
echo ""

if [ $EVAL_EXIT_CODE -eq 0 ]; then
    echo "✓ Status: SUCCESS"
    echo ""
    echo "Results Location:"
    echo "  ${OUTPUT_DIR}"
    echo ""
    echo "View Summary:"
    echo "  cat ${OUTPUT_DIR}/metrics_summary.json"
    echo ""
    
    # Display quick metrics if available
    if [ -f "${OUTPUT_DIR}/metrics_summary.json" ]; then
        echo "Quick Metrics Preview:"
        python -c "import json; m=json.load(open('${OUTPUT_DIR}/metrics_summary.json')); print(f\"  PDM Score: {m.get('pdm_score', 'N/A')}\"); print(f\"  No Collision: {m.get('no_collision', 'N/A')}\"); print(f\"  Drivable Area: {m.get('drivable_area_compliance', 'N/A')}\"); print(f\"  Ego Progress: {m.get('driving_direction_compliance', 'N/A')}\")" 2>/dev/null || echo "  (Run 'cat ${OUTPUT_DIR}/metrics_summary.json' to view metrics)"
        echo ""
    fi
    
    # Next steps based on split
    if [ "$EVAL_SPLIT" = "navmini" ]; then
        echo "Next Steps:"
        echo "  This was a QUICK TEST on navmini (~100 samples)"
        echo "  For full evaluation, modify EVAL_SPLIT='navtest' and increase time limit"
    fi
    
else
    echo "✗ Status: FAILED (exit code: $EVAL_EXIT_CODE)"
    echo ""
    echo "Debugging:"
    echo "  Error log: /scratch/ah7072/experiments/logs/error/eval_${SLURM_JOB_ID}.err"
    echo "  Output log: /scratch/ah7072/experiments/logs/output/eval_${SLURM_JOB_ID}.out"
fi

echo "=============================================="
echo ""

exit $EVAL_EXIT_CODE