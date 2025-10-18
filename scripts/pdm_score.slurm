#!/bin/bash
#SBATCH --job-name=pdm_eval_baseline
#SBATCH --partition=l40s_public
#SBATCH --account=torch_pr_68_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=128GB
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/ah7072/navsim_workspace/exp/logs/eval_%j.out
#SBATCH --error=/scratch/ah7072/navsim_workspace/exp/logs/eval_%j.err

# =============================================================================
# PDM Score Evaluation: Baseline Hydra-MDP Model (Epoch 39)
# Single GPU evaluation on navtest split
# =============================================================================

echo "=============================================="
echo "PDM Score Evaluation - Baseline Model"
echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "=============================================="

# Environment setup
export NAVSIM_DEVKIT_ROOT="/scratch/ah7072/GTRS"
export OPENSCENE_DATA_ROOT="/scratch/ah7072/data/openscene"
export NUPLAN_MAPS_ROOT="/scratch/ah7072/data/maps"
export NAVSIM_EXP_ROOT="/scratch/ah7072/experiments"
export DP_PREDS="none"

# Checkpoint to evaluate (best from 40-epoch training)
CHECKPOINT="/scratch/ah7072/experiments/hydra_plus_16384_weighted_ep_ckpt/epoch=39-step=9680.ckpt"
AGENT="hydra_mdp_v8192_w_ep"

# Metric cache for faster evaluation
METRIC_CACHE="/scratch/ah7072/navsim_workspace/exp/metric_cache"

# Output directory
OUTPUT_DIR="/scratch/ah7072/experiments/eval_baseline_epoch39_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${OUTPUT_DIR}"

echo ""
echo "Evaluation Configuration:"
echo "  Agent: ${AGENT}"
echo "  Checkpoint: ${CHECKPOINT}"
echo "  Output Directory: ${OUTPUT_DIR}"
echo "  Split: navtest"
echo "  Metric Cache: ${METRIC_CACHE}"
echo ""

cd "${NAVSIM_DEVKIT_ROOT}"

# Load conda environment: prefer Miniconda in /scratch/$USER if available
CONDA_ROOT="/scratch/$USER/miniconda3"
if [ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]; then
    # Use conda installed in scratch - activate with full path to avoid issues
    source "${CONDA_ROOT}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ROOT}/envs/navsim"
else
    module purge || true
    module load anaconda3/2025.06 || true
    if command -v conda &> /dev/null; then
        source $(conda info --base)/etc/profile.d/conda.sh || true
        conda activate navsim || source activate navsim || true
    else
        echo "WARNING: conda not found; training may fail"
    fi
fi

# Ensure Python can import the local 'navsim' package (repo root on PYTHONPATH)
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}:${PYTHONPATH:-}"

echo "Environment check:"
echo "  Python: $(which python)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# Verify checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found at $CHECKPOINT"
    exit 1
fi
echo "✓ Checkpoint found: $CHECKPOINT"
echo ""

echo "Starting PDM Score evaluation at $(date)"
echo "=============================================="

# Run PDM Score evaluation
# Use sequential worker to avoid distributed agent loading issues
python ${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score.py \
    train_test_split=navtest \
    agent=${AGENT} \
    agent.checkpoint_path=\"${CHECKPOINT}\" \
    metric_cache_path=\"${METRIC_CACHE}\" \
    output_dir=\"${OUTPUT_DIR}\" \
    worker=sequential

EVAL_EXIT_CODE=$?

echo ""
echo "=============================================="
if [ $EVAL_EXIT_CODE -eq 0 ]; then
    echo "✓ Evaluation completed successfully at $(date)"
    echo ""
    echo "Results saved to: ${OUTPUT_DIR}"
    echo ""
    echo "To view results:"
    echo "  cat ${OUTPUT_DIR}/metrics_summary.json"
else
    echo "✗ Evaluation failed with exit code: $EVAL_EXIT_CODE"
    echo "Check error log for details"
fi
echo "=============================================="
