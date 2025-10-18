#!/bin/bash
# Quick reference for evaluating the trained model

# ============================================================================
# EVALUATION COMMAND FOR BASELINE (EXP-001)
# ============================================================================

# Best checkpoint from 40-epoch training
CHECKPOINT="/scratch/ah7072/experiments/hydra_plus_16384_weighted_ep_ckpt/epoch=39-step=9680.ckpt"

# Output directory
OUTPUT_DIR="/scratch/ah7072/experiments/eval_baseline_epoch39_$(date +%Y%m%d)"

# ============================================================================
# OPTION 1: Interactive Evaluation (Quick Test on navmini)
# ============================================================================

cd /scratch/ah7072/GTRS

# Test on navmini split first (faster, ~100 samples)
python navsim/evaluate/pdm_score.py \
    agent=hydra_mdp_v8192_w_ep \
    checkpoint=$CHECKPOINT \
    split=navmini \
    output_dir="${OUTPUT_DIR}_navmini"

# ============================================================================
# OPTION 2: Full Evaluation on navtest (SLURM Job)
# ============================================================================

# Create SLURM script for full evaluation
cat > eval_baseline_epoch39.slurm << 'EOF'
#!/bin/bash
#SBATCH --job-name=eval_baseline_ep39
#SBATCH --partition=l40s_public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/ah7072/navsim_workspace/exp/logs/eval_%j.out
#SBATCH --error=/scratch/ah7072/navsim_workspace/exp/logs/eval_%j.err

echo "Starting evaluation at $(date)"
echo "Checkpoint: $CHECKPOINT"

# Environment
export NAVSIM_DEVKIT_ROOT="/scratch/ah7072/GTRS"
export OPENSCENE_DATA_ROOT="/scratch/ah7072/data/openscene"
export NUPLAN_MAPS_ROOT="/scratch/ah7072/data/maps"
export NAVSIM_EXP_ROOT="/scratch/ah7072/experiments"

cd $NAVSIM_DEVKIT_ROOT

# Load environment
module purge
module load anaconda3/2025.06
source activate navsim

# Run evaluation on navtest split
python navsim/evaluate/pdm_score.py \
    agent=hydra_mdp_v8192_w_ep \
    checkpoint=/scratch/ah7072/experiments/hydra_plus_16384_weighted_ep_ckpt/epoch=39-step=9680.ckpt \
    split=navtest \
    output_dir=/scratch/ah7072/experiments/eval_baseline_epoch39_$(date +%Y%m%d)

echo "Evaluation complete at $(date)"
EOF

# Submit evaluation job
sbatch eval_baseline_epoch39.slurm

# ============================================================================
# OPTION 3: Evaluate Multiple Checkpoints (Best Selection)
# ============================================================================

# Evaluate epochs 35-39 to find best validation checkpoint
for EPOCH in 35 36 37 38 39; do
    CKPT="/scratch/ah7072/experiments/hydra_plus_16384_weighted_ep_ckpt/epoch=${EPOCH}-step=*.ckpt"
    OUT="/scratch/ah7072/experiments/eval_baseline_epoch${EPOCH}_navmini"
    
    echo "Evaluating epoch $EPOCH..."
    python navsim/evaluate/pdm_score.py \
        agent=hydra_mdp_v8192_w_ep \
        checkpoint=$CKPT \
        split=navmini \
        output_dir=$OUT
done

# ============================================================================
# EXPECTED OUTPUT
# ============================================================================

# The evaluation will produce:
# - PDM-Score (overall metric)
# - No-collision score
# - Drivable area score
# - Time-to-collision score
# - Progress score
# - Comfort score
# - JSON file with detailed results
# - Trajectory predictions saved for visualization

# ============================================================================
# NEXT STEPS AFTER EVALUATION
# ============================================================================

# 1. Record PDM-Score in EXPERIMENT_LOG.md
# 2. Compare with published NAVSIM baselines
# 3. Plot training curves (train/val loss over epochs)
# 4. Analyze failure cases if PDM-Score is low
# 5. Begin I-JEPA integration experiments (Week 2)
