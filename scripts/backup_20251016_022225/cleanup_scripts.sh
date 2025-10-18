#!/bin/bash
# Script to clean up redundant SLURM scripts
# Run with: bash cleanup_scripts.sh

echo "🧹 Cleaning up redundant scripts..."

# Create backup directory
BACKUP_DIR="/scratch/ah7072/scripts/backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "📦 Backup directory: $BACKUP_DIR"

# ============================================
# REDUNDANT CACHING SCRIPTS (failed attempts)
# ============================================
echo ""
echo "Moving REDUNDANT caching scripts to backup..."

# These all failed or are duplicates
REDUNDANT_CACHE=(
    "cache_1gpu.slurm"                          # Failed: Ray OOM
    "cache_ray_gpu.slurm"                       # Failed: Ray OOM
    "cache_ray_limited.slurm"                   # Duplicate attempt
    "cache_sequential_gpu.slurm"                # Unnecessary fallback
    "cache_best.slurm"                          # ThreadPool - didn't work
    "cache_training_features.slurm"             # Old naming
    "cache_training_features_parallel.slurm"    # Duplicate
    "cache_training_features_gpu.slurm"         # Duplicate
)

for script in "${REDUNDANT_CACHE[@]}"; do
    if [ -f "/scratch/ah7072/scripts/$script" ]; then
        mv "/scratch/ah7072/scripts/$script" "$BACKUP_DIR/"
        echo "  ✓ Moved $script"
    fi
done

# ============================================
# REDUNDANT TRAINING SCRIPTS (failed DDP)
# ============================================
echo ""
echo "Moving REDUNDANT training scripts to backup..."

REDUNDANT_TRAIN=(
    "torch_train_8gpu.slurm"                    # Wrong: gres=gpu:h200:1 (only 1 GPU!)
    "torch_train_8gpu_spawn.slurm"              # Failed DDP attempt
    "torch_train_8gpu_torchrun.slurm"           # Failed torchrun attempt
    "torch_train_4gpu.slurm"                    # Duplicate/unnecessary
    "torch_train_optimized.slurm"               # Unclear purpose
    "torch_train_pdm.slurm"                     # Specific variant (can keep if needed)
    "torch_train_w.slurm"                       # Specific variant (can keep if needed)
    "torch_train_w_ep.slurm"                    # Specific variant (can keep if needed)
)

for script in "${REDUNDANT_TRAIN[@]}"; do
    if [ -f "/scratch/ah7072/scripts/$script" ]; then
        mv "/scratch/ah7072/scripts/$script" "$BACKUP_DIR/"
        echo "  ✓ Moved $script"
    fi
done

# ============================================
# SUMMARY
# ============================================
echo ""
echo "✅ Cleanup complete!"
echo ""
echo "📁 KEEPING (useful scripts):"
echo "  - train_8gpu_ddp_fixed.slurm    ⭐ NEW: Correct 8-GPU DDP"
echo "  - torch_train_1gpu.slurm         ✓ Working single GPU baseline"
echo "  - cache_ray_1gpu.slurm           🔧 In-progress caching solution"
echo "  - cache_navtrain_parallel.slurm  🔧 Alternative approach"
echo "  - torch_smoke_test.slurm         ✓ Testing"
echo "  - torch_eval.slurm               ✓ Evaluation"
echo "  - launch_*.sh                    ✓ Batch launchers"
echo "  - *.md, *.sh                     ✓ Documentation & utilities"
echo ""
echo "🗑️  BACKED UP to: $BACKUP_DIR"
echo "    (10+ redundant/failed scripts)"
