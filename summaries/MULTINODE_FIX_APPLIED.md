# 🔧 Multi-Node DDP Fix Applied

## Problem Identified
Job 268798 failed with:
```
ValueError: You set `devices=4` in Lightning, but the number of tasks per node 
configured in SLURM `--ntasks-per-node=1` does not match. HINT: Set `devices=1`.
```

## Root Cause
**For multi-node DDP with SLURM, PyTorch Lightning requires:**
- `--ntasks-per-node` = number of GPUs per node
- Each SLURM task manages one GPU process

**We had**: `--ntasks-per-node=1` (wrong - only 1 process per node!)
**Need**: `--ntasks-per-node=4` (correct - one process per GPU)

## The Fix

### Changed in both scripts:
1. **train_2node_8gpu_l40s.slurm**
2. **train_4node_16gpu_l40s.slurm**

### Before (WRONG):
```bash
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

### After (CORRECT):
```bash
#SBATCH --ntasks-per-node=4    # 4 tasks = 4 GPUs
#SBATCH --cpus-per-task=16     # 64 CPUs / 4 tasks = 16 per task
# CUDA_VISIBLE_DEVICES removed - SLURM assigns GPUs to tasks automatically
```

## Why This Works

### SLURM Side:
- `--gres=gpu:4` requests 4 GPUs per node
- `--ntasks-per-node=4` launches 4 processes per node
- SLURM automatically assigns 1 GPU to each task via `CUDA_VISIBLE_DEVICES`

### Lightning Side:
- Detects 4 tasks per node from `SLURM_NTASKS_PER_NODE`
- Validates: tasks_per_node (4) == GPUs detected per process (1)
- Each Lightning process manages one GPU
- Coordinates across all 8 (or 16) processes via MASTER_ADDR

### Result:
- 2 nodes × 4 tasks/node = 8 total GPU processes ✅
- 4 nodes × 4 tasks/node = 16 total GPU processes ✅

## Ready to Submit!

### 2-Node Option (8 GPUs):
```bash
sbatch --account=torch_pr_68_tandon_advanced scripts/train_2node_8gpu_l40s.slurm
```

### 4-Node Option (16 GPUs - FASTEST):
```bash
sbatch --account=torch_pr_68_tandon_advanced scripts/train_4node_16gpu_l40s.slurm
```

## Expected Behavior

### During startup you should see:
```
[rank: 0] on gl001
[rank: 1] on gl001
[rank: 2] on gl001
[rank: 3] on gl001
[rank: 4] on gl006
[rank: 5] on gl006
[rank: 6] on gl006
[rank: 7] on gl006
```

All 8 ranks (or 16 for 4-node) should initialize successfully!

## Clean Up Done
- ✅ Cancelled failed job 268798
- ✅ Fixed train_2node_8gpu_l40s.slurm
- ✅ Fixed train_4node_16gpu_l40s.slurm
- ✅ Ready to resubmit!
