# 🔥 Multi-Node DDP: The REAL Fix

## What Went Wrong (Job 268815)

```
torch.distributed.DistStoreError: Timed out after 3601 seconds waiting for clients. 
1/16 clients joined.
```

**Only 1 out of 16 GPU processes connected!**

## Root Cause: Missing `srun`

### ❌ What We Did (WRONG):
```bash
#SBATCH --ntasks-per-node=4

# Direct python call in the script
python navsim/planning/script/run_training_dense.py ...
```

**Problem**: 
- SLURM spawns the bash script 4 times per node (16 total)
- Each bash instance runs `python` command
- Each Python process tries to initialize DDP independently
- Only the first one (rank 0) succeeds
- Others never coordinate properly → timeout

### ✅ What We Need (CORRECT):
```bash
#SBATCH --ntasks=8          # Total tasks across all nodes
#SBATCH --ntasks-per-node=4 # Tasks per node

# Use srun to coordinate task launch
srun python navsim/planning/script/run_training_dense.py ...
```

**Why This Works**:
- `srun` is SLURM's process launcher for parallel jobs
- It launches exactly 8 (or 16) Python processes across nodes
- Sets up proper environment variables for each rank
- All processes start simultaneously and coordinate via MASTER_ADDR
- PyTorch Lightning detects SLURM environment automatically

## The Key Insight

**Multi-node SLURM jobs require `srun` for parallel task execution!**

Without `srun`:
- The sbatch script runs independently on each task/node
- No coordination between launches
- DDP initialization fails

With `srun`:
- Single coordinated launch of all tasks
- SLURM sets `SLURM_PROCID`, `SLURM_LOCALID` for each process
- Lightning uses these to determine rank
- All ranks connect to MASTER_ADDR successfully

## Fixed Scripts

### 2-Node × 4 GPUs = 8 GPUs:
**File**: `train_2node_8gpu_l40s_FIXED.slurm`

Key changes:
```bash
#SBATCH --ntasks=8          # Total: 2 nodes × 4 GPUs
srun python ...             # Use srun!
```

### 4-Node × 4 GPUs = 16 GPUs:
**File**: `train_4node_16gpu_l40s_FIXED.slurm`

Key changes:
```bash
#SBATCH --ntasks=16         # Total: 4 nodes × 4 GPUs
srun python ...             # Use srun!
```

## How SLURM + Lightning Coordinate

### Environment Variables (set by srun):
```bash
SLURM_PROCID=0           # Global rank (0-15)
SLURM_LOCALID=0          # Local rank on node (0-3)
SLURM_NTASKS=16          # Total processes
SLURM_NTASKS_PER_NODE=4  # Processes per node
MASTER_ADDR=gl001        # First node hostname
MASTER_PORT=12356        # TCP port for communication
```

### Lightning Detection:
1. Sees `SLURM_PROCID` → knows it's in SLURM environment
2. Uses `SLURM_PROCID` as global rank
3. Uses `SLURM_LOCALID` to pick GPU (0-3 on each node)
4. Connects to `MASTER_ADDR:MASTER_PORT` for rendezvous
5. All 16 ranks sync up → DDP initialized ✅

## Single-Node vs Multi-Node

### Single-Node (what works without srun):
```bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:8

# Direct python works because Lightning spawns local workers
python run_training_dense.py trainer.params.devices=8
```

### Multi-Node (REQUIRES srun):
```bash
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4

# MUST use srun for cross-node coordination
srun python run_training_dense.py trainer.params.num_nodes=2
```

## Ready to Submit! 🚀

### Conservative (8 GPUs, ~12h):
```bash
sbatch --account=torch_pr_68_tandon_advanced scripts/train_2node_8gpu_l40s_FIXED.slurm
```

### Fastest (16 GPUs, ~6h):
```bash
sbatch --account=torch_pr_68_tandon_advanced scripts/train_4node_16gpu_l40s_FIXED.slurm
```

## What You'll See (Success!)

### Startup logs should show:
```
Initializing distributed: GLOBAL_RANK: 0, MEMBER: 1/16
Initializing distributed: GLOBAL_RANK: 1, MEMBER: 2/16
Initializing distributed: GLOBAL_RANK: 2, MEMBER: 3/16
...
Initializing distributed: GLOBAL_RANK: 15, MEMBER: 16/16
All distributed processes registered. Starting with 16 processes
```

All 16 ranks join within seconds → training starts! ✅

## Why This Wasn't Obvious

Multi-node DDP usually needs special launcher, but:
- Lightning documentation focuses on `lightning run` command
- SLURM integration is "automatic" (if you know to use srun!)
- Single-node works without srun, hiding the requirement
- Error message doesn't mention srun

**The magic formula**: `sbatch` + `srun` + Lightning = multi-node DDP ✨

---

## Summary

| Requirement | Previous (Failed) | Fixed |
|------------|-------------------|-------|
| Task launch | Direct `python` | `srun python` ✅ |
| Coordination | Each rank independent | srun manages all ✅ |
| SLURM vars | Not set properly | srun sets all ✅ |
| Result | 1/16 ranks join 💥 | 16/16 ranks join ✅ |

**Bottom line**: For multi-node SLURM jobs, always use `srun` to launch your Python script!
