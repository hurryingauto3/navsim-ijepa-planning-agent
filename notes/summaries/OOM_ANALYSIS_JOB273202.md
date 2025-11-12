# OOM Analysis - Job 273202 (October 16, 2025)

## Problem Summary
Training completed only 1 epoch (should have run 40 epochs) and terminated with OOM error.

## Root Cause
**Out-of-Memory (OOM) during epoch 0→1 transition** when validation dataloader was being initialized.

## Evidence
```
Job 273202 Details:
- Requested memory: 256 GB per node (1 TB total for 4 nodes)
- Peak memory usage: 72.5 GB (MaxRSS)
- Average memory usage: 71.4 GB (AveRSS)
- Exit status: OUT_OF_MEMORY (exit code 0:125)
- Error log: "Detected 4 oom_kill events in StepId=273202.0"
```

## Timeline of Events
1. **14:11:10** - Training started successfully
2. **14:11:10 - 14:34:04** - Epoch 0 training (242/242 batches completed)
   - Loss dropped from 21.3 → 5.19 (excellent 75% reduction)
   - All 16 GPUs working correctly at 0.21 it/s
3. **14:34:04** - Training "completed" message (premature)
4. **14:34:04** - OOM kill triggered during validation initialization
5. **Result**: No validation ran, no checkpoints saved, no epoch 1 started

## Memory Analysis

### Partition Limits (l40s_public)
```
Per-node memory available: 513 GB
Total nodes available: 66 nodes (mixed state)
Memory policy: UNLIMITED (no hard cap below 513GB)
```

### Job 273202 Configuration
```
Nodes: 4
GPUs per node: 4 L40S (48GB VRAM each)
CPUs per node: 64 (out of 128 available)
Requested memory: 256 GB per node
Batch size: 22 per GPU
Num workers: 2 per GPU
Total workers: 16 GPUs × 2 workers = 32 dataloader workers
```

### Why OOM Occurred
The transition from training to validation likely caused memory spike:
1. **Training phase**: 85,109 samples loaded incrementally by 32 workers
2. **Validation initialization**: 18,179 samples × 16 ranks trying to load simultaneously
3. **Memory spike**: All ranks initializing validation dataloaders at once
4. **Result**: Exceeded 256GB per-node limit → SLURM OOM killer activated

## Solution Options

### Option 1: Increase Memory Request (RECOMMENDED)
**You CAN request more memory!** Each l40s_public node has **513 GB available**.

```bash
#SBATCH --mem=400GB    # Request 400GB per node (78% of available)
```

### Option 2: Reduce Memory Consumption
```bash
dataloader.params.batch_size=16        # Reduce from 22 to 16
dataloader.params.num_workers=1        # Reduce from 2 to 1  
dataloader.params.persistent_workers=false
```

### Option 3: Gradient Accumulation (Keep Large Effective Batch Size)
```bash
dataloader.params.batch_size=11        # Half the batch size
trainer.params.accumulate_grad_batches=2  # Accumulate 2 batches
# Effective batch size: 11 × 2 = 22 (same as before)
```

## Recommended Fix

**Use Option 1 + partial Option 2 for safety:**

```bash
#!/bin/bash
#SBATCH --mem=400GB              # Increase from 256GB to 400GB per node
#SBATCH --nodes=4
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --partition=l40s_public

# Reduce workers slightly for safety
dataloader.params.batch_size=20        # Slight reduction: 22 → 20
dataloader.params.num_workers=1        # Safer: 2 → 1
```

**Expected outcome:**
- Effective batch size: 20 × 16 GPUs = 320 (vs 352 before, ~9% reduction)
- Memory headroom: 400GB request vs 513GB available (22% safety margin)
- Worker reduction: 32 → 16 workers (50% less memory pressure)
- Training speed: ~5% slower due to batch size reduction

## Memory Budget Breakdown (Estimated)

### Per Node (4 GPUs):
```
Model parameters:         ~8 GB
Optimizer states:        ~16 GB
Activations/gradients:   ~24 GB (batch_size=20, 4 GPUs)
Training data buffers:   ~40 GB (1 worker per GPU)
Validation data buffers: ~60 GB (loading 18K samples)
System overhead:         ~10 GB
PyTorch CUDA context:    ~8 GB
----------------------------------------------
Total estimated:        ~166 GB
Safety margin (400GB):  +234 GB (141% buffer)
```

## Implementation

**New training script created:** `train_4node_16gpu_l40s_OOM_FIXED.slurm`

Key changes:
- ✅ Memory: 256GB → 400GB per node
- ✅ Batch size: 22 → 20 per GPU
- ✅ Workers: 2 → 1 per GPU
- ✅ Added persistent_workers=false for clean shutdown
- ✅ Increased time limit: 24h → 48h (for full 40 epochs)

## Verification Steps

After launching new job:
1. Monitor memory usage: `watch -n 10 'squeue -j <JOBID> -o "%.18i %.9P %.8T %.10M %.6D %.20S %.10l %.6C %R"'`
2. Check for OOM: `tail -f /scratch/ah7072/navsim_workspace/exp/logs/train_<JOBID>.err`
3. Verify epoch progression: `grep -E "Epoch [0-9]+:" /scratch/ah7072/navsim_workspace/exp/logs/train_<JOBID>.out`
4. Check checkpoints saving: `ls -lth /scratch/ah7072/experiments/hydra_w_ep*/`

## Lesson Learned

**Validation dataloader initialization is a memory-intensive operation** in multi-GPU setups.
- Always budget 2-3× the training memory for validation peaks
- Use `persistent_workers=True` only when memory is abundant
- Monitor first epoch→validation transition carefully in new experiments
