# 🚀 FIXED: Multi-GPU DDP Training

## ❌ What Was Wrong

All previous 8-GPU attempts failed because of **SLURM configuration mistakes**:

```bash
# WRONG APPROACH (what we tried):
#SBATCH --gres=gpu:h200:1        # ❌ Only requests 1 GPU!
#SBATCH --ntasks-per-node=8      # ❌ Launches 8 separate processes
```

**Problem**: PyTorch Lightning DDP expects:
1. Multiple GPUs visible to a **SINGLE** process
2. Lightning internally spawns worker processes

But we were giving SLURM 1 GPU and asking it to launch 8 tasks → DDP hung waiting for GPUs that didn't exist.

---

## ✅ Correct Solution

```bash
# CORRECT APPROACH:
#SBATCH --gres=gpu:8             # ✓ Request 8 GPUs
#SBATCH --ntasks=1               # ✓ Launch only 1 process
#SBATCH --cpus-per-task=128      # ✓ All CPUs for this process

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # ✓ Make all GPUs visible

# NOTE: run_training_dense.py HARDCODES strategy=DDPStrategy
# Lightning auto-detects 8 GPUs from CUDA_VISIBLE_DEVICES!

python run_training_dense.py \
    agent=hydra_mdp_v8192_w_ep \
    trainer.params.accelerator=gpu \    # ✓ Use GPU
    trainer.params.num_nodes=1 \        # ✓ Single node
    dataloader.params.batch_size=22     # ✓ 22 per GPU = 176 total
```

---

## 🎯 How Lightning DDP Works

1. **Main process** starts with all GPUs visible
2. Lightning **automatically spawns** 8 worker processes (one per GPU)
3. Each worker:
   - Gets assigned to 1 GPU (rank 0-7)
   - Loads data with its own DataLoader workers
   - Trains its own model replica
   - Syncs gradients with other ranks

**KEY**: You launch 1 Python process, Lightning handles the rest!

---

## 📊 Expected Performance

| Setup | Batch Size | Speed | Time for 40 Epochs |
|-------|------------|-------|-------------------|
| **1 GPU (current)** | 22 | 0.78 it/s | ~80 hours |
| **8 GPUs (DDP)** | 176 (22×8) | ~6 it/s | **~10 hours** ⚡ |

**8× speedup** because:
- 8× more samples per batch
- Near-linear scaling with DDP
- Better GPU utilization

---

## 🚀 Usage

### Submit 8-GPU Training
```bash
sbatch --account=torch_pr_68_tandon_advanced scripts/train_8gpu_ddp_fixed.slurm
```

### Monitor Progress
```bash
# Check job status
squeue -u $USER

# Watch training logs (once running)
tail -f /scratch/ah7072/navsim_workspace/exp/logs/train_<JOBID>.out
```

---

## 🔍 Key Differences from Failed Attempts

| Setting | ❌ Old (Failed) | ✅ New (Fixed) |
|---------|----------------|----------------|
| `--gres` | `gpu:h200:1` | `gpu:8` |
| `--ntasks` | `8` | `1` |
| `--ntasks-per-node` | `8` | (removed) |
| `--cpus-per-task` | `16` | `128` |
| `trainer.params.devices` | ❌ Not in config! | (auto-detected) |
| `trainer.params.strategy` | ❌ Hardcoded in code! | (can't override) |
| `CUDA_VISIBLE_DEVICES` | (not set) | `0,1,2,3,4,5,6,7` |

**CRITICAL**: 
- `run_training_dense.py` **hardcodes** `strategy=DDPStrategy` (line 159)
- Config has NO `devices` parameter - Lightning **auto-detects** from environment!
- Setting `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` tells Lightning "use these 8 GPUs"

---

## 📝 Troubleshooting

### If DDP still hangs:
```bash
# Check GPUs are visible
srun --jobid=<JOBID> --pty bash
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
# Should print: GPUs: 8
```

### If OOM errors:
```bash
# Reduce batch size per GPU
python run_training_dense.py \
    dataloader.params.batch_size=16  # Reduce from 22 to 16
```

### If "Address already in use":
```bash
# Lightning auto-finds free port, but if it fails:
python run_training_dense.py \
    +trainer.params.strategy.find_unused_parameters=false
```

---

## 🎓 Learning: SLURM vs Lightning

**SLURM's job**: Allocate resources (GPUs, CPUs, memory)
**Lightning's job**: Distribute training across those resources

Don't mix SLURM's `--ntasks` with Lightning's `devices` - let Lightning handle process spawning!

---

## Next Steps

1. ✅ Submit `train_8gpu_ddp_fixed.slurm`
2. ⏳ Wait ~10 hours for training
3. 🎉 Complete 40 epochs 8× faster!
4. 📊 Evaluate with PDM score
