# Hydra-MDP Complete Workflow on Torch

## 🚀 Quick Start (First Time Setup)

```bash
# 1. One-time setup (downloads + caching)
cd /scratch/ah7072/GTRS/scripts
./setup_all.sh

# 2. Monitor cache jobs (wait for completion)
squeue -u ah7072

# 3. Run smoke test (30 mins)
sbatch torch_smoke_test.slurm

# 4. Launch full training (3 experiments, ~48h each)
./launch_all_torch.sh

# 5. After training, evaluate all models
./launch_eval_torch.sh
```

---

## 📋 Complete Workflow

### Phase 1: Prerequisites (One-Time Setup)

#### Downloads (~2 GB total)
```bash
cd /scratch/ah7072/GTRS/scripts

# Download PDM ground truths (for training)
./download_pdm_gt.sh
# → /scratch/ah7072/navsim_workspace/dataset/traj_pdm_v2/ori/navtrain_16384.pkl

# Download pretrained backbone (for training)
./download_backbones.sh
# → /scratch/ah7072/navsim_workspace/dataset/models/dd3d_det_final.pth

# Submit metric caching jobs (for evaluation)
sbatch torch_cache_metrics.slurm navmini  # Quick testing
sbatch torch_cache_metrics.slurm navtest  # Final evaluation

# Monitor cache jobs
squeue -u ah7072
tail -f /scratch/ah7072/navsim_workspace/exp/logs/cache_metrics_*.out
```

**What these do:**
- **PDM GT:** Pre-computed scores for each vocabulary trajectory (needed for multi-target distillation)
- **Backbone:** DD3D pretrained weights (needed for VoV encoder initialization)
- **Metric Cache:** Pre-processed map data (speeds up evaluation)

---

### Phase 2: Validation (Smoke Test)

```bash
# Run 1-epoch test to catch errors early (~30 mins)
cd /scratch/ah7072/GTRS/scripts
sbatch torch_smoke_test.slurm

# Monitor
squeue -u ah7072
tail -f /scratch/ah7072/navsim_workspace/exp/logs/smoke_*.out

# Check for errors
cat /scratch/ah7072/navsim_workspace/exp/logs/smoke_*.err
```

**If smoke test passes → proceed to full training**

---

### Phase 3: Training (3 Experiments)

```bash
# Launch all three experiments
cd /scratch/ah7072/GTRS/scripts
./launch_all_torch.sh

# Or launch individually:
sbatch torch_train_pdm.slurm     # PDM-only baseline (Score 80.2)
sbatch torch_train_w.slurm       # Weighted confidence (Score 85.7)
sbatch torch_train_w_ep.slurm    # Weighted + continuous EP (Score 86.5) - BEST

# Monitor jobs
squeue -u ah7072

# Watch logs
tail -f /scratch/ah7072/navsim_workspace/exp/logs/hydra_pdm_*.out
tail -f /scratch/ah7072/navsim_workspace/exp/logs/hydra_w_*.out
tail -f /scratch/ah7072/navsim_workspace/exp/logs/hydra_w_ep_*.out
```

**Training time:** ~36-48 hours each on H200/L40S

**Checkpoints saved to:**
```
/scratch/ah7072/navsim_workspace/exp/
├── hydra_pdm_v8192_<timestamp>/checkpoints/last.ckpt
├── hydra_w_v8192_<timestamp>/checkpoints/last.ckpt
└── hydra_w_ep_v8192_<timestamp>/checkpoints/last.ckpt
```

---

### Phase 4: Evaluation

```bash
# Option 1: Evaluate all trained models automatically
cd /scratch/ah7072/GTRS/scripts
./launch_eval_torch.sh

# Option 2: Evaluate specific checkpoint
sbatch --export=\
CHECKPOINT_PATH="/scratch/ah7072/navsim_workspace/exp/hydra_w_ep_v8192_<timestamp>/checkpoints/last.ckpt",\
EXPERIMENT_NAME="w_ep_eval",\
CONFIG_NAME="hydra_mdp_v8192_w_ep",\
EVAL_SPLIT="navtest" \
torch_eval.slurm

# Monitor evaluation
squeue -u ah7072
tail -f /scratch/ah7072/navsim_workspace/exp/logs/eval_*.out
```

**Evaluation time:** ~4-6 hours per model on navtest

**Results saved to:**
```
/scratch/ah7072/navsim_workspace/exp/eval_<name>_<timestamp>/
├── pdm_score.json           # Final scores
├── metrics/                 # Per-scene breakdowns
└── logs/
```

---

### Phase 5: Results Analysis

```bash
# Find all evaluation results
find /scratch/ah7072/navsim_workspace/exp -name "pdm_score.json" -type f

# View PDM scores
for f in /scratch/ah7072/navsim_workspace/exp/eval_*/pdm_score.json; do
    echo "=== $(basename $(dirname $f)) ==="
    cat $f | python -m json.tool | grep -E '"score"|"no_collision"|"drivable_area"|"ego_progress"|"time_to_collision"|"comfort"'
    echo ""
done

# Copy results to Greene for analysis
scp -rp /scratch/ah7072/navsim_workspace/exp/eval_* dtn-1:/scratch/ah7072/hydra_results/
```

---

## 📊 Expected Results (Table 1)

| Variant | Script | NC↑ | DAC↑ | EP↑ | TTC↑ | C↑ | **Score↑** |
|---------|--------|-----|------|-----|------|-----|------------|
| PDM-only | `torch_train_pdm.slurm` | 97.5 | 88.9 | 74.8 | 92.5 | 100 | **80.2** |
| Weighted (W) | `torch_train_w.slurm` | 98.1 | 96.1 | 77.8 | 93.9 | 100 | **85.7** |
| W + Continuous EP | `torch_train_w_ep.slurm` | 98.3 | 96.0 | 78.7 | 94.6 | 100 | **86.5** |

---

## 🔧 Useful Commands

### Job Management
```bash
# Check queue
squeue -u ah7072

# Cancel job
scancel <JOB_ID>

# Job details
scontrol show job <JOB_ID>

# Check GPU availability
sinfo -p h200,l40s -O partition,available,nodes,gres
```

### Monitoring
```bash
# Watch logs live
tail -f /scratch/ah7072/navsim_workspace/exp/logs/*.out

# Check disk usage
du -sh /scratch/ah7072/navsim_workspace/exp/*

# Find checkpoints
find /scratch/ah7072/navsim_workspace/exp -name "last.ckpt" -type f
```

### Troubleshooting
```bash
# Check environment
echo $NAVSIM_DEVKIT_ROOT
echo $OPENSCENE_DATA_ROOT
echo $NUPLAN_MAPS_ROOT
echo $NAVSIM_EXP_ROOT

# Verify PDM GT downloaded
ls -lh /scratch/ah7072/navsim_workspace/dataset/traj_pdm_v2/ori/

# Verify backbone downloaded
ls -lh /scratch/ah7072/navsim_workspace/dataset/models/

# Verify cache exists
ls /scratch/ah7072/navsim_workspace/exp/metric_cache/navtest/
```

---

## 📚 Documentation

- **[CACHING_GUIDE.md](CACHING_GUIDE.md)** - Detailed caching explanation
- **[HYDRA_MDP_TORCH_README.md](HYDRA_MDP_TORCH_README.md)** - Config details
- **[PATHS_SETUP.md](PATHS_SETUP.md)** - Path configuration
- **[QUICK_COMMANDS.sh](QUICK_COMMANDS.sh)** - Command reference

---

## ⚠️ Important Notes

1. **Run setup_all.sh first** - Downloads prerequisites (~2 GB)
2. **Wait for cache jobs** - Don't start evaluation without metric cache
3. **Smoke test first** - Catch errors early (30 mins vs 48 hours)
4. **Jobs can be preempted** - Use `--comment="preemption=yes;requeue=yes"` (already in scripts)
5. **Check logs for errors** - Both .out and .err files

---

## 🎯 Success Criteria

✅ Setup complete when:
- PDM GT files exist (~1 GB)
- Backbone file exists (~200 MB)
- Metric cache directories populated

✅ Training successful when:
- No errors in .err logs
- Checkpoints saved to exp/{variant}_*/checkpoints/
- Training curves converge (check .out logs)

✅ Evaluation successful when:
- PDM scores match expected ranges (80-87)
- All sub-metrics computed (NC, DAC, EP, TTC, C)
- Results saved to eval_*/pdm_score.json

---

## 📞 Getting Help

**Check documentation:**
```bash
cd /scratch/ah7072/GTRS/scripts
cat CACHING_GUIDE.md
cat PATHS_SETUP.md
```

**Common issues:**
1. "PDM GT not found" → Run `./download_pdm_gt.sh`
2. "Backbone not found" → Run `./download_backbones.sh`
3. "Slow evaluation" → Run `sbatch torch_cache_metrics.slurm navtest`
4. "Out of memory" → Reduce batch_size in config (default: 256 → try 128)
5. "Job preempted" → Requeue automatic, check logs for checkpoint resume
