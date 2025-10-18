# PDM Score Evaluation Guide

## Quick Start

### Step 1: Quick Test (Recommended First)
Test evaluation on navmini split (~100 samples, ~10-20 minutes):

```bash
cd /scratch/ah7072
sbatch scripts/pdm_score_mini.slurm
```

**Why start with navmini?**
- Fast feedback (~10-20 min vs 2-4 hours)
- Catches configuration errors quickly
- Verifies checkpoint loads correctly
- Tests that all dependencies work

### Step 2: Full Evaluation
Once navmini test passes, run full evaluation on navtest:

```bash
cd /scratch/ah7072
sbatch scripts/pdm_score.slurm
```

**Expected time**: 2-4 hours on single L40S GPU  
**Output**: Complete PDM-Score and all component metrics

---

## What Gets Evaluated

The PDM-Score evaluation measures:

1. **No Collision (NC)**: Avoids collisions with other agents
2. **Drivable Area (DA)**: Stays within legal driving areas
3. **Driving Direction (DD)**: Follows correct traffic flow
4. **Progress (P)**: Makes forward progress toward goal
5. **Time-to-Collision (TTC)**: Maintains safe distances
6. **Comfort (C)**: Smooth acceleration/steering
7. **Lane Following (LF)**: Stays in lane
8. **Traffic Light (TL)**: Obeys traffic signals

**Final PDM-Score**: Weighted combination of all metrics (0-100 scale)

---

## Monitoring Evaluation

### Check job status:
```bash
squeue -u $USER
```

### Watch progress in real-time:
```bash
# For navmini test
tail -f /scratch/ah7072/navsim_workspace/exp/logs/eval_<JOB_ID>.out

# Or get latest
tail -f $(ls -t /scratch/ah7072/navsim_workspace/exp/logs/eval_*.out | head -1)
```

### Check for errors:
```bash
tail -f $(ls -t /scratch/ah7072/navsim_workspace/exp/logs/eval_*.err | head -1)
```

---

## Understanding Results

### View metrics summary:
```bash
# Find your evaluation output directory
ls -ltr /scratch/ah7072/experiments/eval_*

# View metrics (JSON format)
cat /scratch/ah7072/experiments/eval_baseline_epoch39_*/metrics_summary.json
```

### Expected output format:
```json
{
  "pdm_score": 75.3,
  "no_collision": 0.95,
  "drivable_area": 0.88,
  "driving_direction": 0.92,
  "progress": 0.85,
  "ttc": 0.78,
  "comfort": 0.82,
  "lane_following": 0.76,
  "traffic_light": 0.91
}
```

**Higher is better** for all metrics (0-1 scale, except pdm_score is 0-100)

---

## Comparing with Baselines

### NAVSIM Published Baselines (from paper):
- **Constant Velocity**: PDM-Score ~45
- **MLP Baseline**: PDM-Score ~55
- **TransFuser**: PDM-Score ~68
- **Hydra (official)**: PDM-Score ~72-75

**Your goal**: Match or exceed official Hydra baseline (~72-75)

---

## Troubleshooting

### Issue: "Checkpoint not found"
```bash
# Verify checkpoint exists
ls -lh /scratch/ah7072/experiments/hydra_plus_16384_weighted_ep_ckpt/epoch=39-step=9680.ckpt
```

If missing, check available checkpoints:
```bash
ls -lh /scratch/ah7072/experiments/hydra_plus_16384_weighted_ep_ckpt/
```

### Issue: "Out of memory"
If evaluation OOMs, edit the script to request more memory:
```bash
#SBATCH --mem=128GB  # Increase from 64GB
```

### Issue: "CUDA out of memory"
Reduce batch size in evaluation config (if configurable) or use H200 GPU:
```bash
#SBATCH --partition=h200_tandon
#SBATCH --gres=gpu:1
```

### Issue: Evaluation seems stuck
Check if it's actually making progress:
```bash
# Watch for new prediction files being created
watch -n 5 "ls /scratch/ah7072/experiments/eval_*/predictions/ | wc -l"
```

---

## After Evaluation Completes

### 1. Record Results
Update `/scratch/ah7072/EXPERIMENT_LOG.md`:
- Add PDM-Score and all component metrics
- Compare with published baselines
- Note any surprising results (good or bad)

### 2. Analyze Results
```python
# Quick analysis script
import json

results_file = "/scratch/ah7072/experiments/eval_baseline_epoch39_*/metrics_summary.json"
with open(results_file) as f:
    results = json.load(f)

print(f"PDM-Score: {results['pdm_score']:.2f}")
print(f"Strongest metric: {max(results.items(), key=lambda x: x[1] if x[0] != 'pdm_score' else 0)}")
print(f"Weakest metric: {min(results.items(), key=lambda x: x[1] if x[0] != 'pdm_score' else 1)}")
```

### 3. Visualize Predictions (Optional)
If evaluation saved trajectory predictions:
```bash
# Look for prediction files
ls /scratch/ah7072/experiments/eval_baseline_epoch39_*/predictions/

# Visualize with NAVSIM tools (if available)
python navsim/visualization/visualize_predictions.py \
    --predictions /scratch/ah7072/experiments/eval_*/predictions/ \
    --output visualizations/
```

### 4. Update Thesis Document
Document:
- Final PDM-Score
- Training time (14h) vs inference time (2-4h)
- Comparison with published baselines
- Any insights from component metrics

---

## Next Steps After Baseline Evaluation

Once you have the baseline PDM-Score:

1. **Week 2**: Start I-JEPA integration experiments
2. **Week 3**: Label efficiency experiments (10/25/50/100% data)
3. **Week 4**: Ablation studies and results analysis

**Goal**: Show I-JEPA features achieve comparable or better PDM-Score with less labeled data

---

## Quick Reference Commands

```bash
# Submit quick test (navmini)
sbatch scripts/pdm_score_mini.slurm

# Submit full evaluation (navtest)
sbatch scripts/pdm_score.slurm

# Check job status
squeue -u $USER

# View results
cat $(find /scratch/ah7072/experiments/eval_* -name "metrics_summary.json" | tail -1)

# Cancel job if needed
scancel <JOB_ID>
```

---

**Files Created:**
- `/scratch/ah7072/scripts/pdm_score.slurm` - Full evaluation on navtest
- `/scratch/ah7072/scripts/pdm_score_mini.slurm` - Quick test on navmini
- `/scratch/ah7072/EVAL_README.md` - This guide

**Recommended**: Start with navmini test, then run full navtest evaluation!
