# Caching Guide for Hydra-MDP on Torch

## Overview

NAVSIM/GTRS uses two types of caching to speed up training and evaluation:

### 1. **PDM Ground Truth Scores** (Required for Training)
Pre-computed simulation scores for each trajectory in the vocabulary (8192 or 16384 trajectories). These are used during training for multi-target distillation (the "Hydra" heads).

### 2. **Metric Cache** (Required for Evaluation)
Pre-processed map data and coordinate transformations for evaluation. This speeds up PDM score computation significantly.

---

## Quick Start

### Option A: Run Everything at Once
```bash
cd /scratch/ah7072/GTRS/scripts
./setup_all.sh
```

This will:
1. Download PDM ground truths (~1-2 GB)
2. Download pretrained backbones (~200 MB)
3. Submit metric caching jobs for navmini and navtest splits

### Option B: Step by Step

#### 1. Download PDM Ground Truths
```bash
cd /scratch/ah7072/GTRS/scripts
./download_pdm_gt.sh
```

**Location:** `/scratch/ah7072/navsim_workspace/dataset/traj_pdm_v2/`
- `ori/navtrain_8192.pkl` - For 8K vocabulary (used in our configs)
- `ori/navtrain_16384.pkl` - For 16K vocabulary
- `random_aug/` - Augmented versions (not needed for basic experiments)

**Referenced in configs as:**
```yaml
pdm_gt_path: ${oc.env:OPENSCENE_DATA_ROOT}/traj_pdm_v2/ori/navtrain_16384.pkl
```

#### 2. Download Pretrained Backbones
```bash
cd /scratch/ah7072/GTRS/scripts
./download_backbones.sh
```

**Location:** `/scratch/ah7072/navsim_workspace/dataset/models/dd3d_det_final.pth`

**Referenced in configs as:**
```yaml
vov_ckpt: ${oc.env:OPENSCENE_DATA_ROOT}/models/dd3d_det_final.pth
```

#### 3. Generate Metric Cache
```bash
# For navmini (quick testing, ~20-30 mins)
sbatch torch_cache_metrics.slurm navmini

# For navtest (full evaluation, ~2-4 hours)
sbatch torch_cache_metrics.slurm navtest

# Monitor jobs
squeue -u ah7072
```

**Location:** `/scratch/ah7072/navsim_workspace/exp/metric_cache/{split}/`

---

## How Caching Works

### PDM Ground Truth (Training Time)

During **training**, the model learns to predict PDM sub-scores for each trajectory in the vocabulary:

```python
# From gtrs_agent.py
if pdm_gt_path is not None:
    self.vocab_pdm_score_full = pickle.load(open(pdm_gt_path, "rb"))
```

The pickle file contains:
```python
{
    "no_at_fault_collisions": torch.Tensor,    # [num_scenes, vocab_size]
    "drivable_area_compliance": torch.Tensor,  # [num_scenes, vocab_size]
    "time_to_collision_within_bound": torch.Tensor,
    "ego_progress": torch.Tensor,
    "comfort": torch.Tensor,
    # ... other metrics
}
```

Each entry tells the model: "For scene X, trajectory Y from the vocabulary, the PDM sub-scores are: NC=0.95, DAC=1.0, ..."

The model trains its Hydra heads to predict these scores from sensor observations alone.

### Metric Cache (Evaluation Time)

During **evaluation**, the cache stores:
- Map data in ego vehicle frame
- Drivable area polygons
- Lane boundaries
- Traffic light states

This avoids expensive map queries for every evaluation step.

**Usage in evaluation:**
```bash
python navsim/planning/script/run_pdm_score.py \
    agent=hydra_mdp_v8192_w_ep \
    checkpoint_path=/path/to/checkpoint.ckpt \
    split=navtest \
    metric_cache_path=${NAVSIM_EXP_ROOT}/metric_cache  # Uses cached data
```

---

## Verification

### Check PDM GT Downloaded
```bash
ls -lh /scratch/ah7072/navsim_workspace/dataset/traj_pdm_v2/ori/
# Should see:
# navtrain_8192.pkl   (~XXX MB)
# navtrain_16384.pkl  (~XXX MB)
```

### Check Backbone Downloaded
```bash
ls -lh /scratch/ah7072/navsim_workspace/dataset/models/
# Should see:
# dd3d_det_final.pth  (~200 MB)
```

### Check Metric Cache Generated
```bash
ls /scratch/ah7072/navsim_workspace/exp/metric_cache/
# Should see directories:
# navmini/
# navtest/

# Check contents
ls /scratch/ah7072/navsim_workspace/exp/metric_cache/navtest/
# Should see .pkl files for each scene
```

---

## Training vs Evaluation

### Training (Needs PDM GT + Backbone)
```bash
# PDM GT is loaded via agent config
sbatch torch_train_w_ep.slurm  # Uses pdm_gt_path from config
```

The config automatically references:
```yaml
pdm_gt_path: ${oc.env:OPENSCENE_DATA_ROOT}/traj_pdm_v2/ori/navtrain_16384.pkl
```

### Evaluation (Needs Metric Cache)
```bash
# Metric cache is used automatically
sbatch torch_eval.slurm <checkpoint> <name> <config>
```

The evaluation script passes `metric_cache_path` to the evaluator, which speeds up map access.

---

## Troubleshooting

### "PDM GT file not found" during training
```
FileNotFoundError: /scratch/ah7072/navsim_workspace/dataset/traj_pdm_v2/ori/navtrain_16384.pkl
```
**Fix:** Run `./download_pdm_gt.sh`

### "Backbone checkpoint not found" during training
```
FileNotFoundError: /scratch/ah7072/navsim_workspace/dataset/models/dd3d_det_final.pth
```
**Fix:** Run `./download_backbones.sh`

### Evaluation is very slow
**Likely cause:** Metric cache wasn't generated or path is wrong.

**Fix:** 
```bash
sbatch torch_cache_metrics.slurm navtest
# Wait for job to complete, then re-run evaluation
```

### Cache generation fails with map errors
**Likely cause:** Maps not downloaded or `NUPLAN_MAPS_ROOT` incorrect.

**Check maps:**
```bash
ls /scratch/ah7072/navsim_workspace/dataset/maps/
# Should see: nuplan-maps-v1.0.json + map directories
```

---

## Disk Space Requirements

| Component | Size | Purpose |
|-----------|------|---------|
| PDM GT (8K) | ~500 MB | Training with 8192 vocab |
| PDM GT (16K) | ~1 GB | Training with 16384 vocab |
| DD3D Backbone | ~200 MB | Training (VoV-99 encoder) |
| Metric Cache (navmini) | ~50 MB | Quick evaluation |
| Metric Cache (navtest) | ~200 MB | Full evaluation |

**Total:** ~2 GB for full setup

---

## Advanced: Custom Vocabulary

If you want to generate PDM GT for a custom trajectory vocabulary:

1. Create your vocabulary (K-means on trajectory samples)
2. Run PDM simulation for each vocabulary trajectory on all training scenes
3. Save as pickle with structure matching existing files

This is computationally expensive (days of compute) and not recommended unless you're significantly changing the approach.

---

## Summary Commands

```bash
# One-time setup (run once)
cd /scratch/ah7072/GTRS/scripts
./setup_all.sh

# Verify downloads
ls /scratch/ah7072/navsim_workspace/dataset/traj_pdm_v2/ori/
ls /scratch/ah7072/navsim_workspace/dataset/models/
squeue -u ah7072  # Check cache jobs

# After cache jobs complete
ls /scratch/ah7072/navsim_workspace/exp/metric_cache/

# Then you're ready to train!
sbatch torch_smoke_test.slurm  # Quick test first
./launch_all_torch.sh           # Full training
```
