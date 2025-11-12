# Container Issue Fixed - Updated Workflow

## ✅ What Was Fixed

The original scripts tried to use a non-existent PyTorch container. **Torch HPC doesn't need containers** since it already has CUDA drivers installed. All scripts now run directly with your conda environment.

## 🔧 Changes Made

All Slurm scripts (`.slurm` files) now:
- ❌ **Removed:** Singularity/Apptainer container calls
- ✅ **Added:** Direct conda environment activation
- ✅ **Result:** Faster execution, no container overhead

## 🚀 Updated Quick Start

### Step 1: Create Conda Environment (One-Time)
```bash
# On Torch (cs649)
cd /scratch/ah7072/GTRS/scripts

# Option A: Use helper script
./setup_conda_env.sh

# Option B: Manual setup
cd /scratch/ah7072/navsim_workspace/navsim
conda env create -n navsim -f environment.yml
```

**Time:** ~10-15 minutes

###Step 2: Download Prerequisites
```bash
cd /scratch/ah7072/GTRS/scripts

# Set environment variable (add to ~/.bashrc for persistence)
export OPENSCENE_DATA_ROOT=/scratch/ah7072/navsim_workspace/dataset

# Download everything
./setup_all.sh
```

**What it does:**
- Downloads PDM ground truths (~43 GB)
- Downloads DD3D backbone (~308 MB)
- Submits metric caching jobs

**Time:** ~2-3 hours for downloads, then 2-4 hours for cache jobs

### Step 3: Wait for Cache Jobs
```bash
# Check status
squeue -u ah7072

# Monitor logs
tail -f /scratch/ah7072/navsim_workspace/exp/logs/cache_metrics_*.out
```

### Step 4: Run Smoke Test
```bash
cd /scratch/ah7072/GTRS/scripts
sbatch torch_smoke_test.slurm
```

**Time:** ~30 minutes

### Step 5: Full Training
```bash
cd /scratch/ah7072/GTRS/scripts
./launch_all_torch.sh
```

**Time:** ~48 hours per experiment

---

## 📝 Environment Setup Details

### Conda Environment
The scripts automatically try to activate these environments (in order):
1. `navsim` (recommended name)
2. `conda_gtrs` (alternative from GTRS docs)

Create it once with:
```bash
cd /scratch/ah7072/navsim_workspace/navsim
conda env create -n navsim -f environment.yml
```

### Required Packages (from environment.yml)
- PyTorch with CUDA support
- nuPlan devkit
- NAVSIM dependencies
- Hydra for configs
- Lightning for training

---

## 🔍 Verifying Setup

### 1. Check Conda Environment
```bash
conda activate navsim
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

**Expected output:**
```
PyTorch: 2.x.x
CUDA available: True
```

### 2. Check Downloads
```bash
# PDM ground truths
ls -lh /scratch/ah7072/navsim_workspace/dataset/traj_pdm_v2/ori/
# Should show: navtrain_8192.pkl (14GB), navtrain_16384.pkl (28GB)

# Backbone
ls -lh /scratch/ah7072/navsim_workspace/dataset/models/
# Should show: dd3d_det_final.pth (308MB)
```

### 3. Check Cache (after jobs complete)
```bash
ls /scratch/ah7072/navsim_workspace/exp/metric_cache/
# Should show: navmini/, navtest/

# Check contents
ls /scratch/ah7072/navsim_workspace/exp/metric_cache/navtest/ | wc -l
# Should show: number of cached scenes
```

---

## 🐛 Troubleshooting

### "conda: command not found"
```bash
# Load Anaconda module
module load anaconda3
# Or use full path
/share/apps/anaconda3/bin/conda activate navsim
```

### "navsim environment not found"
```bash
# Create it first
cd /scratch/ah7072/GTRS/scripts
./setup_conda_env.sh
```

### "CUDA not available" in Python
```bash
# Check CUDA module
module list  # Should show cuda

# Check GPU access
nvidia-smi  # Should show GPUs

# If nvidia-smi works but torch.cuda.is_available() is False:
# - Reinstall PyTorch with correct CUDA version
# - Check PyTorch was installed with CUDA support
```

### Metric cache job fails
```bash
# Check error log
cat /scratch/ah7072/navsim_workspace/exp/logs/cache_metrics_*.err

# Common issues:
# 1. Conda environment not activated → Run setup_conda_env.sh first
# 2. Missing data → Check OPENSCENE_DATA_ROOT and data downloads
# 3. Memory issues → Job should auto-request 80GB, check with scontrol show job <ID>
```

---

## 📊 Disk Space Requirements

| Item | Size | Location |
|------|------|----------|
| Conda environment | ~5-10 GB | `~/.conda/envs/navsim/` |
| PDM GT (8K) | ~14 GB | `dataset/traj_pdm_v2/ori/` |
| PDM GT (16K) | ~28 GB | `dataset/traj_pdm_v2/ori/` |
| Backbone | ~308 MB | `dataset/models/` |
| Metric cache (navmini) | ~50 MB | `exp/metric_cache/navmini/` |
| Metric cache (navtest) | ~200 MB | `exp/metric_cache/navtest/` |
| Training checkpoints | ~500 MB each | `exp/hydra_*/checkpoints/` |

**Total for full setup:** ~50 GB

---

## ✅ Ready to Train Checklist

- [ ] Conda environment created and working
- [ ] PyTorch with CUDA available
- [ ] PDM ground truths downloaded (43 GB)
- [ ] DD3D backbone downloaded (308 MB)
- [ ] Metric cache jobs completed
- [ ] Smoke test passes

**Then you're ready to train!**

```bash
cd /scratch/ah7072/GTRS/scripts
./launch_all_torch.sh
```

---

## 📚 Updated File List

All scripts now run without containers:

- ✅ `setup_conda_env.sh` - NEW: Creates conda environment
- ✅ `setup_all.sh` - Downloads data + submits cache jobs
- ✅ `download_pdm_gt.sh` - Downloads PDM ground truths
- ✅ `download_backbones.sh` - Downloads pretrained backbone
- ✅ `torch_cache_metrics.slurm` - Caches evaluation metrics (FIXED)
- ✅ `torch_smoke_test.slurm` - Quick validation (FIXED)
- ✅ `torch_train_pdm.slurm` - PDM-only training (FIXED)
- ✅ `torch_train_w.slurm` - Weighted training (FIXED)
- ✅ `torch_train_w_ep.slurm` - Weighted + EP training (FIXED)
- ✅ `torch_eval.slurm` - Evaluation (FIXED)
- ✅ `launch_all_torch.sh` - Launches all experiments
- ✅ `launch_eval_torch.sh` - Evaluates all models

---

## 🎯 Next Steps

1. **Create conda environment:**
   ```bash
   cd /scratch/ah7072/GTRS/scripts
   ./setup_conda_env.sh
   ```

2. **Download data:**
   ```bash
   export OPENSCENE_DATA_ROOT=/scratch/ah7072/navsim_workspace/dataset
   ./setup_all.sh
   ```

3. **Wait for cache jobs, then train:**
   ```bash
   sbatch torch_smoke_test.slurm  # Test first
   ./launch_all_torch.sh           # Full training
   ```
