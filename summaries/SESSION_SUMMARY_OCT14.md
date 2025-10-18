# Torch HPC NAVSIM Setup - Session Summary
**Date:** October 14, 2025  
**User:** ah7072  
**Cluster:** Torch HPC (cs649 partition)

---

## Session Overview

Successfully configured and deployed NAVSIM (Hydra-MDP autonomous driving model) on NYU Torch HPC cluster. Resolved container compatibility issues, created conda environment, installed all dependencies, and initiated metric caching as prerequisite for training experiments.

---

## Problems Encountered & Solutions

### 1. Container Not Available on Torch
**Problem:**
- Scripts referenced `/share/apps/images/pytorch-24.09.sif` container
- Container doesn't exist on Torch HPC (different from Greene)
- Torch has separate filesystem and module system

**Solution:**
- Removed all container dependencies from Slurm scripts
- Switched to conda-based workflow using `anaconda3/2025.06` module
- Updated 8 different scripts to use proper module loading

**Files Modified:**
- `scripts/torch_cache_metrics.slurm`
- `scripts/torch_smoke_test.slurm`
- `scripts/torch_train_pdm.slurm`
- `scripts/torch_train_w.slurm`
- `scripts/torch_train_w_ep.slurm`
- `scripts/torch_eval.slurm`
- `scripts/setup_torch_navsim.sh`

---

### 2. Module Version Specification Required
**Problem:**
- Generic `module load anaconda3` failed
- Torch requires explicit version specification

**Solution:**
- Updated all scripts to use `module load anaconda3/2025.06`
- Verified module availability with `module avail anaconda3`

---

### 3. Conda Environment Creation Network Timeout
**Problem:**
- Initial environment creation hit network timeout during dependency resolution
- Large dependency tree with PyTorch, Ray, nuplan-devkit, etc.

**Solution:**
- Retried with `conda env update --file environment.yml`
- Successfully installed all 200+ dependencies
- Environment created at `/home/ah7072/.conda/envs/navsim`

**Key Packages Installed:**
- Python 3.9.23
- PyTorch 2.8.0+cu128 (CUDA 12.8 support)
- pytorch-lightning 2.2.1
- nuplan-devkit v1.2
- Ray 2.50.0
- xformers 0.0.32
- tensorboard 2.16.2
- All geometric/GIS libraries (geopandas, shapely, fiona, rasterio)

---

### 4. NAVSIM Package Import Error
**Problem:**
- Metric caching jobs failed immediately with `ModuleNotFoundError: No module named 'navsim'`
- Jobs 259993 and 259994 completed in seconds with error
- NAVSIM source code not installed as Python package

**Solution:**
- Installed NAVSIM in development mode:
  ```bash
  cd /scratch/ah7072/navsim_workspace/navsim
  pip install -e .
  ```
- Package now importable from any working directory
- Resubmitted metric caching jobs successfully

---

## Environment Configuration

### Module Setup
```bash
module load anaconda3/2025.06
eval "$(conda shell.bash hook)"
conda activate navsim
```

### Environment Variables
```bash
# Data locations
export OPENSCENE_DATA_ROOT="/scratch/ah7072/data/openscene"
export NUPLAN_MAPS_ROOT="/scratch/ah7072/data/maps"
export NAVSIM_EXP_ROOT="/scratch/ah7072/navsim_workspace/exp"

# Repository root
export NAVSIM_DEVKIT_ROOT="/scratch/ah7072/navsim_workspace/navsim"
```

### Directory Structure
```
/scratch/ah7072/
├── data/
│   ├── openscene/          # Sensor data and logs
│   │   ├── mini_navsim_logs/
│   │   ├── mini_sensor_blobs/
│   │   ├── test_navsim_logs/
│   │   ├── test_sensor_blobs/
│   │   ├── trainval_navsim_logs/
│   │   └── trainval_sensor_blobs/
│   └── maps/               # nuPlan map data
│       ├── sg-one-north/
│       ├── us-ma-boston/
│       ├── us-nv-las-vegas-strip/
│       └── us-pa-pittsburgh-hazelwood/
├── navsim_workspace/
│   ├── navsim/             # NAVSIM source code (installed as package)
│   ├── dataset/            # Symlink to data/openscene
│   └── exp/                # Experiments and logs
│       ├── logs/           # Slurm job outputs
│       └── metric_cache/   # Pre-computed metrics (being generated)
├── GTRS/                   # Original repo (for reference)
└── scripts/                # Slurm job scripts
```

---

## Jobs Executed

### Failed Attempts (Jobs 259993-259994)
- **Job 259993:** cache_metrics_navmini - FAILED (ModuleNotFoundError)
- **Job 259994:** cache_metrics_navtest - FAILED (ModuleNotFoundError)
- **Error:** `ModuleNotFoundError: No module named 'navsim'`
- **Duration:** <1 minute (immediate failure)

### Successful Jobs (Jobs 259996-259997)
- **Job 259996:** cache_metrics_navmini - RUNNING on cs604
- **Job 259997:** cache_metrics_navtest - RUNNING on cs604
- **Status:** Computing metric cache after NAVSIM package installation
- **Command:**
  ```bash
  python navsim/planning/script/run_metric_caching.py \
    split=navmini/navtest \
    cache_path=/scratch/ah7072/navsim_workspace/exp/metric_cache
  ```

---

## Verification Steps Completed

1. ✅ **Module Availability:** Confirmed `anaconda3/2025.06` exists on Torch
2. ✅ **Conda Environment:** Created and activated successfully
3. ✅ **PyTorch + CUDA:** Installed PyTorch 2.8.0 with CUDA 12.8 support
4. ✅ **Package Installation:** All dependencies resolved without conflicts
5. ✅ **NAVSIM Package:** Installed in development mode, importable
6. ✅ **Data Access:** Verified dataset and map files accessible
7. ✅ **Metric Caching:** Jobs submitted and running on GPU nodes

---

## Current Status

### ✅ Completed
- Torch HPC environment fully configured
- All Slurm scripts updated for Torch compatibility
- Conda environment created with all dependencies
- NAVSIM package installed and verified
- Metric caching jobs running (navmini + navtest)

### 🔄 In Progress
- **Job 259996:** Metric caching for `navmini` split (running on cs604)
- **Job 259997:** Metric caching for `navtest` split (running on cs604)
- Expected duration: ~5-15 minutes per split

### ⏳ Pending
1. Wait for metric caching to complete
2. Run smoke test to verify GPU training works
3. Launch full training experiments (3 variants)

---

## Next Steps

### 1. Monitor Metric Caching
```bash
# Check job status
squeue -u ah7072

# Monitor output logs
tail -f /scratch/ah7072/navsim_workspace/exp/logs/cache_metrics_259996.out
tail -f /scratch/ah7072/navsim_workspace/exp/logs/cache_metrics_259997.out

# Check for completion
ls -lh /scratch/ah7072/navsim_workspace/exp/metric_cache/
```

### 2. Run Smoke Test (After Caching Completes)
```bash
cd /scratch/ah7072
sbatch scripts/torch_smoke_test.slurm
```

**Smoke test will:**
- Use 1 GPU for 1 epoch
- Train on small subset (10 batches)
- Validate gradient flow and loss computation
- Verify checkpoint saving works
- Expected duration: ~10-15 minutes

### 3. Launch Full Training (After Smoke Test Passes)
```bash
cd /scratch/ah7072
./scripts/launch_all_torch.sh
```

**Three training experiments:**
1. **PDM-only baseline** (`torch_train_pdm.slurm`)
   - 4 GPUs, 40 epochs
   - Batch size 32
   - PDM score metric only

2. **Weighted combination** (`torch_train_w.slurm`)
   - 4 GPUs, 40 epochs
   - All PDM sub-metrics with learned weights

3. **Weighted + EP** (`torch_train_w_ep.slurm`)
   - 4 GPUs, 40 epochs
   - Best-performing variant from paper
   - Expert prediction + weighted scoring

### 4. Evaluation (After Training)
```bash
# Evaluate on test split
sbatch scripts/torch_eval.slurm <experiment_name> <checkpoint_path>
```

---

## Key Commands Reference

### Environment Activation
```bash
module load anaconda3/2025.06
eval "$(conda shell.bash hook)"
conda activate navsim
```

### Job Management
```bash
# Submit job
sbatch scripts/torch_<job_name>.slurm

# Check queue
squeue -u ah7072

# Check job details
scontrol show job <JOBID>

# Cancel job
scancel <JOBID>

# View output logs
tail -f /scratch/ah7072/navsim_workspace/exp/logs/<job_name>_<JOBID>.out
tail -f /scratch/ah7072/navsim_workspace/exp/logs/<job_name>_<JOBID>.err
```

### Python Environment
```bash
# Verify installation
python -c "import navsim; print('✓ navsim imported successfully')"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# List installed packages
conda list | grep -E "torch|lightning|navsim|ray"
```

---

## System Information

### Compute Node Details
- **Partition:** cs649 (cs604 node currently running jobs)
- **GPU Driver:** CUDA 580.82.07
- **OS:** RHEL 9.6 (Plow)
- **Python:** 3.9.23 (via anaconda3/2025.06)
- **CUDA Toolkit:** 12.8 (via PyTorch package)

### Resource Allocation
- **Metric Caching:** 1 GPU, 8 CPUs, 32GB RAM, 2 hours time limit
- **Smoke Test:** 1 GPU, 8 CPUs, 32GB RAM, 1 hour time limit  
- **Training Jobs:** 4 GPUs, 32 CPUs, 128GB RAM, 48 hours time limit
- **Evaluation:** 1 GPU, 8 CPUs, 32GB RAM, 6 hours time limit

---

## Troubleshooting Notes

### If Jobs Fail with Import Errors
```bash
# Verify navsim package is installed
conda activate navsim
python -c "import navsim"

# Reinstall if needed
cd /scratch/ah7072/navsim_workspace/navsim
pip install -e .
```

### If CUDA Not Available
- **On login nodes:** CUDA will show as unavailable (expected)
- **On compute nodes:** CUDA should be available when GPU is allocated
- Check with: `nvidia-smi` (only works on GPU nodes)

### If Module Not Found
```bash
# Check available modules
module avail anaconda3

# Load specific version
module load anaconda3/2025.06
```

### If Disk Space Issues
```bash
# Check scratch usage
du -sh /scratch/ah7072/*

# Clean conda cache
conda clean --all

# Clean pip cache
pip cache purge
```

---

## Documentation Created

1. **TORCH_SETUP_GUIDE.md** - Complete setup walkthrough
2. **TORCH_EXPLAINED.md** - Torch vs Greene differences
3. **check_torch_setup.sh** - Automated environment checker
4. **setup_torch_navsim.sh** - One-click setup automation
5. **SETUP_COMPLETE.md** - Initial status after environment creation
6. **SESSION_SUMMARY_OCT14.md** - This document (complete session summary)

---

## Success Metrics

✅ **Environment Setup:** 100% complete  
✅ **Dependency Installation:** All 200+ packages installed  
✅ **Script Updates:** 8 scripts modernized for Torch  
✅ **Package Installation:** NAVSIM installed and verified  
✅ **Job Submission:** Metric caching jobs running successfully  

**Time to First Running Job:** ~45 minutes (from initial errors to running jobs)

---

## Lessons Learned

1. **Module Versions Matter:** Torch requires explicit version specification
2. **Container Availability:** Can't assume containers exist across clusters
3. **Package Installation:** Framework code needs `pip install -e .` for imports
4. **Conda Environments:** Network timeouts can be resolved with retry + update
5. **Job Monitoring:** Early failures save time - fix fast and resubmit

---

## Timeline of Events

1. **Initial Problem:** Container errors on Torch HPC
2. **Diagnosis:** Identified module system differences, no container support
3. **Environment Creation:** Built conda environment with all dependencies
4. **Script Updates:** Modified 8 Slurm scripts for Torch compatibility
5. **First Caching Attempt:** Jobs 259993-259994 failed with import error
6. **Package Installation:** Installed NAVSIM with `pip install -e .`
7. **Success:** Jobs 259996-259997 running metric caching
8. **Current State:** Ready for smoke test and training experiments

---

## Contact & Resources

- **User:** ah7072@nyu.edu
- **Cluster:** Torch HPC (cs649 partition)
- **Documentation:** `/scratch/ah7072/TORCH_*.md` and `/scratch/ah7072/SESSION_*.md`
- **Scripts:** `/scratch/ah7072/scripts/`
- **Experiments:** `/scratch/ah7072/navsim_workspace/exp/`
- **NAVSIM Repo:** `/scratch/ah7072/navsim_workspace/navsim/`

---

**Generated:** October 14, 2025  
**Status:** ✅ Environment ready, 🔄 metric caching in progress, ⏳ ready for training experiments
