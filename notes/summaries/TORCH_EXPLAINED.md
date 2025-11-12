# Torch HPC vs Greene: Key Differences for NAVSIM

## Critical Understanding

Torch HPC is **NOT** a drop-in replacement for Greene. It has fundamentally different constraints:

## 🚨 Major Differences

| Aspect | Greene | Torch |
|--------|--------|-------|
| **Filesystem** | Shared `/scratch`, `/home`, `/archive` | **Isolated - manual transfer required** |
| **Python Setup** | Can install on host | **Must use containers or modules** |
| **Container Strategy** | Optional, writable overlays work | **Strongly recommended, overlays UNRELIABLE** |
| **Interactive GPU** | `srun --gres=gpu:1` works | **NOT supported - sbatch only** |
| **Job Preemption** | Rare | **Common on public partitions** |
| **OS Stability** | Stable RHEL | **Will upgrade to RHEL 10 - breaks host installs** |

## 🔧 How to Run NAVSIM on Torch

### The Problem You Hit

Your scripts tried to use:
1. A non-existent container: `/share/apps/images/pytorch-24.09.sif`
2. Direct `python` commands without environment setup
3. `source activate` (old conda syntax) without checking if conda exists

### The Solution

**Three-step process:**

#### 1. Set Up Conda Environment

```bash
# Connect to Torch
ssh greene.hpc.nyu.edu
ssh cs649

# Check available modules
module avail

# If anaconda/miniconda is available:
module load anaconda3  # or miniconda3

# Create NAVSIM environment
cd /scratch/ah7072/navsim_workspace/navsim
conda env create -n navsim -f environment.yml

# Test it
conda activate navsim
python -c "import torch; print(torch.cuda.is_available())"
```

**Alternative if no modules:** Transfer conda env from Greene (see TORCH_SETUP_GUIDE.md)

#### 2. Download Data and Prerequisites

```bash
# Set environment variables
export OPENSCENE_DATA_ROOT="/scratch/ah7072/navsim_workspace/dataset"
export NUPLAN_MAPS_ROOT="/scratch/ah7072/navsim_workspace/dataset/maps"
export NAVSIM_EXP_ROOT="/scratch/ah7072/navsim_workspace/exp"

# Download prerequisites
cd /scratch/ah7072/GTRS/scripts
./setup_all.sh

# This downloads:
# - PDM ground truths (~43 GB)
# - DD3D backbone (~308 MB)
# - Submits metric caching jobs
```

#### 3. Run Training

```bash
# Smoke test first (30 min)
cd /scratch/ah7072/GTRS/scripts
sbatch torch_smoke_test.slurm

# If successful, launch full training
./launch_all_torch.sh
```

## 📁 Data Transfer Workflow

Since Torch doesn't share filesystem with Greene:

### One-Time: Move Data to Torch

```bash
# On Greene
cd /scratch/ah7072

# Transfer to Torch
scp -rp navsim_workspace/dataset cs649:/scratch/ah7072/navsim_workspace/

# For large files, use dtn-1
scp -rp dtn-1:/scratch/ah7072/navsim_workspace/dataset cs649:/scratch/ah7072/navsim_workspace/
```

### Ongoing: Transfer Results Back

```bash
# On Torch - after training
cd /scratch/ah7072

# Transfer results to Greene
scp -rp navsim_workspace/exp/hydra_* dtn-1:/scratch/ah7072/results/

# For archival (permanent storage)
scp -rp navsim_workspace/exp/hydra_* dtn-1:/archive/ah7072/navsim_experiments/
```

## 🐛 Why Your Scripts Failed

### Error 1: Container Not Found
```
FATAL: could not open image /share/apps/images/pytorch-24.09.sif
```

**Cause:** Container doesn't exist on Torch  
**Fix:** Updated all scripts to use conda directly (no container)

### Error 2: Python Not Found
```
FATAL: "python": executable file not found in $PATH
```

**Cause:** Conda environment not activated  
**Fix:** Scripts now properly activate conda with error checking

### Error 3: Old Conda Syntax
```bash
source activate navsim  # Old syntax, doesn't work reliably
```

**Fix:** Now using:
```bash
eval "$(conda shell.bash hook)"
conda activate navsim
```

## ✅ What I Fixed

1. **Updated all Slurm scripts** (`torch_cache_metrics.slurm`, `torch_smoke_test.slurm`, etc.)
   - Removed container dependencies
   - Added proper conda activation with error checking
   - Added module loading for conda

2. **Created comprehensive guides:**
   - `TORCH_SETUP_GUIDE.md` - Complete setup instructions
   - `check_torch_setup.sh` - Automated environment checker

3. **Updated activation pattern** in all scripts:
   ```bash
   # Load module if available
   if command -v module &> /dev/null; then
       module load anaconda3 2>/dev/null || true
   fi
   
   # Activate conda with error checking
   if command -v conda &> /dev/null; then
       eval "$(conda shell.bash hook)"
       conda activate navsim || { echo "ERROR: Create environment first"; exit 1; }
   else
       echo "ERROR: conda not found!"; exit 1
   fi
   ```

## 🎯 Your Action Items

### Immediate (Do Now)

1. **Check what's available on Torch:**
   ```bash
   ssh cs649
   /scratch/ah7072/GTRS/scripts/check_torch_setup.sh
   ```

2. **Set up conda environment** (choose based on what's available):
   - **Option A:** Use module system (if available)
   - **Option B:** Transfer from Greene (if you have it there)
   - **Option C:** Build custom container (most complex, most robust)

3. **Download data if needed:**
   ```bash
   cd /scratch/ah7072/data
   ./download_quick_start.sh
   ```

### Next Steps (After Setup)

4. **Download training prerequisites:**
   ```bash
   cd /scratch/ah7072/GTRS/scripts
   ./setup_all.sh
   ```

5. **Run smoke test:**
   ```bash
   sbatch torch_smoke_test.slurm
   ```

6. **Launch training:**
   ```bash
   ./launch_all_torch.sh
   ```

## 📚 Documentation Updated

- ✅ `TORCH_SETUP_GUIDE.md` - Complete setup walkthrough
- ✅ `check_torch_setup.sh` - Automated checker script
- ✅ All `.slurm` scripts updated with proper conda activation
- ✅ Error handling added to catch setup issues early

## 🔍 Quick Diagnosis

Run this to see your current status:
```bash
ssh cs649
/scratch/ah7072/GTRS/scripts/check_torch_setup.sh
```

This will check:
- ✓ Module system availability
- ✓ Conda installation
- ✓ NAVSIM repository
- ✓ Data availability
- ✓ Training prerequisites
- ✓ Provide specific next steps

## 💡 Key Takeaway

**Torch requires explicit environment setup before running any Python code.**

The scripts now handle this automatically, but you need to create the conda environment once first. After that, all scripts will work as intended.
