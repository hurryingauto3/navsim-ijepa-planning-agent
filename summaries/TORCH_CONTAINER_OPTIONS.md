# Torch Container Solutions for NAVSIM

## Problem
- Torch strongly recommends containers
- Writable overlays (for conda) are NOT reliable
- System installs will break when OS upgrades

## Solution Options (Ranked by Simplicity)

### ⭐ Option 1: Use Module System (SIMPLEST - Try This First)
Torch may have modules available similar to Greene:

```bash
# Check available modules
module avail

# Look for Python/Anaconda/PyTorch modules
module spider python
module spider anaconda
module spider pytorch

# If available, load and use
module load anaconda3
conda activate navsim  # or create new env
```

**Scripts:** Already updated to work with module-loaded conda

---

### Option 2: Copy Container from Greene
If you have a working setup on Greene:

```bash
# On Greene: Pack your conda environment
conda pack -n navsim -o navsim_env.tar.gz

# Transfer to Torch
scp navsim_env.tar.gz cs649:/scratch/$USER/

# On Torch: Unpack
mkdir -p ~/.conda/envs/navsim
cd ~/.conda/envs/navsim
tar -xzf /scratch/$USER/navsim_env.tar.gz
source bin/activate
conda-unpack
```

---

### Option 3: Build Custom Container (MOST ROBUST)
Build on a machine with sudo, then transfer:

1. Use `build_navsim_container.sh` to create definition file
2. Build: `sudo singularity build navsim.sif navsim.def`
3. Transfer to Torch
4. Update scripts to use your container

---

### Option 4: Use Existing Container + Install at Runtime
Use Ubuntu container and install packages each job (SLOW):

```bash
singularity exec /share/apps/images/ubuntu-24.04.3.sif bash -c "
  pip install --user torch torchvision ...
  python your_script.py
"
```

---

## Recommended Approach

**Step 1:** Try module system first
```bash
ssh cs649
module avail 2>&1 | grep -i "python\|conda\|pytorch"
```

**Step 2:** If no modules, check Greene home accessibility
```bash
ls /gpfsnyu/home/$USER/.conda 2>&1
# If this works, you can use Greene conda directly!
```

**Step 3:** If neither works, we build a container

---

## Updated Scripts Status

All scripts in `/scratch/ah7072/GTRS/scripts/` are currently set to:
- Try activating conda environment
- Fall back gracefully if not found

They will work once ANY of the above options is set up.

---

## Quick Test

```bash
# Test if conda is available
which conda

# Test if Python/PyTorch work
python -c "import torch; print(torch.__version__)"

# If both work, you're ready!
cd /scratch/ah7072/GTRS/scripts
sbatch torch_smoke_test.slurm
```
