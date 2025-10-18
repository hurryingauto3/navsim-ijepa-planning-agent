# Complete Setup Guide for NAVSIM on Torch HPC

## Understanding the Problem

Torch HPC has these constraints:
- **No shared filesystem with Greene** - must manually transfer data
- **Container-based setup strongly recommended** - host OS will be upgraded
- **Writable overlays NOT reliable** - can't use overlay for conda in containers
- **No srun for GPU jobs** - must use sbatch

## ✅ Recommended Solution: Conda with Module System

### Step 1: Connect to Torch

```bash
# From your local machine
ssh greene.hpc.nyu.edu

# From Greene, connect to Torch
ssh cs649
```

### Step 2: Check Available Modules

```bash
# Check what's available
module avail

# Look specifically for Python/Conda
module avail 2>&1 | grep -iE "python|conda|anaconda|miniconda"
```

**If modules are available**, proceed to Step 3.  
**If no modules**, jump to Alternative Approaches below.

### Step 3: Load Conda Module and Create Environment

```bash
```bash
# If anaconda/miniconda is available:
module load anaconda3/2025.06  # Use full version

# Create NAVSIM environment
cd /scratch/ah7072/navsim_workspace/navsim
conda env create -n navsim -f environment.yml
```

### Step 4: Verify Installation

```bash
# Activate environment
conda activate navsim

# Test PyTorch with CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Expected output:
# PyTorch: 2.x.x
# CUDA available: True
```

### Step 5: Make Module Load Persistent

Add to your `~/.bashrc` on Torch:

```bash
# On Torch (cs649)
cat >> ~/.bashrc << 'EOF'

# NAVSIM environment setup
if [[ $(hostname) == *"cs649"* ]]; then
    module load anaconda3/2025.06 2>/dev/null || true
    export NAVSIM_DEVKIT_ROOT="/scratch/ah7072/navsim_workspace/navsim"
    export OPENSCENE_DATA_ROOT="/scratch/ah7072/navsim_workspace/dataset"
    export NUPLAN_MAPS_ROOT="/scratch/ah7072/navsim_workspace/dataset/maps"
    export NAVSIM_EXP_ROOT="/scratch/ah7072/navsim_workspace/exp"
    export APPTAINER_BINDPATH=/scratch,/state/partition1,/mnt,/share/apps
fi
EOF

# Reload
source ~/.bashrc
```

### Step 6: Download Prerequisites

```bash
cd /scratch/ah7072/GTRS/scripts

# Download PDM ground truths and backbones
./download_pdm_gt.sh
./download_backbones.sh

# Submit metric caching jobs
sbatch torch_cache_metrics.slurm navmini
sbatch torch_cache_metrics.slurm navtest

# Monitor
squeue -u ah7072
```

### Step 7: Run Smoke Test

```bash
cd /scratch/ah7072/GTRS/scripts
sbatch torch_smoke_test.slurm

# Check logs
tail -f /scratch/ah7072/navsim_workspace/exp/logs/smoke_*.out
```

### Step 8: Launch Training

```bash
# Once smoke test passes
cd /scratch/ah7072/GTRS/scripts
./launch_all_torch.sh
```

---

## Alternative Approaches (If No Modules)

### Alternative A: Transfer from Greene

If you have a working setup on Greene:

```bash
# On Greene
conda pack -n navsim -o /scratch/ah7072/navsim_env.tar.gz

# Transfer to Torch (from Greene)
scp /scratch/ah7072/navsim_env.tar.gz cs649:/scratch/ah7072/

# On Torch
mkdir -p ~/.conda/envs/navsim
cd ~/.conda/envs/navsim
tar -xzf /scratch/ah7072/navsim_env.tar.gz
source bin/activate
conda-unpack

# Add to PATH
echo 'export PATH="$HOME/.conda/envs/navsim/bin:$PATH"' >> ~/.bashrc
```

### Alternative B: Check Greene Home Access

```bash
# On Torch, test if Greene home is accessible
ls /gpfsnyu/home/$USER/.conda

# If this works, you can use Greene's conda directly!
# Add to ~/.bashrc on Torch:
export PATH="/gpfsnyu/home/$USER/.conda/envs/navsim/bin:$PATH"
```

### Alternative C: Build Custom Container (Most Robust)

This requires a machine with sudo (not Torch itself):

1. Use the provided `build_navsim_container.sh` script
2. Build on a machine with sudo: `sudo singularity build navsim.sif navsim.def`
3. Transfer to Torch: `scp navsim.sif cs649:/scratch/ah7072/`
4. Update scripts to use: `/share/apps/apptainer/bin/singularity exec --nv /scratch/ah7072/navsim.sif`

---

## Troubleshooting

### Error: "conda: command not found"

**Solution**: No module system available. Use Alternative A or C above.

### Error: "CUDA not available"

**Check**:
```bash
nvidia-smi  # Should show GPUs on compute node (not login node)
```

**Solution**: This is normal on login node. Submit a GPU job to test:
```bash
sbatch --gres=gpu:1 --constraint="h200|l40s" --time=00:10:00 \
  --wrap "conda activate navsim && python -c 'import torch; print(torch.cuda.is_available())'"
```

### Error: "FATAL: could not open image"

**Cause**: Scripts trying to use non-existent container

**Solution**: Already fixed in updated scripts. If you see this:
1. Pull latest scripts: `git pull origin master`
2. Or manually update the Slurm script to use conda activation instead

### Error: "No space left on device"

**Check quota**:
```bash
df -h /scratch/ah7072
```

**Solution**: Clean up old experiments or request quota increase

---

## Key Differences from Greene

| Aspect | Greene | Torch |
|--------|--------|-------|
| Filesystem | Shared with archive | **Isolated - manual transfer** |
| Containers | Optional | **Strongly recommended** |
| Python setup | Can use host | **Use containers or modules** |
| GPU access | srun or sbatch | **sbatch only** |
| Job preemption | Rare | **Common on h200/l40s partitions** |

---

## Data Transfer Between Greene and Torch

### From Greene to Torch

```bash
# On Greene
scp -rp /scratch/ah7072/some_data cs649:/scratch/ah7072/

# Or for large transfers, use dtn-1
scp -rp dtn-1:/scratch/ah7072/some_data cs649:/scratch/ah7072/
```

### From Torch to Greene

```bash
# On Torch
scp -rp /scratch/ah7072/results dtn-1:/scratch/ah7072/

# For archival
scp -rp /scratch/ah7072/results dtn-1:/archive/ah7072/
```

---

## Checkpoint and Restart Strategy

Since Torch uses preemptible partitions, always enable checkpointing:

```python
# In Lightning training code (already in NAVSIM)
trainer = Trainer(
    callbacks=[
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='checkpoint-{epoch:02d}',
            save_top_k=3,
            monitor='val_loss',
        )
    ]
)

# In Slurm script
#SBATCH --comment="preemption=yes;requeue=yes"
```

When job is preempted, it will automatically restart from last checkpoint.

---

## Quick Commands Reference

```bash
# Connect to Torch
ssh greene.hpc.nyu.edu
ssh cs649

# Check job status
squeue -u ah7072

# Check logs
tail -f /scratch/ah7072/navsim_workspace/exp/logs/*.out

# Cancel job
scancel <job_id>

# Check GPU availability
sinfo -p h200,l40s

# Interactive session (CPU only, no GPU support for interactive)
srun --pty --time=01:00:00 bash
```

---

## Next Steps After Setup

1. ✅ **Verify conda environment works**
2. ✅ **Download prerequisites** (PDM GT, backbones, metric cache)
3. ✅ **Run smoke test** (1 epoch, ~30 min)
4. ✅ **Launch full training** (3 experiments, ~48h each)
5. ✅ **Evaluate on navtest** (use cached metrics)
6. ✅ **Transfer results back to Greene for analysis**

---

## Getting Help

- **HPC support**: hpc@nyu.edu
- **Torch docs**: [NYU HPC Torch Page](https://sites.google.com/nyu.edu/nyu-hpc/hpc-systems/torch)
- **NAVSIM issues**: Check `GTRS/docs/` or GitHub issues
