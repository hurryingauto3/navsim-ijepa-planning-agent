# NYU Torch Cluster Setup for NAVSIM - Status & Plan

**Date**: October 12, 2025  
**User**: ah7072 (Ali Hamza)  
**Cluster**: NYU Torch (cs649)

---

## 🎉 EXCELLENT NEWS - No Storage Limitations!

### Current Torch `/scratch` Status

```
Storage Capacity:  21 Petabytes (21,000 TB)
Storage Used:      547 TB (3% full)
Storage Available: 20+ Petabytes

Inode Capacity:    94.3 BILLION inodes
Inodes Used:       159 million (0.17% used)
Inodes Available:  94+ billion
```

### What This Means for NAVSIM

✅ **NO INODE QUOTA** - You can store millions of small files without limits  
✅ **NO STORAGE QUOTA** (yet) - HPC said quotas will be applied "soon" but not yet  
✅ **VAST Filesystem** - Optimized for AI/ML workloads with random I/O (perfect for NAVSIM)  
✅ **Better Performance** - Torch's VAST is faster than Greene for dataloader workloads  

**Bottom line**: You can download the FULL NAVSIM dataset without ANY restrictions right now.

---

## NAVSIM Dataset Requirements vs. Torch Capacity

| Dataset Split | Storage | Files | Torch Limit | Status |
|--------------|---------|-------|-------------|--------|
| `navmini` | 152 GB | ~1M | 21 PB, 94B inodes | ✅ Trivial |
| `navtrain` | 459 GB | ~3M | 21 PB, 94B inodes | ✅ Easy |
| Full `trainval` | 2 TB | ~10M | 21 PB, 94B inodes | ✅ No problem |

**All NAVSIM datasets are completely feasible on Torch.**

---

## Your Current Torch Setup

```
Location: /scratch/ah7072/
Current Usage: 101 MB (just your repos)
Directories:
  - GTRS/      (NAVSIM fork - training code)
  - navsim/    (original repo?)
  - .github/   (copilot instructions)
```

---

## Recommended Action Plan

### Option A: Native Filesystem (RECOMMENDED - Simplest)

This is the standard approach, requires no special knowledge, and works with NAVSIM out of the box.

#### Step 1: Setup Directory Structure

```bash
# Create NAVSIM data directories on Torch
cd /scratch/ah7072
mkdir -p openscene/sensor_blobs
mkdir -p maps
mkdir -p experiments

# Set environment variables (add to ~/.bashrc on Torch)
export NAVSIM_DEVKIT_ROOT="$HOME/GTRS"
export OPENSCENE_DATA_ROOT="/scratch/$USER/openscene"
export NUPLAN_MAPS_ROOT="/scratch/$USER/maps"
export NAVSIM_EXP_ROOT="/scratch/$USER/experiments"
```

#### Step 2: Download Maps First (Small, Required)

```bash
cd /scratch/ah7072/maps
wget -O nuplan-maps-v1.1.zip https://navsim.s3.eu-central-1.amazonaws.com/nuplan-maps-v1.1.zip
unzip nuplan-maps-v1.1.zip
rm nuplan-maps-v1.1.zip  # Clean up after extraction
```

**Size**: ~9 GB  
**Files**: ~100  
**Time**: ~5-10 minutes

#### Step 3: Download navmini (Test Dataset)

```bash
cd /scratch/ah7072/openscene

# Get the download script from your GTRS repo
cp $HOME/GTRS/download/download_mini.sh .

# Run download (this will take ~2-3 hours)
bash download_mini.sh
```

**What it downloads**:
- Metadata: ~1 GB
- Sensor blobs: ~151 GB
- **Total**: ~152 GB, ~1M files
- **Purpose**: Test your entire workflow before committing to full download

#### Step 4: Test Training on navmini

```bash
# Submit a quick smoke test job
sbatch --gres=gpu:1 --constraint="l40s|h200" \
  --nodes=1 --cpus-per-task=8 --mem=64GB --time=02:00:00 \
  --wrap "cd $NAVSIM_DEVKIT_ROOT && \
          python navsim/planning/script/run_training_dense.py \
          agent=ego_status_mlp \
          split=mini \
          trainer.max_epochs=1 \
          trainer.limit_train_batches=10"
```

**If this works**, you've validated the entire pipeline and can proceed to full download.

#### Step 5: Download navtrain (Full Training Split)

```bash
cd /scratch/ah7072/openscene

# Get the AWS download script (fastest option)
cp $HOME/GTRS/download/download_navtrain_aws.sh .

# Run download (this will take 8-12 hours depending on network)
bash download_navtrain_aws.sh
```

**What it downloads**:
- Metadata: ~14 GB (openscene logs)
- Sensor blobs: ~445 GB (millions of camera/LiDAR files)
- **Total**: ~459 GB, ~3M files

**Note**: Script automatically:
1. Downloads tarballs in chunks
2. Extracts each chunk
3. **Deletes tarballs immediately** to save space
4. Only final extracted files remain

#### Step 6: Run Full Training

```bash
# Your thesis baseline experiment
sbatch --gres=gpu:1 --constraint="l40s|h200" \
  --nodes=1 --cpus-per-task=8 --mem=64GB --time=48:00:00 \
  --comment="preemption=yes;requeue=yes" \
  --wrap "cd $NAVSIM_DEVKIT_ROOT && \
          python navsim/planning/script/run_training_dense.py \
          agent=ego_status_mlp \
          experiment_name=baseline_test_$(date +%Y%m%d) \
          trainer.max_epochs=50 \
          data.batch_size=32 \
          data.num_workers=4"
```

---

### Option B: Singularity/Apptainer Containers (ADVANCED - Optional)

**What is Singularity/Apptainer?**
- Container system (like Docker, but for HPC)
- Lets you package your entire software environment
- Useful if you need specific Python/CUDA versions

**When to use**:
- If you need a specific environment that conflicts with Torch's system
- If you want reproducibility across different systems
- If you're having Python dependency issues

**When NOT to use**:
- You just want to run NAVSIM training (native setup is simpler)
- You're new to containers (steep learning curve)
- Your conda/pip environment works fine

**Status**: You don't need this right now. Start with native setup (Option A).

If you later decide you want containers, the Torch docs mention:
```bash
# Singularity executable location on Torch
/share/apps/apptainer/bin/singularity

# Pre-built images available at
/share/apps/images/
```

But **I strongly recommend ignoring this for now** and using native Python/conda.

---

## Environment Setup on Torch

### Python Environment

You have two options:

#### Option 1: System Python + venv (Simplest)

```bash
# Check Python version on Torch
python3 --version

# Create virtual environment
cd $HOME
python3 -m venv navsim_env
source navsim_env/bin/activate

# Install NAVSIM requirements
cd $HOME/GTRS
pip install --upgrade pip
pip install -e .
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### Option 2: Conda/Mamba (If you prefer conda)

```bash
# Install Miniforge on Torch (if not already)
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p $HOME/miniforge3

# Create NAVSIM environment
source $HOME/miniforge3/bin/activate
cd $HOME/GTRS
mamba env create -f environment.yml
mamba activate navsim
```

### Add to ~/.bashrc on Torch

```bash
# Add these lines to automatically set up environment on login
echo '
# NAVSIM Environment Setup
export NAVSIM_DEVKIT_ROOT="$HOME/GTRS"
export OPENSCENE_DATA_ROOT="/scratch/$USER/openscene"
export NUPLAN_MAPS_ROOT="/scratch/$USER/maps"
export NAVSIM_EXP_ROOT="/scratch/$USER/experiments"

# Activate Python environment
source $HOME/navsim_env/bin/activate  # or: source $HOME/miniforge3/bin/activate

# GPU settings
export CUDA_VISIBLE_DEVICES=0
' >> ~/.bashrc

source ~/.bashrc
```

---

## Key Differences: Greene vs. Torch

| Aspect | Greene (Old) | Torch (New) | Winner |
|--------|-------------|-------------|--------|
| Inode Limit | 1M (hit limit) | 94+ billion | 🏆 Torch |
| Storage Quota | 5 TB | No quota yet | 🏆 Torch |
| Filesystem | Standard NFS | VAST (AI-optimized) | 🏆 Torch |
| Random I/O | Slower | Much faster | 🏆 Torch |
| GPU Options | V100, A100 | L40S (48GB), H200 (141GB) | 🏆 Torch |
| Job Limits | Same | Same (24 GPU × 48h) | Tie |
| Stability | Production | Beta (may preempt jobs) | ⚠️ Greene |
| File Transfer | N/A | Must copy from Greene | ⚠️ Torch |

**For NAVSIM workloads**: Torch is much better due to inode capacity and VAST I/O performance.

**Caveat**: Torch is in beta, so jobs may be preempted. Use `--comment="preemption=yes;requeue=yes"` and implement checkpointing.

---

## Checkpoint/Restart Setup (Required for Torch)

Since Torch can preempt jobs, you need checkpoint/restart capability.

NAVSIM's Lightning trainer already supports this:

```python
# Your training config should include:
trainer:
  default_root_dir: ${navsim_exp_root}/${experiment_name}
  enable_checkpointing: true
  callbacks:
    - class_path: lightning.pytorch.callbacks.ModelCheckpoint
      init_args:
        save_top_k: 3
        monitor: val_loss
        mode: min
        save_last: true  # Always save last checkpoint
```

When job is preempted and requeued, resume with:

```bash
python navsim/planning/script/run_training_dense.py \
  agent=your_agent \
  ckpt_path=/scratch/ah7072/experiments/your_run/checkpoints/last.ckpt
```

---

## File Transfer Between Greene and Torch

**Important**: Greene and Torch do NOT share filesystems.

### Copying Code/Small Files from Greene to Torch

```bash
# From Greene, copy to Torch
scp -rp /scratch/ah7072/.github ah7072@cs649:/scratch/ah7072/

# From Torch, pull from Greene
scp -rp ah7072@greene.hpc.nyu.edu:/scratch/ah7072/GTRS /scratch/ah7072/
```

### Downloading Datasets Directly on Torch (Recommended)

**Don't transfer NAVSIM data from Greene to Torch**. Instead, download directly on Torch:
- Faster (no double-hop through Greene)
- Saves Greene's inode quota
- Cleaner setup

---

## Timeline Estimate

### Conservative Timeline (Recommended)

| Task | Time | Total Elapsed |
|------|------|---------------|
| Setup directories & env | 30 min | 30 min |
| Download maps | 10 min | 40 min |
| Download navmini | 2-3 hours | 3-4 hours |
| Test training on navmini | 1 hour | 4-5 hours |
| Download navtrain | 8-12 hours | 12-17 hours |
| Full training run | 24-48 hours | 36-65 hours |

**You could have your first real training run completed within 2-3 days.**

### Aggressive Timeline (Parallel Download)

If you're confident and want to move fast:
1. Start navmini download in background
2. Start navtrain download in parallel (different terminal)
3. Test on navmini while navtrain downloads
4. Launch full training as soon as navtrain finishes

**Could have results in 24-36 hours** (but riskier if something breaks).

---

## Storage Quota Warning ⚠️

From HPC's email:
> "Currently, there are no storage quotas set up on Torch. However, we will be applying quotas soon."

**This means**:
- ✅ Download everything NOW while there are no quotas
- ✅ Your data will be grandfathered in (they won't delete existing data)
- ⚠️ Future downloads might hit quota limits

**Action**: Prioritize getting navtrain downloaded ASAP before quotas are enforced.

---

## Monitoring & Debugging

### Check Download Progress

```bash
# Watch disk usage grow
watch -n 60 'du -sh /scratch/$USER/openscene'

# Count files (slower, but informative)
watch -n 300 'find /scratch/$USER/openscene -type f | wc -l'
```

### Check Job Status

```bash
# List your jobs
squeue -u $USER

# Check job details
scontrol show job <JOBID>

# View job output (if job is running)
tail -f slurm-<JOBID>.out
```

### Check GPU Usage (While Job is Running)

```bash
# SSH to the compute node your job is on
squeue -u $USER  # Note the node name
ssh <node_name>
nvidia-smi
```

---

## Troubleshooting Common Issues

### Issue: "No space left on device"
**Unlikely on Torch**, but if it happens:
```bash
df -h /scratch  # Check filesystem capacity
du -sh /scratch/$USER/*  # Find what's using space
```

### Issue: "Too many open files"
This shouldn't happen, but if DataLoader complains:
```python
# Reduce num_workers in your config
data:
  num_workers: 2  # Instead of 4 or 8
```

### Issue: Job gets preempted
This is expected on Torch. Solution:
```bash
# Add requeue flag when submitting
--comment="preemption=yes;requeue=yes"

# Make sure your training script supports resuming from checkpoint
# (NAVSIM already does via Lightning)
```

### Issue: Slow DataLoader
```python
# Check DataLoader settings
data:
  batch_size: 32
  num_workers: 4  # Match CPU cores requested in sbatch
  pin_memory: true
  persistent_workers: true
```

---

## Next Steps - Immediate Actions

### Today (30 minutes)
1. ✅ SSH to Torch: `ssh greene.hpc.nyu.edu` then `ssh cs649`
2. ✅ Create directory structure (5 min)
3. ✅ Add environment variables to ~/.bashrc (5 min)
4. ✅ Set up Python environment (20 min)

### Tomorrow (3-4 hours)
5. ✅ Download maps (~10 min)
6. ✅ Start navmini download (~2-3 hours)
7. ✅ Test training on navmini (~1 hour)

### This Week (1-2 days)
8. ✅ Download navtrain (~8-12 hours, run overnight)
9. ✅ Launch baseline experiment (~24-48 hours)
10. ✅ Analyze results, iterate

---

## Questions for HPC (Optional Follow-up Email)

If you want to be extra safe, you could email HPC to confirm:

> Hi HPC Team,
> 
> Thank you for enabling my access to Torch! I've confirmed that:
> - /scratch has 94B inodes available (no quota set yet)
> - VAST filesystem is ideal for my NAVSIM workload
> 
> Quick questions:
> 1. Is it okay to download ~3M files (~459GB) for my thesis dataset now, before quotas are enforced?
> 2. Will existing data be grandfathered when quotas are applied, or should I expect retroactive limits?
> 3. Any timeline for when quotas will be enforced?
> 
> I want to ensure I'm following best practices and not causing issues for the cluster.
> 
> Best,
> Ali

But honestly, based on their email and the current state, **you're fine to proceed immediately**.

---

## Summary: Your Path Forward

### The Good News ✅
- Torch has **NO meaningful storage limits** for your use case
- VAST filesystem is **perfect for NAVSIM's random I/O patterns**
- You can download the **full dataset immediately**
- Better GPU options (L40S, H200) than Greene

### The "Caveat" ⚠️
- Torch is beta, jobs may be preempted (use checkpointing)
- No shared filesystem with Greene (download directly on Torch)
- Quotas coming "soon" (but you'll download everything first)

### The Answer to "What Are My Limitations?"
**Practically none.** You can run NAVSIM exactly as designed, with native filesystem access, and should have better performance than Greene would have offered.

### Recommended Approach
**Use Option A (Native Filesystem)**, ignore Singularity/containers for now. Follow the step-by-step plan above. You'll have working experiments within days.

---

## Contact & Help

- HPC Support: hpc@nyu.edu
- Torch docs: https://sites.google.com/nyu.edu/nyu-hpc/hpc-systems/torch/torch-new-user-guide
- NAVSIM docs: https://github.com/autonomousvision/navsim

**You're in great shape. Time to download data and start training! 🚀**
