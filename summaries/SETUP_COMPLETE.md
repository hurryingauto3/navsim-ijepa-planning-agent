# ✅ Torch Setup Complete!

**Date:** October 14, 2025  
**Environment:** `navsim` on Torch HPC (cs649)

## Summary

Your NAVSIM environment is now ready on Torch HPC! All scripts have been updated with the correct module version (`anaconda3/2025.06`).

## What Was Fixed

1. **Module Version Issue**: Scripts needed full version `anaconda3/2025.06` instead of just `anaconda3`
2. **All Slurm Scripts Updated**: Fixed module loading in all training/evaluation scripts
3. **Environment Created**: Successfully installed navsim conda environment with PyTorch 2.8.0

## Environment Verification

```bash
✓ Conda environment: navsim
✓ Python: 3.9.x
✓ PyTorch: 2.8.0+cu128
✓ CUDA: Will be available on GPU compute nodes
```

## Quick Start

### Activate Environment
```bash
module load anaconda3/2025.06
conda activate navsim
```

### Check Status
```bash
/scratch/ah7072/GTRS/scripts/check_torch_setup.sh
```

### Run Smoke Test
```bash
cd /scratch/ah7072/GTRS/scripts
sbatch torch_smoke_test.slurm
```

### Launch Training
```bash
./launch_all_torch.sh
```

## Next Steps

1. **Verify Data**: Check if OpenScene data exists
   ```bash
   ls /scratch/ah7072/navsim_workspace/dataset/openscene/
   ```

2. **Run Smoke Test** (30 min on GPU)
   ```bash
   sbatch torch_smoke_test.slurm
   squeue -u ah7072
   ```

3. **Launch Training** (if smoke test passes)
   ```bash
   ./launch_all_torch.sh
   ```

## Environment Variables (Optional)

Add to `~/.bashrc` for convenience:

```bash
# NAVSIM on Torch
if [[ $(hostname) == *"cs649"* ]]; then
    module load anaconda3/2025.06 2>/dev/null || true
    export NAVSIM_DEVKIT_ROOT="/scratch/ah7072/navsim_workspace/navsim"
    export OPENSCENE_DATA_ROOT="/scratch/ah7072/navsim_workspace/dataset"
    export NUPLAN_MAPS_ROOT="/scratch/ah7072/navsim_workspace/dataset/maps"
    export NAVSIM_EXP_ROOT="/scratch/ah7072/navsim_workspace/exp"
    alias navsim='conda activate navsim'
fi
```

Then reload: `source ~/.bashrc`

## Files Updated

- ✅ `torch_cache_metrics.slurm` - Fixed module version
- ✅ `torch_smoke_test.slurm` - Fixed module version
- ✅ `torch_train_pdm.slurm` - Fixed module version
- ✅ `torch_train_w.slurm` - Fixed module version  
- ✅ `torch_train_w_ep.slurm` - Fixed module version + added proper conda activation
- ✅ `torch_eval.slurm` - Fixed module version + added proper conda activation
- ✅ `setup_torch_navsim.sh` - Fixed module version
- ✅ `TORCH_SETUP_GUIDE.md` - Updated with correct module version

## Troubleshooting

### If conda activation fails
```bash
# Manually check module
module load anaconda3/2025.06
which conda  # Should show path

# Activate environment
conda activate navsim
python --version  # Should show Python 3.9.x
```

### If smoke test fails
```bash
# Check logs
cat /scratch/ah7072/navsim_workspace/exp/logs/smoke_*.err

# Try manual test
cd /scratch/ah7072/navsim_workspace/navsim
module load anaconda3/2025.06
conda activate navsim
python -c "import torch; print(torch.cuda.is_available())"
```

### If training jobs fail
```bash
# Check SLURM output
cat /scratch/ah7072/navsim_workspace/exp/logs/hydra_*.err

# Verify environment variables
echo $OPENSCENE_DATA_ROOT
echo $NAVSIM_DEVKIT_ROOT
```

## Documentation

- **Complete Guide**: `/scratch/ah7072/GTRS/scripts/TORCH_SETUP_GUIDE.md`
- **Torch vs Greene**: `/scratch/ah7072/TORCH_EXPLAINED.md`
- **Caching Guide**: `/scratch/ah7072/GTRS/scripts/CACHING_GUIDE.md`
- **Complete Workflow**: `/scratch/ah7072/GTRS/scripts/COMPLETE_WORKFLOW.md`

## Ready to Go! 🚀

Your setup is complete. Run the smoke test to verify everything works on a GPU node:

```bash
cd /scratch/ah7072/GTRS/scripts
sbatch torch_smoke_test.slurm
```

Monitor with: `squeue -u ah7072` and `tail -f /scratch/ah7072/navsim_workspace/exp/logs/smoke_*.out`
