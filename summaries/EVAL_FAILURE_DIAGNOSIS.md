# Evaluation Failure Diagnosis - October 18, 2025

## Problem Summary
PDM Score evaluation job (296969) failed on ALL 388 scenarios with `AttributeError: 'list' object has no attribute 'unsqueeze'`

## Root Cause
**Bug in agent feature builder** - `/scratch/ah7072/GTRS/navsim/agents/gtrs_dense/hydra_features.py`

The `HydraFeatureBuilder.compute_features()` method is returning Python lists instead of PyTorch tensors for some features. When the abstract agent tries to add batch dimension with `.unsqueeze(0)`, it fails.

**Error trace:**
```
File "/scratch/ah7072/GTRS/navsim/agents/abstract_agent.py", line 76, in compute_trajectory
    features = {k: v.unsqueeze(0) for k, v in features.items()}
AttributeError: 'list' object has no attribute 'unsqueeze'
```

## What Worked
✅ Conda environment setup in /scratch/$USER/miniconda3
✅ SLURM job submission and resource allocation  
✅ Hydra configuration parsing (after fixing overrides)
✅ Model checkpoint loading
✅ Data loading (388 scenarios loaded)

## What Failed
❌ Feature extraction - returns lists instead of tensors
❌ Every single scenario evaluation (0/388 succeeded)

## Configuration Issues Resolved
1. ✅ Fixed conda activation to use full path: `conda activate "${CONDA_ROOT}/envs/navsim"`
2. ✅ Fixed Hydra overrides: `agent.checkpoint_path=\"${CHECKPOINT}\"`  
3. ✅ Changed worker from `ray_distributed` to `sequential`
4. ✅ Exported PYTHONPATH for navsim package import

## Next Steps

### Option 1: Check for Working Examples
```bash
# Look for evaluation scripts that actually work
find /scratch/ah7072/GTRS -name "*eval*" -o -name "*pdm*" | grep -E "\.(sh|slurm)$"
ls -la /scratch/ah7072/GTRS/scripts/evaluation/
```

### Option 2: Check Git History
```bash
cd /scratch/ah7072/GTRS
git log --oneline --all --grep="eval" | head -20
git log --oneline -- navsim/agents/gtrs_dense/hydra_features.py | head -10
```

### Option 3: Ask Advisor
Questions to ask:
- Is there a known working evaluation command for the baseline Hydra model?
- Is the feature builder bug documented?
- Should we use a different agent config for evaluation?

### Option 4: Debug Feature Builder
Inspect `/scratch/ah7072/GTRS/navsim/agents/gtrs_dense/hydra_features.py:64-180` (compute_features method) and ensure all returned values are torch.Tensor, not list.

## Files Changed
- `/scratch/ah7072/scripts/pdm_score_mini.slurm` - Fixed conda, Hydra overrides, worker
- `/scratch/ah7072/scripts/pdm_score.slurm` - Same fixes
- `/scratch/ah7072/GTRS/scripts/archives/setup_conda_in_scratch.sh` - Added Miniconda installer

## Job Logs
- Output: `/scratch/ah7072/navsim_workspace/exp/logs/eval_296969.out`
- Error: `/scratch/ah7072/navsim_workspace/exp/logs/eval_296969.err`

## Recommendation
**Do NOT modify the SLURM scripts further** - they are correctly configured. The issue is in the agent code itself. Focus on finding a working evaluation example or fixing the feature builder bug.
