# Hydra-MDP Torch Setup - Correct Paths

## Environment Variables
All scripts now use the correct paths for your workspace:

```bash
export NAVSIM_DEVKIT_ROOT="/scratch/ah7072/navsim_workspace/navsim"
export OPENSCENE_DATA_ROOT="/scratch/ah7072/navsim_workspace/dataset"
export NUPLAN_MAPS_ROOT="/scratch/ah7072/navsim_workspace/dataset/maps"
export NAVSIM_EXP_ROOT="/scratch/ah7072/navsim_workspace/exp"
```

## Directory Structure Expected

```
/scratch/ah7072/
├── GTRS/                              # Repo with scripts & configs
│   └── scripts/                       # Training/eval scripts (HERE)
├── navsim_workspace/
│   ├── navsim/                        # NAVSIM_DEVKIT_ROOT
│   ├── dataset/                       # OPENSCENE_DATA_ROOT
│   │   └── maps/                      # NUPLAN_MAPS_ROOT
│   └── exp/                           # NAVSIM_EXP_ROOT
│       ├── logs/                      # Slurm logs
│       ├── hydra_pdm_v8192_*/        # Training run outputs
│       ├── hydra_w_v8192_*/
│       └── hydra_w_ep_v8192_*/
```

## Quick Start

### 1. Create logs directory
```bash
mkdir -p /scratch/ah7072/navsim_workspace/exp/logs
```

### 2. Run smoke test (1 epoch, 10 batches)
```bash
cd /scratch/ah7072/GTRS/scripts
sbatch torch_smoke_test.slurm
```

### 3. Launch all three experiments
```bash
cd /scratch/ah7072/GTRS/scripts
./launch_all_torch.sh
```

### 4. Monitor jobs
```bash
squeue -u ah7072
tail -f /scratch/ah7072/navsim_workspace/exp/logs/hydra_*.out
```

### 5. Evaluate after training
```bash
cd /scratch/ah7072/GTRS/scripts
./launch_eval_torch.sh
```

## What Each Config Does

### `hydra_mdp_v8192_pdm.yaml` → PDM-Only Training
- **Script:** `torch_train_pdm.slurm`
- **Learning:** Overall PDM score only (no individual metric heads)
- **Expected:** Score 80.2 (NC=97.5, DAC=88.9, EP=74.8, TTC=92.5, C=100)
- **Purpose:** Baseline showing why multi-head learning is needed

### `hydra_mdp_v8192_w.yaml` → Weighted Confidence
- **Script:** `torch_train_w.slurm`
- **Learning:** All 5 metrics via separate heads (NC, DAC, TTC, C, EP)
- **Expected:** Score 85.7 (NC=98.1, DAC=96.1, EP=77.8, TTC=93.9, C=100)
- **Purpose:** Main ablation showing multi-target distillation works

### `hydra_mdp_v8192_w_ep.yaml` → Weighted + Continuous EP
- **Script:** `torch_train_w_ep.slurm`
- **Learning:** Same as W, but EP trained as continuous regression
- **Expected:** Score 86.5 (NC=98.3, DAC=96.0, EP=78.7, TTC=94.6, C=100)
- **Purpose:** Best performance, final submission quality

## Files Updated with Correct Paths

✅ All environment variables corrected in:
- `torch_smoke_test.slurm`
- `torch_train_pdm.slurm`
- `torch_train_w.slurm`
- `torch_train_w_ep.slurm`
- `torch_eval.slurm`
- `launch_all_torch.sh`
- `launch_eval_torch.sh`

✅ All log paths point to: `/scratch/ah7072/navsim_workspace/exp/logs/`

## Expected Training Time
- Each experiment: ~36-48 hours on H200 or L40S (20 epochs, 256 batch size)
- Smoke test: ~30 minutes
- Evaluation per model: ~4-6 hours on navtest split
