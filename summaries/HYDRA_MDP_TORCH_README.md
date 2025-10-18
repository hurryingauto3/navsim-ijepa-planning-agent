# Hydra-MDP Ablation Study on Torch

This directory contains scripts to reproduce the missing Hydra-MDP results from Table 1 of the paper.

## Paper Reference
**Hydra-MDP: End-to-end Multimodal Planning with Multi-target Hydra-Distillation**
- arXiv: 2406.06978v3
- Challenge: 1st place in NAVSIM

## What Each Config Does

### 1. `hydra_mdp_v8192_pdm.yaml` - PDM-Only Baseline
**Paper Results:** Score 80.2 (NC=97.5, DAC=88.9, EP=74.8, TTC=92.5, C=100)

- **Learning paradigm:** Trains on overall PDM score only
- **Key settings:**
  - `pdm_only_training: true`
  - `trajectory_imi_weight: 0.0` (no trajectory imitation)
- **Purpose:** Demonstrates that learning from aggregated PDM score underperforms multi-target learning
- **Training script:** `torch_train_pdm.slurm`

### 2. `hydra_mdp_v8192_w.yaml` - Weighted Confidence
**Paper Results:** Score 85.7 (NC=98.1, DAC=96.1, EP=77.8, TTC=93.9, C=100)

- **Learning paradigm:** Multi-target Hydra distillation with 5 separate prediction heads
- **Key settings:**
  - `weighted_confidence_inference: true`
  - `trajectory_imi_weight: 1.0`
  - 5 heads: No Collision (NC), Drivable Area Compliance (DAC), Time to Collision (TTC), Comfort (C), Ego Progress (EP)
- **Purpose:** Shows major improvement from learning individual metrics with weighted confidence
- **Training script:** `torch_train_w.slurm`

### 3. `hydra_mdp_v8192_w_ep.yaml` - Weighted + Continuous EP
**Paper Results:** Score 86.5 (NC=98.3, DAC=96.0, EP=78.7, TTC=94.6, C=100)

- **Learning paradigm:** Same as W variant, but EP uses continuous regression
- **Key settings:**
  - `weighted_confidence_inference: true`
  - `continuous_ep_training: true` (EP as regression vs. binary classification)
  - `trajectory_imi_weight: 1.0`
  - `progress_weight: 2.0`
- **Purpose:** Best-performing variant - shows continuous EP training improves ego progress metric
- **Training script:** `torch_train_w_ep.slurm`

## Key Differences Summary

| Variant | PDM-only | Weighted (W) | W + Continuous EP |
|---------|----------|--------------|-------------------|
| Multi-head prediction | ❌ | ✅ | ✅ |
| Weighted confidence | ❌ | ✅ | ✅ |
| Trajectory imitation | ❌ | ✅ | ✅ |
| Continuous EP training | ❌ | ❌ | ✅ |
| **Expected Score** | **80.2** | **85.7** | **86.5** |

## How to Run on Torch

### Prerequisites
1. **Access Torch from Greene:**
   ```bash
   ssh greene.hpc.nyu.edu
   ssh cs649
   ```

2. **Verify data paths exist:**
   ```bash
   ls /scratch/$USER/openscene/trainval_navsim_logs
   ls /scratch/$USER/maps
   ls $HOME/GTRS/traj_final/16384.npy
   ```

3. **Ensure container image exists:**
   ```bash
   ls /share/apps/images/pytorch-24.09.sif
   ```
   (Update `CONTAINER_IMAGE` path in scripts if different)

### Training

**Option 1: Launch all three experiments at once**
```bash
cd /scratch/$USER/GTRS/scripts
chmod +x launch_all_torch.sh
./launch_all_torch.sh
```

**Option 2: Launch individually**
```bash
cd /scratch/$USER/GTRS/scripts
sbatch torch_train_pdm.slurm      # PDM-only baseline
sbatch torch_train_w.slurm        # Weighted confidence
sbatch torch_train_w_ep.slurm     # Weighted + continuous EP
```

**Monitor jobs:**
```bash
squeue -u $USER
```

**Check training progress:**
```bash
tail -f /scratch/$USER/experiments/logs/hydra_*_<job_id>.out
```

### Evaluation

After training completes (20 epochs, ~24-48 hours per experiment):

**Option 1: Auto-evaluate all models**
```bash
cd /scratch/$USER/GTRS/scripts
chmod +x launch_eval_torch.sh
./launch_eval_torch.sh
```

**Option 2: Evaluate specific checkpoint**
```bash
sbatch torch_eval.slurm \
  /scratch/$USER/experiments/hydra_pdm_v8192_<timestamp>/checkpoints/last.ckpt \
  pdm_eval \
  hydra_mdp_v8192_pdm
```

### Results Location

**Training outputs:**
```
/scratch/$USER/experiments/
├── hydra_pdm_v8192_<timestamp>/
│   ├── checkpoints/
│   │   └── last.ckpt
│   ├── logs/
│   └── notes/
│       └── experiment_info.txt
├── hydra_w_v8192_<timestamp>/
└── hydra_w_ep_v8192_<timestamp>/
```

**Evaluation outputs:**
```
/scratch/$USER/experiments/
├── eval_pdm_<timestamp>/
│   └── pdm_scores.json
├── eval_w_<timestamp>/
└── eval_w_ep_<timestamp>/
```

## Expected Results (from Paper Table 1)

| Method | NC↑ | DAC↑ | EP↑ | TTC↑ | C↑ | Score↑ |
|--------|-----|------|-----|------|----|----|
| Hydra-MDP-V₈₁₉₂-PDM | 97.5 | 88.9 | 74.8 | 92.5 | 100 | **80.2** |
| Hydra-MDP-V₈₁₉₂-W | 98.1 | 96.1 | 77.8 | 93.9 | 100 | **85.7** |
| Hydra-MDP-V₈₁₉₂-W-EP | 98.3 | 96.0 | 78.7 | 94.6 | 100 | **86.5** |

## Resource Requirements (per experiment)

- **GPUs:** 1x H200 or L40S
- **Memory:** 80GB RAM
- **Training time:** ~24-48 hours (20 epochs)
- **Disk space:** ~10GB per experiment (checkpoints + logs)
- **Total disk:** ~30GB for all three variants

## Troubleshooting

### Job preempted?
Scripts already include `--comment="preemption=yes;requeue=yes"`, so jobs will auto-restart. Ensure your training code supports checkpointing.

### Container not found?
Update `CONTAINER_IMAGE` path in `.slurm` files:
```bash
CONTAINER_IMAGE="/path/to/your/pytorch.sif"
```

### Data not found?
Verify environment variables:
```bash
export OPENSCENE_DATA_ROOT="/scratch/$USER/openscene"
export NUPLAN_MAPS_ROOT="/scratch/$USER/maps"
export NAVSIM_DEVKIT_ROOT="$HOME/GTRS"
```

### Out of memory?
Reduce batch size in slurm scripts:
```bash
data.batch_size=16  # Instead of 32
```

## Understanding the Learning Paradigms

### Paradigm A: Single-modal Planning (Traditional IL)
```
Loss = L_imitation(T*, T_expert)
```
Just mimics human drivers. No closed-loop awareness.

### Paradigm B: Multimodal Planning + Post-processing (VAD, Transfuser)
```
Loss = Σ L_imitation(T_i, T_expert)
T* = argmin f(T_i, Perception)  # Non-differentiable
```
Predicts multiple trajectories, post-processes with perception. Not end-to-end.

### Paradigm C: Multi-target Hydra Distillation (Hydra-MDP)
```
Loss = Σ L_imitation(T_i, T_expert) + L_kd(f(T_i, GT), f_student(T_i, obs))
T* = argmin f_student(T_i, obs)  # Fully differentiable
```
**Key innovation:** Student learns to predict rule-based costs from observations (no GT perception at inference). Fully end-to-end.

## Citation
If you use these configs/results in your thesis:
```bibtex
@article{li2024hydramdp,
  title={Hydra-MDP: End-to-end Multimodal Planning with Multi-target Hydra-Distillation},
  author={Li, Zhenxin and Li, Kailin and Wang, Shihao and Lan, Shiyi and Yu, Zhiding and Ji, Yishen and Li, Zhiqi and Zhu, Ziyue and Kautz, Jan and Wu, Zuxuan and Jiang, Yu-Gang and Alvarez, Jose M.},
  journal={arXiv preprint arXiv:2406.06978},
  year={2024}
}
```

## Next Steps for Your Thesis

After reproducing these baselines, you can:
1. **Replace the vision backbone** with I-JEPA encoder
2. **Freeze I-JEPA** and only train the planning heads
3. **Run label-efficiency ablations** (10%/25%/50%/100% data)
4. **Compare** I-JEPA features vs. ResNet/ViT features for planning

This gives you strong baselines to demonstrate I-JEPA's value for trajectory planning.
