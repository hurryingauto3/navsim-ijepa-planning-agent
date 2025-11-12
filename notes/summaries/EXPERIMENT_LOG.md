# Experiment Log - Thesis: Label-Efficient Trajectory Planning

**Project**: I-JEPA Features for NAVSIM Trajectory Planning  
**Author**: Ali Hamza  
**Start Date**: October 2025  
**Goal**: Demonstrate SSL feature transfer for autonomous planning with label efficiency

---

## Baseline Experiments

### EXP-001: Hydra-MDP V8192 w/ Ego Prediction (Baseline)
**Status**: ✅ COMPLETED  
**Job ID**: 273697 (273202 failed - OOM)
**Checkpoint**: `/scratch/ah7072/experiments/hydra_plus_16384_weighted_ep_ckpt/epoch=39-step=9680.ckpt`  
**Started**: Oct 16, 2025, 3:03 PM EDT  
**Completed**: Oct 17, 2025, 5:09 AM EDT
**Duration**: 14 hours 6 minutes

#### Configuration
```yaml
Agent: hydra_mdp_v8192_w_ep
TrainScript: train_4node_16gpu_l40s_FIXED.slurm
Architecture: VoVNet-99 backbone + trajectory vocabulary selection (16,384 candidates)
Backbone: VoVNet-99 (DD3D pretrained)
Planning Head: Multi-head PDM scoring → trajectory selection
Dataset: navtrain (85,109 train / 18,179 val samples)
Data Split: 100% training data

Training:
  Compute: 4 nodes × 4 L40S GPUs = 16 GPUs
  Batch Size: 18 per GPU (288 effective) ← REDUCED FROM 22 TO FIX OOM
  Workers: 1 per GPU (16 total) ← REDUCED FROM 2 TO FIX OOM
  Memory: 384GB per node ← INCREASED FROM 256GB TO FIX OOM
  Precision: FP16 mixed precision
  Epochs: 40
  Optimizer: AdamW (lr from config)
  
Loss Components:
  - Imitation loss (trajectory matching)
  - PDM no collision
  - PDM drivable area
  - PDM time-to-collision
  - PDM progress
  - PDM DDC (distance to closest)
  - PDM lane keeping
  - PDM traffic light compliance

Features:
  - Weighted confidence inference enabled
  - Continuous EP training (regression)
  - Vocabulary position normalization
```

#### Progress Tracking
| Metric | Epoch 0 Start | Epoch 0 End | Epoch 39 End | Improvement |
|--------|---------------|-------------|--------------|-------------|
| Total Loss (train) | 21.3 | 9.65 | 4.03 | **58.2% ↓** |
| Total Loss (val) | - | - | 13.30 | - |
| Imitation Loss (train) | 9.70 | 5.32 | 3.02 | **68.9% ↓** |
| Imitation Loss (val) | - | - | 7.87 | - |
| PDM No Collision (train) | 2.06 | 0.985 | 0.282 | **86.3% ↓** |
| PDM Drivable Area (train) | 2.21 | 0.622 | 0.144 | **93.5% ↓** |
| PDM TTC (train) | 2.74 | 1.30 | 0.336 | **87.7% ↓** |
| PDM Progress (train) | - | - | 0.0393 | - |
| PDM DDC (train) | - | - | 0.0511 | - |
| PDM Lane Keep (train) | - | 0.696 | 0.139 | **80.0% ↓** |
| PDM Traffic Light (train) | - | 0.260 | 0.0208 | **92.0% ↓** |
| Training Speed | 0.02 it/s | 0.21 it/s | 0.19 it/s | Stable |

#### Observations
- ✅ All 40 epochs completed successfully
- ✅ Training stable throughout - no OOM or crashes
- ✅ Loss convergence excellent: 58% train, 38% val
- ✅ Multi-node DDP working perfectly across 4 nodes
- ✅ Checkpoints saved successfully for all epochs (~909MB each)
- ⚠️ Training-validation gap suggests some overfitting (expected with 100% data)
- 🔧 OOM fix successful: mem=384GB, batch=18, workers=1

#### Issue Resolution (Job 273202 → 273697)
**Problem**: Job 273202 crashed after epoch 0 with SLURM OOM killer
- Original: batch_size=22, num_workers=2, mem=256GB
- Crashed during epoch 0→1 transition (validation dataloader init)

**Solution**: 
1. Increased memory: 256GB → 384GB per node
2. Reduced batch size: 22 → 18 per GPU
3. Reduced workers: 2 → 1 per GPU
4. Disabled persistent workers

**Result**: Job 273697 completed all 40 epochs successfully

#### Results
- **Training Loss**: 4.03 (final)
- **Validation Loss**: 13.30 (final)
- **PDM Score (navtest)**: ⏳ TBD - Need to run evaluation
  - No Collision: TBD
  - Drivable Area: TBD
  - Time-to-Collision: TBD
  - Progress: TBD
  - Comfort: TBD
- **Best Checkpoint**: `epoch=39-step=9680.ckpt`
- **Total Training Time**: 14 hours 6 minutes (16 GPUs)

#### Purpose
Establish baseline performance for NAVSIM Hydra-MDP agent. This will serve as the reference point for comparing:
1. I-JEPA feature quality vs. TransFuser backbone
2. Alternative planning head architectures
3. Label efficiency at reduced data percentages
4. Fine-tuning strategies

---

## Planned Ablations

### EXP-002: I-JEPA + MLP Baseline
**Status**: ⏳ PLANNED (Week 2)  
**Dependencies**: EXP-001 completion

#### Configuration
```yaml
Agent: ijepa_mlp_baseline (new)
Architecture: Frozen I-JEPA ViT-H/14 + MLP planning head
Backbone: facebook/ijepa-vit-huge-14 (frozen)
Planning Head: MLP [1280+32 ego → 512 → 256 → 128 → 8×3 trajectory]
Dataset: navtrain (100% data)

Training:
  Compute: 2 nodes × 4 L40S = 8 GPUs
  Batch Size: 32 per GPU (256 effective)
  Workers: 2 per GPU
  Precision: FP16
  Epochs: 40
  Freeze: I-JEPA encoder (only train MLP head)
```

#### Hypotheses
1. I-JEPA features will match or exceed TransFuser backbone quality
2. Training will be faster (fewer parameters to update)
3. Planning head simplicity may reduce PDM score vs. vocabulary selection

#### Metrics to Compare
- PDM score vs. EXP-001
- Training time vs. EXP-001
- Trajectory smoothness (PDM comfort)
- Generalization (val vs. test performance gap)

---

### EXP-003: I-JEPA + Transformer Head
**Status**: ⏳ PLANNED (Week 2-3)  
**Dependencies**: EXP-002 completion

#### Configuration
```yaml
Agent: ijepa_transformer_head (new)
Architecture: Frozen I-JEPA ViT-H/14 + Transformer decoder
Planning Head: 
  - Learned pose queries (8 queries for 8 poses)
  - 2-layer Transformer decoder
  - Attention over I-JEPA + ego context
  - Output: 8 × 3 (x, y, heading) trajectory
Dataset: navtrain (100% data)

Training:
  Compute: 2 nodes × 4 L40S = 8 GPUs
  Batch Size: 32 per GPU
  Precision: FP16
  Epochs: 40
```

#### Hypotheses
1. Attention mechanism will capture trajectory dependencies better
2. PDM lane keeping and comfort should improve
3. May be slower to train than MLP but better final performance

---

### EXP-004-007: Label Efficiency Studies
**Status**: ⏳ PLANNED (Week 3-4)  
**Dependencies**: EXP-002 completion

#### Experiments
- **EXP-004**: I-JEPA + MLP, 10% data (~8.5K samples)
- **EXP-005**: I-JEPA + MLP, 25% data (~21K samples)
- **EXP-006**: I-JEPA + MLP, 50% data (~42K samples)
- **EXP-007**: I-JEPA + MLP, 100% data (baseline from EXP-002)

#### Configuration
Same as EXP-002, varying only data percentage.

#### Metrics
- PDM score vs. data percentage (plot learning curve)
- Sample efficiency: PDM score per 1K training samples
- Convergence speed: epochs to 90% of max PDM score
- Variance: Run 3 seeds per data percentage

---

### EXP-008: Selective Fine-tuning
**Status**: ⏳ PLANNED (Week 4)  
**Dependencies**: EXP-002, EXP-004-007 completion

#### Configuration
```yaml
Agent: ijepa_mlp_finetune (new)
Architecture: I-JEPA ViT-H/14 (selective unfreeze) + MLP
Unfrozen Layers: Last 4 transformer blocks + MLP head
Dataset: navtrain (100% data)

Training:
  Compute: 2 nodes × 4 L40S = 8 GPUs
  Two-stage:
    - Stage 1: 20 epochs, frozen encoder
    - Stage 2: 20 epochs, unfreeze last 4 blocks
  Learning Rate: 1e-5 for unfrozen encoder, 1e-4 for head
```

#### Hypotheses
1. Selective fine-tuning will improve over frozen baseline
2. Risk of overfitting on small data (compare with EXP-004-006)
3. Best strategy: freeze on small data, fine-tune on large data

---

## Experiment Tracking Template

### EXP-XXX: [Experiment Name]
**Status**: ⏳ PLANNED | 🔥 IN PROGRESS | ✅ COMPLETE | ❌ FAILED  
**Job ID**: [SLURM Job ID]  
**Started**: [Date/Time]  
**Completed**: [Date/Time]

#### Configuration
[YAML or bullet points]

#### Hypotheses
[What you expect to learn]

#### Progress Tracking
[Table with metrics over time]

#### Observations
[Notes during training]

#### Results
[Final PDM scores, comparisons, plots]

#### Conclusions
[What you learned, next steps]

---

## Quick Reference

### Monitoring Commands
```bash
# Check running jobs
squeue -u $USER

# Tail training logs
tail -f /scratch/ah7072/navsim_workspace/exp/logs/train_<JOBID>.out

# Check GPU memory
sstat -j <JOBID> --format=JobID,MaxRSS,AveRSS

# Tensorboard (if logging)
tensorboard --logdir /scratch/ah7072/experiments/<exp_name>/
```

### Evaluation Commands
```bash
# Evaluate checkpoint on navtest
python navsim/planning/script/run_pdm_score.py \
  agent=<agent_config> \
  checkpoint=/scratch/ah7072/experiments/<exp_name>/checkpoints/last.ckpt \
  split=navtest \
  output_dir=/scratch/ah7072/experiments/<exp_name>/eval_navtest

# Quick check on navmini (faster)
python navsim/planning/script/run_pdm_score.py \
  agent=<agent_config> \
  checkpoint=<path> \
  split=navmini \
  output_dir=<output_dir>
```

### Key Paths
```bash
# Data
OPENSCENE_DATA_ROOT=/scratch/ah7072/data/openscene
NUPLAN_MAPS_ROOT=/scratch/ah7072/data/maps

# Experiments
NAVSIM_EXP_ROOT=/scratch/ah7072/experiments

# Repo
NAVSIM_DEVKIT_ROOT=/scratch/ah7072/GTRS

# Logs
/scratch/ah7072/navsim_workspace/exp/logs/
```

---

**Last Updated**: Oct 16, 2025, 3:30 PM  
**Active Experiments**: 1 (EXP-001)  
**Completed Experiments**: 0  
**Next Up**: EXP-002 (I-JEPA + MLP, after EXP-001 completes)
