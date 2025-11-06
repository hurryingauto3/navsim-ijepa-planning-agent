# System Architecture

Simple technical overview for developers.

---

## High-Level Design

```
Input → Feature Extraction → Planning → Output
```

**Components**:
1. **I-JEPA Encoder** (frozen): Extracts visual features from camera
2. **Ego History Builder**: Processes vehicle state (position, velocity)
3. **Planning Head** (MLP): Predicts trajectory from features
4. **Loss Function**: Trains head to match ground truth trajectories

---

## Agent Architecture

### IJEPAPlanningAgentV2

```python
class IJEPAPlanningAgent(pl.LightningModule):
    def __init__(self):
        self.encoder = AutoModel.from_pretrained("facebook/ijepa_vith14")
        self.encoder.eval()  # Frozen
        
        self.mlp_head = nn.Sequential(
            nn.Linear(1280 + 32, 512),  # Features + ego state
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 8 * 3)  # 8 waypoints × (x, y, heading)
        )
    
    def forward(self, features):
        # Extract I-JEPA features (frozen)
        visual = self.encoder(features["image"]).last_hidden_state[:, 0]
        
        # Concatenate with ego history
        combined = torch.cat([visual, features["ego_history"]], dim=-1)
        
        # Predict trajectory
        trajectory = self.mlp_head(combined).view(-1, 8, 3)
        return {"trajectory": trajectory}
```

**Key Point**: Only the MLP head trains. Encoder stays frozen.

---

## Training Pipeline

### PyTorch Lightning

```python
# Automatic multi-GPU training
trainer = Trainer(
    accelerator="gpu",
    devices=4,              # 4 GPUs
    strategy="ddp",         # Distributed training
    precision="16-mixed",   # fp16 for speed
    max_epochs=50
)

trainer.fit(agent, datamodule)
```

### Loss Function

```python
def compute_loss(pred, target):
    # Position loss (L1)
    pos_loss = F.l1_loss(pred[..., :2], target[..., :2])
    
    # Heading loss (circular distance)
    heading_loss = angle_loss(pred[..., 2], target[..., 2])
    
    # Optional: velocity/acceleration regularization
    vel_loss = velocity_consistency(pred)
    
    return pos_loss + 0.1 * heading_loss + 0.01 * vel_loss
```

---

## Data Flow

```
NAVSIM Dataset
    ↓
DataLoader (8 workers, prefetch)
    ↓
Feature Builders:
  - Image: Resize → Normalize → Tensor
  - Ego History: [x, y, vx, vy, heading] × 4 frames
    ↓
Agent Forward Pass
    ↓
Loss Computation
    ↓
Optimizer Step (only MLP params)
    ↓
Log to W&B
```

---

## HPC Setup (NYU Greene)

### Slurm Script

```bash
#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=10:00:00

# Setup
export NAVSIM_DEVKIT_ROOT=$HOME/GTRS
export OPENSCENE_DATA_ROOT=/scratch/$USER/openscene

# Run
srun python navsim/planning/script/run_training_dense.py \
    agent=ijepa_mlp_v2 \
    trainer.devices=4 \
    trainer.num_nodes=4
```

### Key Settings
- **4 nodes × 4 GPUs** = 16 GPUs total
- **Batch size**: 32 per GPU → 512 effective
- **Training time**: ~10 hours for 50 epochs

---

## Configuration (Hydra)

### Agent Config
```yaml
# configs/agent/ijepa_mlp_v2.yaml
_target_: navsim.agents.IJEPAPlanningAgentV2
ijepa_model_id: facebook/ijepa_vith14_1k
learning_rate: 1.0e-4
freeze_encoder: true
num_history_frames: 4
hidden_dims: [512, 256, 128]
```

### Override from Command Line
```bash
python train.py \
    agent.learning_rate=5e-5 \
    agent.hidden_dims=[1024,512,256] \
    trainer.max_epochs=100
```

---

## Evaluation

### PDM Score Components

| Metric | Weight | Description |
|--------|--------|-------------|
| No Collision | 0.25 | Avoid hitting objects |
| Drivable Area | 0.25 | Stay on road |
| Progress | 0.20 | Move toward goal |
| Time to Collision | 0.15 | Maintain safe distance |
| Comfort | 0.15 | Smooth driving |

**PDMS** = Weighted average of all components.

### Running Evaluation

```bash
python navsim/evaluate/pdm_score.py \
    checkpoint=/path/to/model.ckpt \
    split=navtest \
    output_dir=./results/
```

Output: JSON with per-scene and aggregate scores.

---

## Performance Benchmarks

### Training Speed
| Setup | Samples/sec | Time/Epoch |
|-------|-------------|------------|
| 1× L40S | 130 | 3.2 hours |
| 4× L40S | 520 | 42 min |
| 16× L40S (4 nodes) | 1850 | 12 min |
| 8× H100 | 3200 | 7 min |

### Memory Usage
- **Single GPU**: ~28 GB (with batch size 32)
- **Mixed precision (fp16)**: ~18 GB
- **Gradient checkpointing**: ~12 GB (slower training)

---

## Common Issues

### Out of Memory
```bash
# Solution 1: Reduce batch size
python train.py data.batch_size=16

# Solution 2: Enable gradient accumulation
python train.py trainer.accumulate_grad_batches=2

# Solution 3: Use fp16
python train.py trainer.precision="16-mixed"
```

### DDP Hangs
```bash
# Check NCCL
export NCCL_DEBUG=INFO

# Use different backend
python train.py trainer.strategy=ddp_spawn
```

### Slow Data Loading
```bash
# Increase workers
python train.py data.num_workers=8

# Use pre-cached data
python scripts/cache_dataset.py  # Run once
python train.py data.use_cache=true
```

---

## File Structure

```
navsim/navsim/agents/IJEPAPlanningAgentV2.py
    ├── __init__()           # Setup model, freeze encoder
    ├── forward()            # Inference
    ├── training_step()      # Loss computation + backward
    ├── validation_step()    # Eval metrics
    └── configure_optimizers()  # Adam, cosine schedule
```

---

## Extending the Agent

### Add New Feature
```python
class MyFeatureBuilder(AbstractFeatureBuilder):
    def compute_features(self, agent_input):
        # Custom preprocessing
        return {"my_feature": processed_data}

# Register in config
agent:
  feature_builders:
    - _target_: path.to.MyFeatureBuilder
```

### Change Planning Head
```python
class TransformerHead(nn.Module):
    def __init__(self, input_dim, num_waypoints=8):
        self.decoder = nn.TransformerDecoder(...)
    
    def forward(self, features):
        return self.decoder(features)

# Use in agent
self.planning_head = TransformerHead(input_dim=1280)
```

---

**For more details**, see the full thesis in `reports/final_report/`.
