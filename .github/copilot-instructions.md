# VS Code Copilot Instructions — Ali Hamza's Thesis Project

- Don't make MD files unless explicitly asked.

## Quick Facts
- Project: Master's thesis on label-efficient trajectory planning in NAVSIM using I-JEPA features.
- Repository root: `GTRS/` (NAVSIM fork). Work happens inside `navsim/` unless otherwise stated.
- Primary compute: NYU Greene HPC. Heavy training/eval must run through Slurm with data and outputs under `/scratch/$USER/`.
- Key scratch paths: `OPENSCENE_DATA_ROOT=/scratch/$USER/openscene`, `NUPLAN_MAPS_ROOT=/scratch/$USER/maps`, `NAVSIM_EXP_ROOT=/scratch/$USER/experiments`.
- Guiding principle: "Ship experiments > polish code." Only request complexity that moves the next experiment forward.

## Project Objectives
1. Baseline: Frozen I-JEPA (ViT-H/14) encoder + lightweight planning head.
2. Extensions: Modular heads (Transformer, GRU), selective encoder fine-tuning.
3. Analysis: Label efficiency runs at 10/25/50/100% data, plus ablations on head/backbone choices.
4. Contribution: Demonstrate SSL feature transfer for autonomous trajectory planning in NAVSIM.

Keep completions pointed at these deliverables unless the user explicitly changes scope.

## Current Push (October 2025)
- Reproduce I-JEPA + MLP baseline and log metrics (Week 1).
- Stand up Transformer planning head variant (Weeks 2–3).
- Run first ablations + draft preliminary results section (Week 4).

Always bias suggestions toward the items above unless the user explicitly changes scope.

### Milestone Rhythm
- Weekly: finish at least one experiment, sync metrics + plots, push code, prep advisor update.
- Monthly checkpoint: verify goals, refresh results section, tidy code/comments, plan next sprint.
- Thesis defense window: May–June 2026, so leave buffer for polished experiments and writing.

## Copilot Priorities
- Default to minimal, working code that fits NAVSIM patterns.
- Prefer edits inside `navsim/agents/thesis_agents/` and matching Hydra configs over touching core framework.
- Promote quick validation: dry runs, `navmini` split, 1-epoch smoke tests on Greene before long jobs.
- Surface experiment-tracking reminders (Weights & Biases) when relevant.

## Repo Map
```
navsim/
├── agents/                    # Agent implementations
│   ├── thesis_agents/         # Thesis-specific agents live here
│   ├── ego_status_mlp_agent.py
│   └── transfuser/            # Reference for vision-based agents
├── planning/script/
│   ├── config/                # Hydra configs (agent/training/eval)
│   └── run_training_dense.py  # Main training entrypoint
├── planning/training/         # Lightning module, dataset, builders
└── evaluate/pdm_score.py      # Top-level evaluation script
```

## Implementation Playbooks

### New Agent Workflow
1. Create file under `navsim/agents/thesis_agents/` and inherit from `AbstractAgent`.
2. Implement `get_sensor_config`, `forward`, `compute_loss`, `configure_optimizers`.
3. Reuse existing feature/target builders or subclass from `abstract_feature_target_builder.py`.
4. Keep the I-JEPA encoder frozen unless the experiment says otherwise.
5. Add matching Hydra config in `navsim/planning/script/config/common/agent/`.

```python
class IJEPAMLPAgent(AbstractAgent):
    def __init__(self, model_path="facebook/ijepa-vit-huge-14", hidden_dims=[512, 256, 128],
                 ego_history_frames=4, num_output_poses=8, learning_rate=1e-4, freeze_encoder=True):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_path)
        if freeze_encoder:
            self.encoder.eval()
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.ego_dim = 8 * ego_history_frames
        self.head = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size + self.ego_dim, hidden_dims[0]),
            nn.ReLU(), nn.LayerNorm(hidden_dims[0]),
            nn.Linear(hidden_dims[0], hidden_dims[1]), nn.ReLU(),
            nn.Linear(hidden_dims[1], num_output_poses * 3)
        )

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        vis = self.encoder(features["image"]).last_hidden_state[:, 0, :]
        ego = features["ego_status"]
        traj = self.head(torch.cat([vis, ego], dim=-1)).view(-1, 8, 3)
        return {"trajectory": traj}
```

```python
class IJEPAImageFeatureBuilder(AbstractFeatureBuilder):
    def __init__(self, processor: AutoProcessor):
        super().__init__()
        self.processor = processor

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        front_cam = agent_input.sensors.cameras[-1]["CAM_F0"]
        processed = self.processor(images=front_cam.image, return_tensors="pt")
        return {"image": processed["pixel_values"].squeeze(0)}


class TrajectoryTargetBuilder(AbstractTargetBuilder):
    def __init__(self, num_poses: int = 8):
        super().__init__()
        self.num_poses = num_poses

    def compute_targets(self, scene) -> Dict[str, torch.Tensor]:
        future = scene.get_future_trajectory(num_poses=self.num_poses)
        current_pose = scene.get_ego_state().pose
        rel = self._to_relative(future, current_pose)
        return {"trajectory": torch.tensor(rel, dtype=torch.float32)}

    def _to_relative(self, trajectory, reference_pose):
        raise NotImplementedError("Implement ego frame conversion before training")
```

### Transformer Head Variant
- Use a lightweight decoder like `TransformerPlanningHead` (batch-first, learned pose queries).
- Integrate by replacing the MLP head while keeping encoder + feature builders unchanged.
- Surface config knobs: `hidden_dim`, `num_layers`, `num_heads`, `dropout`.
- Suggest training command with clear experiment name `transformer_head_<date>`.

```python
class TransformerPlanningHead(nn.Module):
    def __init__(self, context_dim: int = 1280 + 32, hidden_dim: int = 512,
                 num_layers: int = 2, num_heads: int = 8, num_poses: int = 8):
        super().__init__()
        self.input_proj = nn.Linear(context_dim, hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=2048,
            dropout=0.1, activation="gelu", batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.pose_queries = nn.Parameter(torch.randn(num_poses, hidden_dim))
        self.output_proj = nn.Linear(hidden_dim, 3)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        bsz = context.size(0)
        memory = self.input_proj(context).unsqueeze(1)
        queries = self.pose_queries.unsqueeze(0).expand(bsz, -1, -1)
        decoded = self.decoder(queries, memory)
        return self.output_proj(decoded)
```

### Hydra Config Pattern
```yaml
# navsim/planning/script/config/common/agent/ijepa_mlp_baseline.yaml
_target_: navsim.agents.thesis_agents.ijepa_mlp_agent.IJEPAMLPAgent
model_path: facebook/ijepa-vit-huge-14
hidden_dims: [512, 256, 128]
ego_history_frames: 4
num_output_poses: 8
freeze_encoder: true
learning_rate: 1.0e-4
feature_builders:
  - _target_: navsim.agents.thesis_agents.ijepa_mlp_agent.IJEPAImageFeatureBuilder
  - _target_: navsim.agents.thesis_agents.ijepa_mlp_agent.EgoStatusFeatureBuilder
    history_frames: 4
target_builder:
  _target_: navsim.agents.thesis_agents.ijepa_mlp_agent.TrajectoryTargetBuilder
  num_poses: 8
```

## Pattern Library

### Multi-Head Scoring Agent
Use this when proposing multi-objective scoring similar to NAVSIM production agents.

```python
class HydraStyleAgent(AbstractAgent):
    def __init__(self, encoder_dim: int, vocab_size: int = 8192):
        super().__init__()
        self.encoder = self._build_encoder()
        self.heads = nn.ModuleDict({
            "no_collision": self._make_head(encoder_dim, vocab_size),
            "drivable_area": self._make_head(encoder_dim, vocab_size),
            "ego_progress": self._make_head(encoder_dim, vocab_size),
            "ttc": self._make_head(encoder_dim, vocab_size),
            "comfort": self._make_head(encoder_dim, vocab_size),
        })
        self.trajectory_vocab = self._load_vocab(vocab_size)

    def _make_head(self, encoder_dim: int, vocab_size: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(encoder_dim, 512),
            nn.ReLU(),
            nn.Linear(512, vocab_size)
        )

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        encoded = self.encoder(features)
        scores = {name: head(encoded) for name, head in self.heads.items()}
        blended = sum(scores.values()) / len(scores)
        best_idx = blended.argmax(dim=-1)
        return {"trajectory": self.trajectory_vocab[best_idx], "scores": scores}
```

## Training & Evaluation (Run on Greene HPC)
- Always ensure `NAVSIM_DEVKIT_ROOT` points to the repo clone on the compute node.
- Export data/experiment roots to scratch before launching jobs:
  ```bash
  export NAVSIM_DEVKIT_ROOT="$HOME/projects/GTRS"
  export OPENSCENE_DATA_ROOT="/scratch/$USER/openscene"
  export NUPLAN_MAPS_ROOT="/scratch/$USER/maps"
  export NAVSIM_EXP_ROOT="/scratch/$USER/experiments"
  ```
- Example Slurm launch (fill in partition/time as needed):
  ```bash
  sbatch gtrs_multinode_train.slurm \
    --wrap "python navsim/planning/script/run_training_dense.py \
      agent=ijepa_mlp_baseline \
      experiment_name=baseline_ijepa_mlp_$(date +%Y%m%d) \
      trainer.max_epochs=50 trainer.devices=1 trainer.accelerator=gpu \
      data.batch_size=32 data.num_workers=4"
  ```
- Quick smoke test (interactive GPU node):
  ```bash
  python navsim/planning/script/run_training_dense.py \
    agent=ijepa_mlp_baseline trainer.max_epochs=1 \
    trainer.limit_train_batches=10 trainer.limit_val_batches=5
  ```
- Evaluation template:
  ```bash
  python navsim/planning/script/run_pdm_score.py \
    agent=ijepa_mlp_baseline \
    checkpoint=/scratch/$USER/experiments/<run>/checkpoints/last.ckpt \
    split=navmini \
    output_dir=/scratch/$USER/experiments/<run>/eval_navmini
  ```
  Swap to `navtest` for full PDMS once the navmini sanity check passes.

## Experiment Tracking
- Every training run should log to Weights & Biases `thesis_navsim` project.
```python
import wandb
wandb.init(project="thesis_navsim", name=f"{agent}_{variant}_{date}", config=cfg)
wandb.log({"train_loss": train_loss, "val_loss": val_loss, "learning_rate": lr, "epoch": epoch})
```
- Log final PDMS and sub-metrics after evaluation.
- Store experiment notes + plots under `/scratch/$USER/experiments/<run>/notes/`.

## Guardrails for Suggestions
- ✅ Keep edits localized to thesis agents/configs unless the user confirms broader changes.
- ✅ Encourage incremental testing: navmini split, single epoch, gradient checks.
- ✅ Remind about freezing/unfreezing the encoder based on experiment goal.
- ❌ Do not propose altering `navsim/agents` core abstractions or data loaders unless explicitly asked.
- ❌ Avoid introducing brand-new training scripts; extend via configs instead.
- ❌ No suggestions that skip coordinate conversions between absolute and ego-relative frames.

## Troubleshooting Prompts
- If training fails: ask for stack trace, confirm env vars, consider reducing batch size or num_workers.
- If metrics regress: compare feature magnitudes to baseline, visualize trajectories, verify loss scaling.
- GPU memory issues: suggest gradient checkpointing or smaller batch, never silently drop features.

## Weekly / Monthly Reminders
- Weekly (by Friday): finish ≥1 experiment, sync results to log, push code, prep advisor update.
- Monthly checkpoint: confirm goals met/adjusted, refresh results section, clean code/comments, align next sprint.

## Golden Rule
Any suggestion should answer: “Does this move Ali closer to the next experiment delivering concrete results on Greene HPC?” If not, propose deferring.
