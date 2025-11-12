# Label-Efficient Trajectory Planning with I-JEPA

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Navsim](https://img.shields.io/badge/Navsim-brightgreen.svg)](https://github.com/hurryingauto3/navsim)
[![GTRS](https://img.shields.io/badge/GTRS-orange.svg)](https://github.com/hurryingauto3/GTRS)
---
Deep learning agent for autonomous vehicle trajectory planning using I-JEPA (self-supervised learning) to achieve **90% label efficiency**.
**Master's Thesis** | NYU Tandon | 2025-2026  
**Author**: Ali Hamza
---



## What This Does

Uses self-supervised learning to train autonomous vehicle planning with **90% less labeled data**. Achieves 0.8466 PDMS using only 10% of training data.

**Key Insight**: Pre-trained vision models (I-JEPA) already understand driving scenes. We just need to train a small planning head.

---

## Results

| Method | Data Used | Score (PDMS ↑) |
|--------|-----------|----------------|
| Baseline (ego-only) | 100% | 0.7821 |
| **Our Method (I-JEPA)** | **10%** | **0.8466** |
| State-of-the-art | 100% | 0.8912 |

**Bottom Line**: Match most baselines with 10× less data, within 5% of SOTA.

---

## How It Works

```
Camera Image → I-JEPA Encoder (frozen) → Features → MLP Head → Trajectory
                 630M params                          500K params (trainable)
```

1. **I-JEPA Encoder**: Pre-trained ViT-H/14 on ImageNet (frozen, no training)
2. **Planning Head**: Small MLP that learns to map features to trajectories
3. **Training**: Only train the 500K param head, not the 630M encoder

**Why This Works**: I-JEPA already learned to recognize cars, lanes, pedestrians. We just teach it what actions to take.

---

## Repository Structure

```
navsim-ijepa/
├── GTRS/                             # Fork: Baseline implementations
├── navsim/                           # Fork: Core framework
│   └── agents/IJEPAPlanningAgentV2.py  # ⭐ Main agent (800 lines)
└── navsim-ijepa-planning-agent/      # Research workspace
    ├── code/ijepa/                   # Agent implementations
    ├── scripts/                      # HPC training scripts (Slurm)
    ├── reports/                      # Thesis LaTeX
    ├── summaries/                    # Experiment logs
    └── web/                          # Web showcase + demo assets
```

---

## Quick Start

### Installation (5 minutes)

```bash
# Clone repos
mkdir navsim-ijepa
cd navsim-ijepa
git clone https://github.com/hurryingauto3/navsim-planning-agent.git
git clone https://github.com/hurryingauto3/navsim.git
git clone https://github.com/hurryingauto3/GTRS.git

# Install dependencies
conda env create -f navsim/environment.yml
conda activate navsim
pip install -e .

# Set paths
export NAVSIM_DEVKIT_ROOT="$(pwd)"
export OPENSCENE_DATA_ROOT="/scratch/$USER/openscene"
export NAVSIM_EXP_ROOT="/scratch/$USER/experiments"
```

### Training (Single GPU, navmini dataset)

```bash
python navsim/planning/script/run_training_dense.py \
    agent=ijepa_mlp_v2 \
    experiment_name=test_run \
    split=navmini \
    trainer.max_epochs=10 \
    trainer.devices=1
```

### Training (Multi-GPU on HPC)

```bash
# Edit scripts/train_4node.slurm with your config
sbatch scripts/train_4node.slurm
```

### Evaluation

```bash
python navsim/evaluate/pdm_score.py \
    agent=ijepa_mlp_v2 \
    checkpoint=/path/to/checkpoint.ckpt \
    split=navtest
```

---

## Technical Details

### Architecture
- **Encoder**: I-JEPA ViT-H/14 (630M params, frozen)
- **Planning Head**: 3-layer MLP (500K params)
- **Input**: Front camera (512×2048) + ego history (4 frames)
- **Output**: 8 future waypoints (x, y, heading)

### Training Setup
- **Framework**: PyTorch Lightning + Hydra
- **Hardware**: 4 nodes × 4 GPUs (L40S/A100/H100)
- **Time**: 6-10 hours for 50 epochs
- **Data**: NAVSIM navtrain dataset (10% split)

### Engineering Highlights
- Multi-node distributed training (DDP)
- Mixed-precision (fp16) for 2× speedup
- Automatic checkpointing + recovery
- Weights & Biases experiment tracking

---

## Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System design (if you want technical details)
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Development guide
- **`reports/final_report/`** - Full thesis LaTeX
- **`web/`** - Browser-first demo (Next.js + FastAPI)

---

## Web Showcase (Product Demo)

A polished demo lives in `navsim-ijepa-planning-agent/web/`. It ships a full stack for interviewing and conference demos:

- **Frontend** (`frontend/`): Next.js (App Router) + deck.gl rendering cached or live trajectories. Includes in-browser ONNX runtime (WebGPU/WASM) for lightweight models.
- **Backend** (`backend/`): FastAPI service providing discovery endpoints and a websocket streaming API. Stubbed live mode can be replaced with real NavSim agents via `navsim_bridge.py`.
- **Data** (`data/`): Scene manifest and cached run JSON used for instant replays.
- **Infrastructure** (`infra/`): Docker Compose + Dockerfiles for one-command local runs.
- **Scripts** (`scripts/`): Utilities to export the IJEPA MLP ONNX model and sync cached runs to Cloudflare R2.

### Quick Start Demo

```bash
cd navsim-ijepa-planning-agent/web/navsim-showcase

# Backend (localhost:8000)
uvicorn backend.main:app --reload

# Frontend (localhost:3000, new terminal)
cd frontend
pnpm install
pnpm dev
```

Open http://localhost:3000, select `IJEPA-MLP` + `scene_001`, click **Run** to watch the replay and inspect metrics. Switch to **Live** mode for the streaming stub, or press **Test ONNX** to validate browser inference. `infra/docker-compose.yml` mirrors the same flow inside containers.
