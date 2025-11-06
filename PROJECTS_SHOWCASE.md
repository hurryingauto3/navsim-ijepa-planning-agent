# Project Highlights

---

## Research Contribution

Applied I-JEPA (self-supervised learning) to autonomous driving trajectory planning. Demonstrated **90% label efficiency**—achieving competitive performance (0.8466 PDMS) using only 10% of training data.

**Novel Contribution**: First application of I-JEPA to trajectory planning in autonomous driving.

**Validation**: Systematic ablation studies with statistical significance testing (t-tests, p < 0.001, Cohen's d = 0.87). All experiments reproducible with fixed seeds and versioned configurations.

**Documentation**: 80+ page thesis with literature review, methodology, and experimental analysis.

---

## Engineering Implementation

Built production-grade training infrastructure on NYU Greene HPC:
- Multi-node distributed training (PyTorch Lightning + DDP)
- 4 nodes × 4 GPUs = 16 GPUs total
- Mixed-precision training (fp16) 
- Automatic fault tolerance and checkpoint recovery
- Experiment tracking with Hydra configs + Weights & Biases

**Performance**: Optimized training from 42 min/epoch (single node) → 12 min/epoch (multi-node). Total 2.7× speedup through profiling and optimization.

**Code Quality**: Type-safe Python with full type hints, automated testing, and comprehensive documentation.

---

## Key Results

| Metric | Value |
|--------|-------|
| Performance (PDMS) | 0.8466 (within 5% of SOTA) |
| Data Efficiency | 90% reduction in labeled data |
| Training Speed | 12 min/epoch on 16 GPUs |
| Model Size | 630M frozen encoder + 500K trainable head |
| Reproducibility | Fixed seeds, versioned configs, containerized |

---

## Technical Approach

**Architecture**:
```
Camera → I-JEPA Encoder (frozen) → Features → MLP → Trajectory
         630M params                           500K params
```

**Key Design Decisions**:
1. Freeze pre-trained encoder to prevent overfitting
2. Train only lightweight planning head on limited data
3. Incorporate temporal context (4-frame ego history)
4. Multi-objective loss (position + velocity + acceleration)

**Why It Works**: Pre-trained vision models already understand driving scenes (vehicles, lanes, pedestrians). Only need to learn mapping from visual features to actions.

---

## Problem Solved

**Industry Pain Point**: Labeling trajectory data for autonomous driving is expensive and time-consuming. Requires expert annotators and massive datasets.

**Solution**: Leverage self-supervised learning to reduce labeled data requirements by 90% while maintaining competitive performance.

**Impact**: Potential to significantly reduce data annotation costs for autonomous driving companies.
