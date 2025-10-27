# TransFuser ⨉ I-JEPA: Vision–LiDAR Fusion Backbone

This README documents the architecture, math, and design choices behind integrating a **frozen I-JEPA** vision tower with a **LiDAR BEV encoder** inside the TransFuser agent. It includes flow diagrams, equations, interfaces, and recommended experiments. It also records open issues and next steps (including preserving your **wide-FOV** images end-to-end).

---

## High-level idea

We replace the camera ResNet branch with a **frozen I-JEPA ViT** to extract strong global/patch features from the **entire camera frame**. Instead of forcing an artificial token grid alignment with LiDAR, we adopt a **Perceiver-style latent cross-attention**: a learnable latent array queries the I-JEPA patch tokens to distill a compact **global camera vector**. We then **broadcast** this vector over the LiDAR BEV grid and perform pointwise fusion to produce a BEV feature map usable by the rest of TransFuser (FPN/top-down heads, detection/semantic heads, etc.).

---

## System Diagram (overview)

```mermaid
flowchart LR
    A[RGB image (wide FOV)] --> B[I-JEPA ViT (frozen)]
    B --> C[Patch tokens X ∈ R^{N×d_v}]
    C --> D[Latent array L ∈ R^{L×d_f}]
    D -->|cross-attention| E[Refined latents L' ∈ R^{L×d_f}]
    E -->|mean| F[Global camera vector g ∈ R^{d_f}]
    G[LiDAR BEV encoder (timm)] --> H[LiDAR map ℓ ∈ R^{C_ℓ×H×W}]
    F --> I[Broadcast g → G ∈ R^{d_f×H×W}]
    H --> J
    I --> J[Concat [G ; ℓ] → Linear(1×1) → F_bev ∈ R^{C×H×W}]
    J --> K[Top-down refine (FPN-like)]
    K --> O1[High-res BEV for heads]
    J --> O2[Pool(8×8) → z ∈ R^{C·8·8} (global)]
    B -. frozen .-.- B
```

---

## Notation

* Camera:

  * Input image (I \in \mathbb{R}^{B\times 3 \times H_{img}\times W_{img}})
  * I-JEPA patch tokens (X \in \mathbb{R}^{B\times N \times d_v})
* Latents:

  * Learnable (L \in \mathbb{R}^{L \times d_f}) (broadcast over batch)
* LiDAR:

  * BEV feature map (\ell \in \mathbb{R}^{B\times C_\ell \times H \times W})
* Fusion:

  * Fusion dimension (d_f), output BEV channels (C)

---

## Mathematical formulation

### 1) Vision backbone (I-JEPA, frozen)

I-JEPA’s ViT produces patch tokens:
[
X = \mathrm{IJEPA}(I)\in \mathbb{R}^{B\times N\times d_v}.
]
We project them to the fusion dimension:
[
\tilde{X} = X W_{x}\quad \text{with } W_x \in \mathbb{R}^{d_v\times d_f}.
]

> We do **not** modify I-JEPA’s pretraining objective; we only use it as a fixed feature extractor.

### 2) Perceiver-style cross-attention (image → latents)

Let (L_0 \in \mathbb{R}^{L\times d_f}) be learnable latents (tiled over the batch). For each of (K) refinement layers:
[
\begin{aligned}
Q &= L_{k-1} W_Q,\qquad K = \tilde{X} W_K,\qquad V = \tilde{X} W_V,\
A &= \mathrm{softmax}!\left(\frac{QK^\top}{\sqrt{d_f}}\right),\
L_k &= A V,
\end{aligned}
]
with (W_Q,W_K,W_V\in \mathbb{R}^{d_f\times d_f}). After (K) layers,
[
\bar{g} = \mathrm{mean}_{\text{latent}}(L_K) \in \mathbb{R}^{B\times d_f}.
]

### 3) Broadcast & pointwise fusion with LiDAR

Tile (\bar{g}) over the LiDAR grid:
[
G = \mathrm{tile}(\bar{g}) \in \mathbb{R}^{B\times d_f \times H \times W}.
]
Concatenate with LiDAR BEV:
[
F_{\text{cat}} = [G;\ \ell] \in \mathbb{R}^{B\times (d_f+C_\ell)\times H\times W}.
]
Fuse with a per-pixel linear head (equivalently 1×1 conv):
[
F_{\text{bev}} = \phi(F_{\text{cat}}) \in \mathbb{R}^{B\times C\times H\times W}.
]

### 4) Top-down refinement (FPN-style)

Starting from the last LiDAR stage (C_5) and the fused map (F_{\text{bev}}), we produce a supervised BEV at higher resolution with learned upsamplers (\mathcal{U}) and 3×3 convs (\psi):
[
\begin{aligned}
P_5 &= \psi_5(F_{\text{bev}}),\
P_4 &= \psi_4!\big(\mathcal{U}(P_5)\big),\
P_3 &= \psi_3!\big(\mathcal{U}'(P_4)\big),
\end{aligned}
]
and expose:

* **High-res BEV** (P_3) to semantic/detection heads,
* **Low-res BEV** (F_{\text{bev}}) (for pooling/global features),
* **Camera spatial features** for visualization/losses if needed.

### 5) Global embedding for planning heads

We pool to a fixed (8\times 8) map and flatten:
[
z = \mathrm{vec}\big(\mathrm{Pool}*{8\times 8}(F*{\text{bev}})\big) \in \mathbb{R}^{B\times (C\cdot 8\cdot 8)}.
]

---

## Module diagram (internals)

```mermaid
flowchart TB
    subgraph Vision (frozen)
      V1[ViT-H/14 blocks] --> V2[patch tokens X (B×N×d_v)]
      V2 --> VX[Linear W_x → (B×N×d_f)]
    end

    subgraph Cross-Attn (Perceiver-style)
      L0[Learnable latents L (L×d_f)]
      VX --> CA1[Cross-Attn 1]
      L0 --> CA1
      CA1 --> CA2[Cross-Attn 2]
      CA2 --> GL[Mean over L → g ∈ R^{B×d_f}]
    end

    subgraph LiDAR
      LID1[timm BEV encoder] --> LID2[ℓ ∈ R^{B×C_ℓ×H×W}]
    end

    GL --> BR[Broadcast g → G ∈ R^{B×d_f×H×W}]
    LID2 --> CC
    BR --> CC[Concat [G;ℓ] → 1×1 Linear → F_bev ∈ R^{B×C×H×W}]

    subgraph Heads
      CC --> TD[Top-down refine]
      TD --> HR[High-res BEV (supervision)]
      CC --> PO[Pool 8×8 → z]
    end
```

---

## Interfaces & expected shapes (conceptual)

* `IJEPABackbone(image) → tokens X or fmap`: returns patch tokens (sequence) or folded map (both supported by fusion).
* `timm` LiDAR encoder returns multi-stage list; we use the last stage (\ell).
* `CrossAttentionFusion(tokens, ℓ) → F_bev`.
* `TransfuserBackbone.forward` returns **(high-res BEV, low-res BEV, camera grid)** to match TransFuser expectations.

---

## Why Perceiver-style latent cross-attention?

* **Asymmetric modalities**: camera = sequence of patch tokens; LiDAR = spatial grid. Latent queries avoid brittle reshaping/force-gridding.
* **Capacity control**: latent count (L) and width (d_f) bound attention compute (\mathcal{O}(BLN)).
* **Strong inductive bias**: we distill **global** camera context (g), then fuse with BEV **locally** via 1×1 mixing.

> Compare to direct token-grid alignment: this remains robust across aspect ratios, crops, or dynamic token counts (N).

---

## Wide-FOV camera support (do **not** warp)

**Requirement:** use the **entire** wide-FOV frame without aspect distortion. Two robust options:

1. **Aspect-preserving resize + pad (letterbox)**
   Let ((H_0,W_0)) be input size. Choose scale (s) so that
   [
   \max!\big(\lfloor s H_0 \rfloor, \lfloor s W_0 \rfloor\big)=S,\quad S \in 14\mathbb{Z},
   ]
   then **pad** to ((S,S)). That preserves FOV and yields a grid divisible by the ViT patch size (14). No warping.

2. **Rectangular tokens (if the ViT supports it)**
   Many ViTs interpolate positional embeddings to arbitrary ((H',W')) divisible by 14. Choose ((H',W')) maintaining aspect ratio and let I-JEPA handle pos-embed interpolation internally.

Either way, **no cropping/warping**. Keep a record of the letterbox mask if any view-frustum losses need it later.

---

## Design choices & ablations to run

* **Latent count (L)**: 32/64/128. Larger (L) increases capacity but adds (\mathcal{O}(L N)) memory/time.
* **Fusion width (d_f)**: 256 vs 512.
* **Where to fuse**: last LiDAR stage (current) vs multi-scale (add lateral fusions).
* **Pooling scheme**: mean over latents vs attention-pooling with a special learnable token.
* **Fine-tune vs frozen I-JEPA**: start frozen; later unfreeze top blocks with low LR.
* **Temporal context**: extend latents to attend over ([t-k,\dots,t]) image tokens if multi-frame camera inputs are available.

---

## Complexity

* Attention (per layer): (\mathcal{O}(B\cdot L\cdot N \cdot d_f)) for the matrix product (QK^\top), plus (\mathcal{O}(B\cdot L\cdot N \cdot d_f)) for (AV).
* Pointwise fusion: linear in (B\cdot H\cdot W\cdot(d_f+C_\ell)).
* Memory critical paths: keys/values for image tokens; keep (N) moderate by choosing patch stride/size sensibly.

---

## Integration points

* **Config**: `config.backbone ∈ {"resnet","ijepa"}`; `ijepa_model_id` points to local checkpoint.
* **Backbone**: two **disjoint** code-paths; I-JEPA branch never calls ResNet-specific iteration.
* **Returns**: `(bev_feature_upscale, bev_feature, image_feature_grid)` for compatibility with existing heads.
* **SLURM/Hydra**: override `agent.config.backbone=ijepa`.

---

## Known issues & current status

* **Preprocessing (wide FOV):** current path that force-resizes to a square must be replaced with **aspect-preserving letterbox or rectangular tokens** as above.
* **Single-scale fusion:** present fusion uses last LiDAR stage only; multi-scale cross-attention may further improve near-field geometry and far-field semantics.
* **Frozen vision tower:** fine-tuning could yield gains but risks distribution shift; schedule careful LR and regularization.
* **Temporal LiDAR:** current fusion is frame-wise; if (T>1), consider temporal pooling or temporal latents.

---

## Immediate next steps

1. **Implement letterbox/rectangular input** to preserve the wide camera FOV; guarantee both sides divisible by 14.
2. **Ablation grid:** (L\in{32,64,128}), (d_f\in{256,512}), with/without fine-tuning top-k ViT blocks.
3. **Multi-scale fusion (optional):** add lateral 1×1 projections from intermediate LiDAR stages with small cross-attn blocks.
4. **Instrumentation:** log latent attention maps over tokens; sanity-check that (g) attends to drivable space, lane boundaries, actors.
5. **Evaluation:** replicate TransFuser baselines; report planning, intervention, and PD metrics; stratify by lighting/weather/FOV.

---

## Reference ResNet path (for parity)

The original TransFuser branch performs multi-scale feature exchange via small GPT-style blocks operating on pooled grids from image/LiDAR at 4 scales, then uses FPN-style top-down fusion and global pooling. Keep this intact for **A/B** comparisons.

---

## Minimal pseudocode (fusion core)

```python
# I-JEPA tokens
X = ijepa(image)                        # (B, N, d_v)
X = X @ W_x                             # (B, N, d_f)

# Latent cross-attention (K layers)
L = L0.expand(B, -1, -1)                # (B, L, d_f)
for k in range(K):
    Q = L @ W_Q; K_ = X @ W_K; V = X @ W_V
    A = softmax(Q @ K_.transpose(-1,-2) / sqrt(d_f))
    L = A @ V

g = L.mean(dim=1)                       # (B, d_f)
G = g[:, :, None, None].expand(-1, -1, H, W)  # (B, d_f, H, W)

# LiDAR BEV map
ℓ = lidar(bev_input)                    # (B, C_ℓ, H, W)

# Pointwise fusion
F = concat([G, ℓ], dim=1)               # (B, d_f + C_ℓ, H, W)
F_bev = Linear_1x1(F)                   # (B, C, H, W)
```

---

## Bibliography (primary sources)

* **I-JEPA (Image-based Joint-Embedding Predictive Architecture).**
  Mahmoud Assran et al., *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture*, CVPR 2023. ([ResearchGate][1])

* **Perceiver (latent cross-attention).**
  Andrew Jaegle et al., *Perceiver: General Perception with Iterative Attention*, ICML 2021.

* **Perceiver IO (general I/O for latents).**
  Andrew Jaegle et al., *Perceiver IO: A General Architecture for Structured Inputs & Outputs*, NeurIPS 2021.

* **TransFuser (sensor fusion via transformers).**
  Kashyap Chitta et al., *TransFuser: Imitation with Transformer-Based Sensor Fusion*, arXiv:2205.15997.

* **Feature Pyramid Networks (top-down refinement).**
  Tsung-Yi Lin et al., *Feature Pyramid Networks for Object Detection*, CVPR 2017. ([CVF Open Access][2])

---

## Appendix: configuration knobs

* `backbone`: `"resnet"` or `"ijepa"`.
* `ijepa_model_id`: local path to ViT-H/14 I-JEPA weights.
* `fusion_dim (d_f)`, `num_latents (L)`, `num_layers (K)`, `num_heads`.
* `bev_features_channels (C)`, `bev_upsample_factor`, `bev_down_sample_factor`.
* `lidar_architecture`, `lidar_seq_len`.
* **FOV policy**: `preserve_aspect=True`, `pad_value=0 or mean`, `size_multiple=14`.

---

### TL;DR

* **Frozen I-JEPA** → strong image tokens
* **Latent cross-attention** → compact camera context (g)
* **Broadcast + pointwise fusion** with LiDAR BEV → **clean BEV features** for TransFuser heads
* Preserve **wide FOV** via **letterbox** or **rectangular tokens**, never warp.

[1]: https://www.researchgate.net/publication/353470279_Adaptive_Feature_Pyramid_Networks_for_Object_Detection?utm_source=chatgpt.com "Adaptive Feature Pyramid Networks for Object Detection"
[2]: https://openaccess.thecvf.com/CVPR2017.py?utm_source=chatgpt.com "CVPR 2017 Open Access Repository"
