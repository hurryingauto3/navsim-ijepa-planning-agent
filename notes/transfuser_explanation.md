I'll analyze the Transfuser agent code in detail. Let me start by reading through all the key files. 

Read [](file:///scratch/ah7072/navsim/navsim/agents/transfuser/transfuser_agent.py)

Read [](file:///scratch/ah7072/navsim/navsim/agents/transfuser/transfuser_config.py#1-1), lines 1 to 100

Read [](file:///scratch/ah7072/navsim/navsim/agents/transfuser/transfuser_model.py#1-1), lines 1 to 150

Read [](file:///scratch/ah7072/navsim/navsim/agents/transfuser/transfuser_model.py#150-150), lines 150 to 197

Read [](file:///scratch/ah7072/navsim/navsim/agents/transfuser/transfuser_backbone.py#1-1), lines 1 to 200

Read [](file:///scratch/ah7072/navsim/navsim/agents/transfuser/transfuser_features.py#1-1), lines 1 to 200

Read [](file:///scratch/ah7072/navsim/navsim/agents/transfuser/transfuser_features.py#200-200), lines 200 to 419

Read [](file:///scratch/ah7072/navsim/navsim/agents/transfuser/transfuser_backbone.py#200-200), lines 200 to 901

Read [](file:///scratch/ah7072/navsim/navsim/agents/transfuser/transfuser_loss.py#1-1), lines 1 to 150

Read [](file:///scratch/ah7072/navsim/navsim/agents/transfuser/transfuser_loss.py#150-150), lines 150 to 159

Read [](file:///scratch/ah7072/navsim/navsim/agents/transfuser/transfuser_config.py#100-100), lines 100 to 100

Read [](file:///scratch/ah7072/navsim/navsim/agents/transfuser/transfuser_callback.py#1-1), lines 1 to 100

Now let me read the rest of the config and callback files: 

Read [](file:///scratch/ah7072/navsim/navsim/agents/transfuser/transfuser_config.py#100-100), lines 100 to 128

Read [](file:///scratch/ah7072/navsim/navsim/agents/transfuser/transfuser_callback.py#100-100), lines 100 to 231

Perfect! Now I have a complete picture of the Transfuser codebase. Let me provide you with a detailed explanation.

## **Detailed Transfuser Code Walkthrough**

This is a **multi-modal trajectory planning agent** that fuses camera images and LiDAR point clouds to predict ego vehicle trajectories, detect surrounding agents, and generate BEV semantic maps. Here's the complete breakdown:

---

## **1. Architecture Overview (transfuser_agent.py)**

The `TransfuserAgent` is the main entry point that inherits from `AbstractAgent`:

### Key Components:
- **Trajectory Planning**: Predicts 8 future poses (x, y, θ) over 4 seconds
- **Agent Detection**: Detects up to 30 surrounding vehicles with 2D bounding boxes
- **BEV Semantic Segmentation**: Predicts 7-class semantic map (roads, walkways, centerlines, objects, vehicles, pedestrians)

### Important Methods:
```python
get_sensor_config() # Requests camera (F0, L0, R0) + LiDAR at current timestep only
get_feature_builders() # Returns TransfuserFeatureBuilder (preprocesses inputs)
get_target_builders() # Returns TransfuserTargetBuilder (prepares ground truth)
forward() # Runs TransfuserModel inference
compute_loss() # Calls transfuser_loss() with Hungarian matching
```

---

## **2. Configuration (transfuser_config.py)**

The `TransfuserConfig` dataclass holds all hyperparameters:

### Vision/LiDAR Settings:
- **Camera**: 3 stitched views (L0, F0, R0) resized to 1024×256
- **LiDAR**: 64×64m square around ego, binned into 256×256 grid at 4 pixels/meter
- **LiDAR Features**: Split at 0.2m height → "above ground" histogram (below optional)

### Backbone Options:
- **`backbone="resnet"`**: Uses ResNet34 encoders + GPT fusion transformer
- **`backbone="ijepa"`**: Uses frozen I-JEPA (ViT-H/14) for vision + ResNet34 for LiDAR + cross-attention fusion

### Transformer Decoder:
- 3 layers, 8 heads, d_model=256, d_ffn=1024
- Queries: 1 trajectory query + 30 agent queries

### Loss Weights:
```python
trajectory_weight = 10.0
agent_class_weight = 10.0
agent_box_weight = 1.0
bev_semantic_weight = 10.0
```

---

## **3. Feature Building (transfuser_features.py)**

### **TransfuserFeatureBuilder.compute_features()**

**Camera Processing:**
```python
# 1. Crop side cameras to ensure 4:1 aspect ratio
l0 = cam_l0[28:-28, 416:-416]  # crop to 768×1200
f0 = cam_f0[28:-28]            # crop to 768×1600
r0 = cam_r0[28:-28, 416:-416]  # crop to 768×1200

# 2. Stitch horizontally: 768×4000 total
stitched = np.concatenate([l0, f0, r0], axis=1)

# 3. Resize to 256×1024
camera_feature = resize(stitched, (256, 1024))
```

**LiDAR Processing:**
```python
# 1. Filter points above 100m height and split at 0.2m
below = lidar_pc[z <= 0.2]
above = lidar_pc[z > 0.2]

# 2. Create 2D histogram in 64×64m ego frame
# Bins: 256×256 grid (4 pixels/meter)
hist = np.histogramdd(above[:, :2], bins=(xbins, ybins))
hist = np.clip(hist, 0, 5) / 5.0  # normalize to [0,1]

# 3. Optional: add "below ground" channel if use_ground_plane=True
```

**Ego Status:**
```python
status_feature = concat([
    driving_command,    # 4D one-hot (left/right/straight/stop)
    ego_velocity,       # 2D (vx, vy)
    ego_acceleration,   # 2D (ax, ay)
])  # Total: 8D vector
```

---

### **TransfuserTargetBuilder.compute_targets()**

**1. Trajectory Target:**
```python
# Extract 8 future poses from scene at 0.5s intervals
trajectory = scene.get_future_trajectory(num_poses=8)
# Returns: (8, 3) array of (x, y, θ) in ego frame
```

**2. Agent Targets (2D Bounding Boxes):**
```python
# Filter vehicles within LiDAR range [-32, 32]m
# Keep 30 nearest vehicles
# Output: (30, 5) array of (x, y, θ, length, width)
# Labels: (30,) binary mask (True for valid boxes)
```

**3. BEV Semantic Map:**
```python
# Render 7-class semantic map (128×256 pixels)
# Classes:
# 0: background
# 1: road polygons (lanes + intersections)
# 2: walkways
# 3: centerlines (linestrings)
# 4: static objects (cones, barriers)
# 5: vehicles (oriented boxes)
# 6: pedestrians (oriented boxes)

# Process:
# - Query map API for nearby polygons/linestrings
# - Transform to ego frame
# - Rasterize with OpenCV fillPoly/polylines
```

---

## **4. Model Architecture (transfuser_model.py)**

### **TransfuserModel** Flow:

```
Input:
  camera_feature:  (B, 3, 256, 1024)
  lidar_feature:   (B, 1, 256, 256)
  status_feature:  (B, 8)

↓ (Backbone)

Fused BEV Features:
  bev_feature_upscale: (B, 64, 128, 256)  # for semantic head
  bev_feature:         (B, 512, 8, 8)     # for transformer decoder

↓ (Transformer Decoder)

Queries → Memory Attention:
  - Memory: flattened 8×8 grid + status = 65 tokens
  - Queries: 1 trajectory + 30 agents = 31 queries
  
↓ (Task Heads)

Outputs:
  trajectory:         (B, 8, 3)         # x, y, θ for 8 timesteps
  agent_states:       (B, 30, 5)        # x, y, θ, L, W
  agent_labels:       (B, 30)           # binary classification logits
  bev_semantic_map:   (B, 7, 128, 256)  # 7-class logits
```

---

### **Key Components:**

**A. Transformer Decoder Setup:**
```python
# Positional embeddings for memory (spatial grid + status)
keyval_embedding = nn.Embedding(8*8 + 1, 256)

# Learned query embeddings (1 traj + 30 agents)
query_embedding = nn.Embedding(31, 256)

# Standard PyTorch decoder
tf_decoder = nn.TransformerDecoder(layer, num_layers=3)
```

**B. Trajectory Head:**
```python
# Simple MLP: query → hidden → 8×3 output
trajectory = MLP(query[0])  # (B, 1, 256) → (B, 8, 3)
trajectory[..., θ] = tanh(θ) * π  # constrain heading to [-π, π]
```

**C. Agent Head:**
```python
# Predict per-agent states and classification
agent_states = MLP(queries[1:31])  # (B, 30, 256) → (B, 30, 5)
agent_states[..., xy] = tanh(xy) * 32  # constrain to LiDAR range
agent_states[..., θ] = tanh(θ) * π

agent_labels = Linear(queries[1:31])  # (B, 30, 256) → (B, 30)
```

**D. BEV Semantic Head:**
```python
# Upsample high-res BEV features
bev_up = Upsample(bev_feature_upscale)  # (B, 64, 128, 256)
semantic_map = Conv2d(bev_up)           # (B, 7, 128, 256)
```

---

## **5. Backbone Fusion (transfuser_backbone.py)**

### **ResNet Backbone (Original):**

```python
# Multi-scale fusion with GPT transformers
for i in range(4):  # 4 encoder stages
    img_feats = image_encoder.stage_i(img_feats)
    lid_feats = lidar_encoder.stage_i(lid_feats)
    
    # GPT fusion at each scale
    img_feats, lid_feats = GPT_i(
        avgpool(img_feats),   # 8×32 spatial
        avgpool(lid_feats),   # 8×8 spatial
    )
    
    # Add residual connections back to full resolution
    img_feats = img_feats + interpolate(fused_img)
    lid_feats = lid_feats + interpolate(fused_lid)

# Final: lidar_feats is (B, 512, 8, 8)
```

**GPT Module:**
- Learnable positional embeddings for 8×32 (img) + 8×8 (lidar) grids
- Multi-layer self-attention over concatenated tokens
- Splits output back into image/lidar spatial grids

---

### **I-JEPA Backbone (Your Addition):**

```python
# 1. Vision encoder (frozen I-JEPA ViT-H/14)
img_patches = IJEPABackbone(camera)  # (B, 1280, H', W')

# 2. LiDAR encoder (trainable ResNet34)
lidar_feats = lidar_encoder(lidar)[-1]  # (B, 512, 8, 8)

# 3. Cross-attention fusion
fused_bev = CrossAttentionFusion(
    image_features=img_patches.flatten(2,3).transpose(1,2),  # (B, N, 1280)
    lidar_features=lidar_feats,  # (B, 512, 8, 8)
)  # → (B, 64, 8, 8)
```

**IJEPABackbone:**
```python
# 1. Resize to 224×224 (I-JEPA expected input)
x = interpolate(camera, size=(224, 224))

# 2. Normalize with I-JEPA stats
x = (x / 255.0 - mean) / std

# 3. Extract patch embeddings
tokens = vision_tower(x).last_hidden_state  # (B, 257, 1280) w/ CLS

# 4. Drop CLS and reshape to spatial grid
patches = tokens[:, 1:, :]  # (B, 256, 1280)
feat = patches.reshape(B, 16, 16, 1280).permute(0, 3, 1, 2)
```

**CrossAttentionFusion (Perceiver-style):**
```python
# 1. Project I-JEPA features
img_tokens = Linear(img_patches)  # (B, N, 1280) → (B, N, 256)

# 2. Learnable latent queries
latents = nn.Parameter(randn(64, 256))

# 3. Cross-attend latents → image tokens
for _ in range(2):
    latents = MultiheadAttention(
        query=latents,        # (B, 64, 256)
        key=img_tokens,       # (B, N, 256)
        value=img_tokens,
    )

# 4. Pool to global camera vector
cam_vec = latents.mean(dim=1)  # (B, 256)

# 5. Broadcast over LiDAR BEV grid
cam_grid = cam_vec.expand(B, 256, 8, 8)

# 6. Fuse via concatenation + linear
fused = Linear(concat([cam_grid, lidar_feats], dim=1))  # (B, 64, 8, 8)
```

---

## **6. Loss Function (transfuser_loss.py)**

### **Trajectory Loss:**
```python
trajectory_loss = L1(pred_trajectory, gt_trajectory)
# Simple L1 distance on (x, y, θ)
```

### **Agent Detection Loss (Hungarian Matching):**
```python
# 1. Build cost matrix (B, 30_pred, 30_gt)
ce_cost = BCE_cost(pred_labels, gt_valid)
l1_cost = L1_cost(pred_boxes, gt_boxes) * gt_valid
cost = λ_class * ce_cost + λ_box * l1_cost

# 2. Hungarian matching per sample
indices = [linear_sum_assignment(cost[b]) for b in range(B)]

# 3. Compute losses only on matched pairs
matched_pred_boxes = pred_boxes[matched_indices]
matched_gt_boxes = gt_boxes[matched_indices]

box_loss = L1(matched_pred_boxes, matched_gt_boxes) * gt_valid
class_loss = BCE(pred_labels[matched_indices], gt_valid)
```

**Key Insight**: This is DETR-style set prediction—no NMS needed, and order-invariant.

### **BEV Semantic Loss:**
```python
bev_loss = CrossEntropy(pred_map, gt_map)
# Standard pixel-wise classification
```

### **Total Loss:**
```python
loss = 10*traj + 10*agent_class + 1*agent_box + 10*bev_semantic
```

---

## **7. Visualization Callback (transfuser_callback.py)**

Creates training/validation visualizations every epoch:

```
┌─────────────────────────────────┐
│  Stitched Camera (top)          │
├─────────────┬───────────────────┤
│ GT BEV Sem  │ Pred BEV Semantic │
│ (bot-left)  │ (bot-center)      │
├─────────────┴───────────────────┤
│ LiDAR Map + Boxes + Trajectory  │
│ (right): GT green, Pred red     │
└─────────────────────────────────┘
```

Logs to TensorBoard via `trainer.logger.experiment.add_image()`.

---

## **Key Architectural Decisions**

1. **Single-frame input**: Only uses current timestep (no temporal history), unlike many AV models.

2. **Separate encoders**: Image and LiDAR processed independently until fusion, allowing modality-specific augmentations.

3. **Multi-scale fusion** (ResNet): GPT transformers at 4 encoder stages enable early cross-modal interaction.

4. **Late fusion** (I-JEPA): Cross-attention only after extracting high-level features—simpler but potentially less expressive.

5. **Unified decoder**: Single transformer predicts all tasks (trajectory, detection, segmentation) from shared BEV representation.

6. **Hungarian matching**: Avoids hand-crafted anchor boxes and NMS for detection—learns set prediction end-to-end.

---

## **Critical Differences: ResNet vs I-JEPA Mode**

| Aspect | ResNet | I-JEPA |
|--------|--------|--------|
| **Vision Encoder** | ResNet34 (trained) | ViT-H/14 (frozen) |
| **Fusion Strategy** | Multi-scale GPT at 4 stages | Single-stage cross-attention |
| **Computational Cost** | Moderate | High (ViT-H is 630M params) |
| **Trainable Params** | ~50M | ~15M (fusion only) |
| **Feature Quality** | Task-specific | Generic SSL features |

Your I-JEPA variant is designed to test whether **frozen pretrained vision features** can match task-specific encoders—exactly aligning with your thesis on label-efficient learning.

---

## **Data Flow Summary**

```
Raw Sensors
  ↓
[Feature Builder]
  camera: 3 views → stitch → resize (256×1024×3)
  lidar: point cloud → 2D histogram (256×256×1)
  status: velocity + accel + command (8D)
  ↓
[Backbone]
  ResNet/I-JEPA → fused BEV features (8×8 grid)
  ↓
[Transformer Decoder]
  31 queries attend to 65 memory tokens (64 spatial + 1 status)
  ↓
[Task Heads]
  Query 0 → Trajectory (8×3)
  Queries 1-30 → Agent boxes (30×5) + labels (30×1)
  BEV features → Semantic map (128×256×7)
  ↓
[Loss]
  L1(trajectory) + Hungarian(agents) + CrossEntropy(BEV)
  ↓
[Backprop]
  (Only fusion + decoder trainable if I-JEPA mode)
```

---

This is a **dense, multi-task architecture** that jointly learns perception (BEV segmentation + detection) and planning (trajectory prediction) in a shared latent space. The I-JEPA variant you added is a clever way to decouple vision pretraining from task-specific fine-tuning—perfect for your label-efficiency experiments.

Let me know if you want me to dive deeper into any specific component!