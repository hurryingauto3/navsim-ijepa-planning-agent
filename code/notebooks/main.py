import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoModel
import hydra
from hydra.utils import instantiate
from tqdm import tqdm

from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig
from planning_agent.PlanningHead import PlanningHead
from planning_agent.NavsimTrajectoryDataset import NavsimTrajectoryDataset, collate_fn_skip_none

# ── Configuration ───────────────────────────────────────────────────────────────
SPLIT = "mini"
FILTER = "all_scenes"
DATA_ROOT = Path("../dataset")
NAVSIM_LOGS = DATA_ROOT / f"{SPLIT}_navsim_logs" / SPLIT
SENSOR_BLOBS = DATA_ROOT / f"{SPLIT}_sensor_blobs/sensor_blobs/{SPLIT}"

NUM_HISTORY_FRAMES = 4
NUM_FUTURE_FRAMES = 8
BATCH_SIZE = 64
EPOCHS = 1
LEARNING_RATE = 1e-4
HIDDEN_DIM = 256
MODEL_ID = "facebook/ijepa_vith14_1k"
IJEP_DIM = 1280
EGO_DIM = 8  # vel_x, vel_y, acc_x, acc_y, cmd_left, cmd_straight, cmd_right

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Data Loading ────────────────────────────────────────────────────────────────
hydra.initialize(
    config_path="../navsim/navsim/planning/script/config/common/train_test_split/scene_filter",
    version_base=None
)
scene_filter: SceneFilter = instantiate(hydra.compose(config_name=FILTER))
scene_filter.num_history_frames = NUM_HISTORY_FRAMES
scene_filter.num_future_frames = NUM_FUTURE_FRAMES

sensor_cfg = SensorConfig(
    cam_f0=True, # Only need front camera image
    cam_l0=False, cam_l1=False, cam_l2=False,
    cam_r0=False, cam_r1=False, cam_r2=False,
    cam_b0=False, lidar_pc=False
)
                          
scene_loader = SceneLoader(
    data_path=NAVSIM_LOGS,
    original_sensor_path=SENSOR_BLOBS,
    scene_filter=scene_filter,
    synthetic_sensor_path=None,
    synthetic_scenes_path=None,
    sensor_config=sensor_cfg,
)

processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=True)
dataset = NavsimTrajectoryDataset(
    scene_loader,
    processor,
    NUM_HISTORY_FRAMES,
    NUM_FUTURE_FRAMES,
    DEVICE
)
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    drop_last=True,
    collate_fn=collate_fn_skip_none,
    persistent_workers=True
)

# ── Model Setup ─────────────────────────────────────────────────────────────────
ijepa_encoder = AutoModel.from_pretrained(MODEL_ID).to(DEVICE)
ijepa_encoder.eval()
for param in ijepa_encoder.parameters():
    param.requires_grad = False

mlp_head = PlanningHead(
    ijep_dim=IJEP_DIM,
    ego_dim=EGO_DIM,
    hidden_dim=HIDDEN_DIM,
    output_dim=NUM_FUTURE_FRAMES * 3
).to(DEVICE)

# ── Training ─────────────────────────────────────────────────────────────────────
history = mlp_head.fit(
    dataloader=dataloader,
    ijepa_encoder=ijepa_encoder,
    device=DEVICE,
    epochs=EPOCHS,
    lr=LEARNING_RATE,
    save_dir="checkpoints",
    resume_from=None,
    checkpoint_interval=1,
    use_cls_token=True
)

print("Training finished. Avg losses per epoch:", history)