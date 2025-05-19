import torch
import os
import glob
import re
from datetime import datetime
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
from PIL import Image
import numpy as np
import numpy.typing as npt
from pathlib import Path
from transformers import AutoProcessor, AutoModel
from typing import Dict, List, Union, Tuple, Any, Optional

import pytorch_lightning as pl
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

# Correct import for TrajectorySampling
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import (
    AgentInput,
    Trajectory,
    EgoStatus,
    SensorConfig,
    Scene,
)

# Assuming convert_absolute_to_relative_se2_array is available (though not used in current target builder)
from nuplan.common.actor_state.state_representation import StateSE2
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
    convert_absolute_to_relative_se2_array,
)

from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)

import logging # Use standard logging
# Configure basic logging if not already configured
if not logging.getLogger('').handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

# --- Callbacks ---

class FirstBatchDebugger(pl.Callback):
    """
    Logs information about the first training batch.
    """
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # Only log on the very first optimisation step
        if trainer.global_step == 0 and batch_idx == 0:
            features, targets = batch

            # Assuming features/targets are dicts and contain expected keys
            try:
                # Check if trajectory_gt exists and has data
                if "trajectory_gt" in targets and targets["trajectory_gt"] is not None and targets["trajectory_gt"].numel() > 0:
                    # ----- ground-truth (absolute) -----
                    # Detach from graph and move to CPU safely
                    gt_abs = targets["trajectory_gt"][0].detach().cpu().numpy()        # (T,3)
                    logger.info("\n[DEBUG] FIRST-BATCH  GT  (first 3 poses): %s", gt_abs[:min(3, gt_abs.shape[0])]) # Print up to 3 poses

                    # ----- model output (relative) -----
                    with torch.no_grad():
                        # Call forward safely, detach, and move to CPU
                        predictions = pl_module.agent.forward(features)
                        if "trajectory" in predictions and predictions["trajectory"] is not None and predictions["trajectory"].numel() > 0:
                             pred_rel = predictions["trajectory"][0].detach().cpu()  # (T,3)
                             pred_rel = pred_rel.cpu().numpy()
                             logger.info("[DEBUG] FIRST-BATCH  PRED (relative, first 3): %s", pred_rel[:min(3, pred_rel.shape[0])]) # Print up to 3 poses

                             # ----- convert to absolute for easy comparison -----
                             # Get origin from GT absolute first pose (safe indexing)
                             origin = gt_abs[0:1, :]                    # (1,3)  current pose
                             pred_abs = pred_rel + origin               # broadcast over T

                             logger.info("[DEBUG] FIRST-BATCH  PRED (ABS, first 3): %s", pred_abs[:min(3, pred_abs.shape[0])]) # Print up to 3 poses
                        else:
                            logger.warning("[DEBUG] FIRST-BATCH: 'trajectory' not found or is empty in model predictions. Skipping prediction logging.")
                else:
                    logger.warning("[DEBUG] FIRST-BATCH: 'trajectory_gt' not found or is empty in targets. Skipping debugger.")
            except Exception as e:
                 logger.error(f"[DEBUG] FIRST-BATCH Debugger encountered an error: {e}", exc_info=True)


# Example BatchInspector (commented out by default)
# class BatchInspector(pl.Callback):
#     """
#     Logs information about batch tensors periodically.
#     """
#     def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
#         if batch_idx > 0 and batch_idx % 100 == 0: # Log every 100 batches (after the first)
#             features, targets = batch
#             loss = outputs # Assuming outputs is the loss tensor

#             logger.info(f"\n[Batch {batch_idx}] --- Tensor Shapes ---")
#             for key, tensor in features.items():
#                 logger.info(f"[Batch {batch_idx}] Features['{key}'] shape: {tensor.shape}, dtype: {tensor.dtype}")
#             for key, tensor in targets.items():
#                 logger.info(f"[Batch {batch_idx}] Targets['{key}'] shape: {tensor.shape}, dtype: {tensor.dtype}")
#             if isinstance(loss, torch.Tensor):
#                  logger.info(f"[Batch {batch_idx}] Loss shape: {loss.shape}, dtype: {loss.dtype}")
#             else:
#                  logger.info(f"[Batch {batch_idx}] Loss type: {type(loss)}")


#             logger.info(f"\n[Batch {batch_idx}] --- Sample Data (First Sample) ---")
#             # Safely move to CPU and convert to numpy for printing
#             if "ego_features" in features and features["ego_features"].shape[0] > 0:
#                 logger.info(f"[Batch {batch_idx}] First ego features: %s", features["ego_features"][0,:].detach().cpu().numpy())
#             if "trajectory_gt" in targets and targets["trajectory_gt"].shape[0] > 0:
#                  # Print first pose of the first trajectory
#                  logger.info(f"[Batch {batch_idx}] First target GT pose: %s", targets["trajectory_gt"][0, 0, :].detach().cpu().numpy())
#             # Add checks and logging for other features/targets as needed


# --- Feature Builders ---

class CameraImageFeatureBuilder(AbstractFeatureBuilder):
    """
    Feature builder for extracting the front camera image.
    Returns the raw image data as a tensor (CHW, float) to be processed by the agent's forward.
    """
    def get_unique_name(self) -> str:
        return "front_camera_image"

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        # Use the last camera frame for the current image
        if not agent_input.cameras or len(agent_input.cameras) == 0 or \
           not hasattr(agent_input.cameras[-1], "cam_f0") or \
           agent_input.cameras[-1].cam_f0.image is None:
            logger.warning(f"{self.get_unique_name()}: Front camera image missing in last history frame. Returning zero placeholder.")
            # Using a reasonable default size (e.g., 224x224) based on common vision models
            dummy_image = np.zeros((224, 224, 3), dtype=np.uint8)
            # Permute to CHW and convert to float. The HF processor expects uint8 PIL image or float tensor usually 0-1 or -1 to 1
            # Let's assume the input image is already float [0, 255] as per common practice in some datasets
            return {self.get_unique_name(): torch.from_numpy(dummy_image).permute(2, 0, 1).float()}

        image_np = agent_input.cameras[-1].cam_f0.image
        # Assuming image_np is HWC numpy array (uint8 or float). Convert to CHW float.
        # Assuming input is already float [0, 255]
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() # CHW, float [0, 255]?
        # If your image data is uint8, you might need to divide by 255 here: .float() / 255.0
        return {self.get_unique_name(): image_tensor}

class EgoFeatureBuilder(AbstractFeatureBuilder):
    """
    Feature builder for extracting historical ego status features (velocity, acceleration, driving command).
    Concatenates velocity, acceleration, and driving command from the last N history frames.
    """
    NUM_DRIVING_COMMANDS = 4 # Should match the definition elsewhere if used
    SINGLE_FRAME_EGO_DIM = 2 + 2 + NUM_DRIVING_COMMANDS # vx, vy, ax, ay, command_one_hot

    def __init__(self, num_history_frames: int = 4): # Add num_history_frames parameter
        """
        Initializes the EgoFeatureBuilder with history.
        :param num_history_frames: The number of history frames to include in the feature vector.
        """
        super().__init__()
        if num_history_frames < 1:
             raise ValueError("EgoFeatureBuilder requires num_history_frames >= 1")
        self._num_history_frames = num_history_frames
        self._total_ego_history_dim = self._num_history_frames * self.SINGLE_FRAME_EGO_DIM

    def get_unique_name(self) -> str:
        return "ego_features" # Keep the same key name as in your agent's forward method

    # Ego feature builder for num_history_frames
    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        # Check if there's enough history frames available (including the current frame at index -1)
        if not agent_input.ego_statuses or len(agent_input.ego_statuses) < self._num_history_frames:
            logger.warning(f"{self.get_unique_name()}: Insufficient ego history frames ({len(agent_input.ego_statuses)} vs {self._num_history_frames}). Returning zero placeholder of size {self._total_ego_history_dim}.")
            # Calculate the expected output size: (velocity + acceleration + command) * num_history_frames
            return {self.get_unique_name(): torch.zeros(self._total_ego_history_dim, dtype=torch.float32)}

        # Process the last num_history_frames (including the current frame at index -1)
        history_ego_features_list = []
        # Slice the last N frames [-self._num_history_frames:]
        # Iterate forward through this slice to maintain chronological order in the concatenated vector
        historical_statuses = agent_input.ego_statuses[-self._num_history_frames:]

        for i, ego_status in enumerate(historical_statuses):
            # Ensure velocity and acceleration are 2D
            velocity = torch.tensor(ego_status.ego_velocity, dtype=torch.float32) # Expected Shape [2]
            acceleration = torch.tensor(ego_status.ego_acceleration, dtype=torch.float32) # Expected Shape [2]

            if velocity.shape[-1] != 2 or acceleration.shape[-1] != 2:
                logger.warning(f"Builder: Velocity or acceleration not 2D for history frame offset {-self._num_history_frames + i}. Vel shape: {velocity.shape}, Accel shape: {acceleration.shape}. Using zeros for this frame.")
                velocity = torch.zeros(2, dtype=torch.float32)
                acceleration = torch.zeros(2, dtype=torch.float32)


            command_raw = ego_status.driving_command
            command_one_hot = torch.zeros(self.NUM_DRIVING_COMMANDS, dtype=torch.float32) # Shape [NUM_DRIVING_COMMANDS]

            # --- Robust command handling ---
            try:
                if isinstance(command_raw, np.ndarray):
                    if command_raw.size == 1:
                        command_index = int(command_raw.item())
                        if 0 <= command_index < self.NUM_DRIVING_COMMANDS: command_one_hot[command_index] = 1.0
                        else: logger.warning(f"Builder: Invalid command index {command_index} in array for history frame offset {-self._num_history_frames + i}.")
                    elif command_raw.size == self.NUM_DRIVING_COMMANDS:
                        command_one_hot = torch.from_numpy(command_raw).float()
                    else: logger.warning(f"Builder: Unexpected numpy array size {command_raw.shape} for history frame offset {-self._num_history_frames + i}.")
                elif isinstance(command_raw, (int, float)):
                    command_index = int(command_raw)
                    if 0 <= command_index < self.NUM_DRIVING_COMMANDS: command_one_hot[command_index] = 1.0
                    else: logger.warning(f"Builder: Invalid command index {command_index} for history frame offset {-self._num_history_frames + i}.")
                elif isinstance(command_raw, (list, tuple)) and len(command_raw) == self.NUM_DRIVING_COMMANDS:
                    command_one_hot = torch.tensor(command_raw, dtype=torch.float32)
                else: logger.warning(f"Builder: Unexpected command format {type(command_raw)} for history frame offset {-self._num_history_frames + i}. Using zero vector.")
                    command_one_hot = torch.zeros(self.NUM_DRIVING_COMMANDS, dtype=torch.float32) # Use zero vector on unexpected format
            except Exception as e:
                logger.error(f"Error processing driving command '{command_raw}' for history frame offset {-self._num_history_frames + i}: {e}. Using zero vector.", exc_info=True)
                command_one_hot = torch.zeros(self.NUM_DRIVING_COMMANDS, dtype=torch.float32)
            # --- End robust command handling ---

            # Concatenate features for the current historical frame [vel_x, vel_y, acc_x, acc_y, cmd_0..3]
            current_frame_features = torch.cat([velocity, acceleration, command_one_hot], dim=-1) # Shape [SINGLE_FRAME_EGO_DIM]
            if current_frame_features.shape[-1] != self.SINGLE_FRAME_EGO_DIM:
                 logger.error(f"Builder: Mismatch in single frame feature dim {current_frame_features.shape[-1]} for history frame offset {-self._num_history_frames + i}. Expected {self.SINGLE_FRAME_EGO_DIM}. Skipping this frame.")
                 continue # Skip this frame if features are malformed

            history_ego_features_list.append(current_frame_features)

        if len(history_ego_features_list) != self._num_history_frames:
             # This indicates some frames were skipped due to internal errors
             logger.error(f"Builder: Mismatch in *successfully processed* history frames ({len(history_ego_features_list)} vs {self._num_history_frames}). Returning zero placeholder of size {self._total_ego_history_dim}.")
             return {self.get_unique_name(): torch.zeros(self._total_ego_history_dim, dtype=torch.float32)}

        # Concatenate features ACROSS the history dimension (flattening the history)
        ego_features = torch.cat(history_ego_features_list, dim=-1) # Shape [num_history_frames * SINGLE_FRAME_EGO_DIM]

        # Final safety check on shape - should match the expected concatenated size
        if ego_features.shape[-1] != self._total_ego_history_dim:
            logger.error(f"Builder: Final ego features dim mismatch {ego_features.shape[-1]} vs expected {self._total_ego_history_dim}. Returning zero placeholder.")
            return {self.get_unique_name(): torch.zeros(self._total_ego_history_dim, dtype=torch.float32)}

        return {self.get_unique_name(): ego_features}


# --- Target Builder (Cleaned) ---

class TrajectoryTargetBuilderGT(AbstractTargetBuilder):
    """
    Target builder for extracting the ground truth future trajectory poses.
    Returns absolute poses.
    """
    # CHANGE: Removed num_history_frames from constructor as it's not used for returning absolute targets
    def __init__(self, trajectory_sampling: TrajectorySampling):
        """
        Initializes the target builder.
        :param trajectory_sampling: Trajectory sampling specification (for num_poses).
        """
        super().__init__()
        self._trajectory_sampling = trajectory_sampling


    def get_unique_name(self) -> str:
        return "trajectory_gt"

    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        """
        Return the future trajectory in absolute map coordinates.
        """
        # 1) get the future trajectory from the scene
        future_traj: Trajectory = scene.get_future_trajectory(
            num_trajectory_frames=self._trajectory_sampling.num_poses
        )

        # 2) fallback to zeros if trajectory missing / too short
        if (future_traj is None or future_traj.poses is None or
            future_traj.poses.shape[0] < self._trajectory_sampling.num_poses):
            logger.warning(f"{self.get_unique_name()}: insufficient future poses "
                f"({0 if future_traj is None else future_traj.poses.shape[0]} "
                f"vs {self._trajectory_sampling.num_poses}). Returning zeros.")
            return {self.get_unique_name():
                    torch.zeros(self._trajectory_sampling.num_poses, 3,
                                dtype=torch.float32)}

        # 3) use absolute poses directly
        gt_abs = torch.as_tensor(future_traj.poses, dtype=torch.float32)

        # 4) final safety-check on shape
        if gt_abs.shape != (self._trajectory_sampling.num_poses, 3):
            logger.warning(f"{self.get_unique_name()}: shape mismatch {gt_abs.shape}, "
                "returning zeros.")
            gt_abs = torch.zeros(self._trajectory_sampling.num_poses, 3,
                                dtype=torch.float32)

        return {self.get_unique_name(): gt_abs}


# --- Agent ---

class IJEPAPlanningAgent(AbstractAgent, pl.LightningModule): # Inherit from LightningModule
    """
    NAVSIM Agent combining I-JEPA encoding and an MLP planning head,
    compatible with the standard NAVSIM training framework, as a PyTorch Lightning Module.
    """
    # --- Constants ---
    # NUM_FUTURE_FRAMES is now derived from trajectory_sampling
    IJEP_DIM = 1280
    SINGLE_FRAME_EGO_DIM = EgoFeatureBuilder.SINGLE_FRAME_EGO_DIM # Match the builder
    HIDDEN_DIM = 256 # Using 2x256 MLP based on local Exp 5 preference
    NUM_DRIVING_COMMANDS = EgoFeatureBuilder.NUM_DRIVING_COMMANDS # Match the builder
    # --- End Constants ---

    def __init__(
        self,
        mlp_weights_path: Optional[str] = None,
        ijepa_model_id: str = "facebook/ijepa_vith14_1k",
        trajectory_sampling: TrajectorySampling = TrajectorySampling(
            time_horizon=4, interval_length=0.5
        ),
        num_history_frames: int = 4,
        use_cls_token_if_available: bool = True,
        # requires_scene is typically False for agents that don't need direct scene access in forward/compute_loss
        # The builders provide necessary data.
        requires_scene: bool = False,
        learning_rate: float = 1e-4,
        loss_criterion: str = "l1", # Base loss type (l1 or mse)
        accel_loss_weight: float = 0.0, # Weight for acceleration penalty
        jerk_loss_weight: float = 0.0,  # Weight for jerk penalty
        weight_decay: float = 0.0,      # Weight decay for optimizer
        max_epochs: int = 50, # Needed for scheduler T_max
        use_cosine_scheduler: bool = True, # Option to use scheduler
        cosine_eta_min_ratio: float = 0.01 # Factor for eta_min = lr * ratio
    ):
        # Initialize AbstractAgent part
        super().__init__(
            trajectory_sampling=trajectory_sampling,
            requires_scene=requires_scene
        )
        # Initialize LightningModule part (implicitly handled by pytorch_lightning)

        # Store parameters
        self._num_history_frames = num_history_frames
        # Calculate the effective Ego input dimension based on history frames
        self.EGO_DIM = self.SINGLE_FRAME_EGO_DIM * self._num_history_frames

        self._mlp_weights_path_config = mlp_weights_path
        self._ijepa_model_id = ijepa_model_id
        self._use_cls_token = use_cls_token_if_available
        self._learning_rate = learning_rate
        self._loss_criterion_type = loss_criterion
        self._accel_loss_weight = accel_loss_weight
        self._jerk_loss_weight = jerk_loss_weight
        self._weight_decay = weight_decay
        self._max_epochs = max_epochs
        self._use_cosine_scheduler = use_cosine_scheduler
        self._cosine_eta_min = self._learning_rate * cosine_eta_min_ratio # Calculate eta_min

        # Total input dimension for MLP: Visual features + Ego History features
        mlp_input_dim = self.IJEP_DIM + self.EGO_DIM
        logger.info(f"MLP input dimension: {mlp_input_dim}")

        # Define MLP (Using 2x256 based on local Exp 5 preference)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(mlp_input_dim, self.HIDDEN_DIM), # Corrected input dim
            torch.nn.ReLU(),
            torch.nn.LayerNorm(self.HIDDEN_DIM),
            torch.nn.Linear(self.HIDDEN_DIM, self.HIDDEN_DIM),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(self.HIDDEN_DIM),
            # Removed the 3rd layer based on local Exp 6 results
            torch.nn.Linear(self.HIDDEN_DIM, self._trajectory_sampling.num_poses * 3),
        )

        # Define Base Loss Criterion
        if self._loss_criterion_type.lower() == "l1":
            self.base_criterion = torch.nn.L1Loss()
        elif self._loss_criterion_type.lower() == "mse":
            self.base_criterion = torch.nn.MSELoss()
        else:
            raise ValueError(f"Unsupported base loss: {self._loss_criterion_type}. Choose 'l1' or 'mse'.")

        # Store trajectory interval for derivative calculations
        self._interval_length = self._trajectory_sampling.interval_length
        if self._interval_length <= 0:
             raise ValueError("Trajectory sampling interval_length must be positive for derivative calculations.")

        # Flag to track if I-JEPA encoder is initialized
        self._ijepa_initialized = False


    def name(self) -> str:
        return self.__class__.__name__

    def initialize(self) -> None:
        """
        Initializes components that might need delayed loading (like models).
        Called by the training framework.
        """
        if self._ijepa_initialized:
            logger.info(f"{self.name()} already initialized.")
            return

        logger.info(f"Initializing {self.name()}...")
        # These settings are good, but might be better placed in the trainer setup
        # torch.backends.cudnn.benchmark = True
        # torch.set_float32_matmul_precision("high")

        device = self.device # Use self.device provided by LightningModule
        logger.info(f"Using device: {device}")
        self.to(device) # Ensure MLP is on the correct device

        # Load I-JEPA Processor and *Frozen* Encoder
        try:
            logger.info(f"Loading I-JEPA model: {self._ijepa_model_id}")
            self._processor = AutoProcessor.from_pretrained(self._ijepa_model_id, use_fast=True)
            # Move encoder to device immediately after loading
            self._ijepa_encoder = AutoModel.from_pretrained(self._ijepa_model_id).to(device)
            # Freeze encoder parameters
            for param in self._ijepa_encoder.parameters():
                param.requires_grad = False
            self._ijepa_encoder.eval() # Ensure encoder is in eval mode
            logger.info(f"Loaded and froze I-JEPA encoder: {self._ijepa_model_id}")
        except Exception as e:
            # Log error and raise a specific runtime error
            logger.error(f"Failed to load I-JEPA model/processor: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load I-JEPA model/processor: {e}") from e

        # Determine Feature Extraction Method
        self._feature_extraction_method = "Mean Pooling" # Default
        if self._use_cls_token:
             try:
                 with torch.no_grad():
                     # Use a dummy batch size > 0
                     dummy_batch_size = 2 # Use a small batch size > 1 for robustness check
                     dummy_input = torch.zeros(dummy_batch_size, 3, 224, 224).to(device) # Assuming 224x224 input
                     outputs = self._ijepa_encoder(pixel_values=dummy_input)
                     pooler = getattr(outputs, "pooler_output", None)
                     hidden = getattr(outputs, "last_hidden_state", None)

                     if pooler is not None and pooler.shape[-1] == self.IJEP_DIM and pooler.shape[0] == dummy_batch_size:
                         self._feature_extraction_method = "Pooler Output"
                         logger.info(f"Using Pooler Output for I-JEPA features. Shape: {pooler.shape}")
                     elif hidden is not None and hidden.shape[-1] == self.IJEP_DIM and hidden.shape[1] > 0 and hidden.shape[0] == dummy_batch_size:
                          # Assumes [CLS] is the first token
                          self._feature_extraction_method = "CLS Token"
                          logger.info(f"Using CLS Token (first token) for I-JEPA features. Shape: {hidden[:, 0, :].shape}")
                     else:
                          logger.warning("I-JEPA Pooler/CLS token unavailable or incorrect shape/batch size; defaulting to Mean Pooling")
             except Exception as e:
                  logger.warning(f"Error checking I-JEPA features, defaulting to Mean Pooling: {e}", exc_info=True)

        if self._feature_extraction_method == "Mean Pooling":
            logger.info("Using Mean Pooling over last_hidden_state for I-JEPA features.")


        # Load Optional MLP Weights (Improved Robustness and Logging)
        if self._mlp_weights_path_config:
            weights_path = Path(self._mlp_weights_path_config)
            if weights_path.is_file():
                logger.info(f"Attempting to load MLP weights from: {weights_path}")
                try:
                    raw_state_dict = torch.load(weights_path, map_location=device)
                    # Check for typical LightningModule state_dict structure ('state_dict' key)
                    ckpt = raw_state_dict.get("state_dict", raw_state_dict)

                    mlp_state_dict = {}
                    # Look for keys specifically for the MLP within the agent ('agent.mlp.' prefix)
                    found_prefixed_keys = False
                    for k, v in ckpt.items():
                        if k.startswith("agent.mlp."):
                            mlp_state_dict[k.replace("agent.mlp.", "")] = v
                            found_prefixed_keys = True

                    if not mlp_state_dict and not found_prefixed_keys:
                         # Fallback: If no "agent.mlp." keys found, assume the checkpoint is just the MLP state_dict
                         logger.warning("No 'agent.mlp.' keys found in checkpoint. Attempting direct load of the whole checkpoint as MLP state_dict.")
                         mlp_state_dict = ckpt
                         # Clean up potential "module." prefix if using DataParallel/DDP
                         mlp_state_dict = {k.replace('module.', ''): v for k, v in mlp_state_dict.items()}

                    if not mlp_state_dict:
                         raise ValueError("No MLP weights found in checkpoint after trying prefixes and direct load.")

                    # Load the state_dict, allowing for missing/unexpected keys (strict=False)
                    info = self.mlp.load_state_dict(mlp_state_dict, strict=False)
                    logger.info(f"MLP weights loaded successfully with {len(mlp_state_dict)} tensors.")
                    if info.missing_keys:
                        logger.warning(f"MLP load missing keys: {info.missing_keys}")
                    if info.unexpected_keys:
                        logger.warning(f"MLP load unexpected keys: {info.unexpected_keys}")

                except Exception as e:
                    logger.warning(f"Failed to load MLP weights from {weights_path} ({e}). Starting fresh.", exc_info=True)
            else:
                logger.warning(f"MLP weights file not found: {weights_path}. Starting fresh.")
        else:
            logger.info("No MLP weights path provided. Starting training from scratch.")

        # Ensure MLP is on the correct device after potential loading
        self.mlp.to(device)
        self.mlp.train() # MLP should be in train mode during training

        self._ijepa_initialized = True
        logger.info(f"{self.name()} initialization complete.")


    def get_sensor_config(self) -> SensorConfig:
        # Only front camera needed by builders
        return SensorConfig(cam_f0=True, cam_l0=False, cam_l1=False, cam_l2=False, cam_r0=False, cam_r1=False, cam_r2=False, cam_b0=False, lidar_pc=False)

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        # Return the EgoFeatureBuilder configured for history
        return [
            CameraImageFeatureBuilder(),
            EgoFeatureBuilder(num_history_frames=self._num_history_frames)
        ]

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        # CHANGE: Removed num_history_frames from TrajectoryTargetBuilderGT constructor call
        return [TrajectoryTargetBuilderGT(trajectory_sampling=self._trajectory_sampling)]

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Check if essential components are initialized
        if not self._ijepa_initialized or self._processor is None or self._ijepa_encoder is None or self.mlp is None:
            # If not initialized, perform initialization (might happen during inference if not trained first)
            # Note: This is a simplified approach; a proper training loop initializes before training.
            # For pure inference, you should call initialize explicitly before the first forward pass.
            if not self._ijepa_initialized:
                 logger.warning("Agent components not initialized. Calling initialize().")
                 # Attempt initialization. If it fails, the subsequent try/except will handle it.
                 try:
                      self.initialize()
                 except Exception as init_err:
                      logger.error(f"Initialization failed in forward pass: {init_err}. Cannot proceed.", exc_info=True)
                      # Return zero predictions gracefully if initialization fails
                      # Need a way to determine batch size if features is empty or malformed
                      try:
                           batch_size = next(iter(features.values())).shape[0]
                      except Exception:
                           batch_size = 1 # Default to 1 if batch size cannot be determined
                      zero_preds = torch.zeros(batch_size, self._trajectory_sampling.num_poses, 3, dtype=torch.float32, device=self.device)
                      return {"trajectory": zero_preds}

        device = self.device # Use self.device
        # Ensure modules are on the correct device and encoder is in eval mode
        self.to(device)
        self._ijepa_encoder.eval() # Keep encoder in eval mode
        # MLP mode (train during training_step, eval during validation_step/inference)
        # Lightning handles train/eval mode toggling for self.train()/self.eval()


        # Get features from builders' output
        try:
            # Using the key defined in EgoFeatureBuilder
            image_tensor_raw = features["front_camera_image"].to(device)
            ego_history_features_tensor = features["ego_features"].to(device) # Using "ego_features" key
        except KeyError as e:
             logger.error(f"Missing required feature key in forward: {e}. Expected 'front_camera_image' and 'ego_features'.", exc_info=True)
             # Return zero predictions gracefully if essential features are missing
             # Need a way to determine batch size
             try:
                  # Try getting batch size from available features
                  batch_size = image_tensor_raw.shape[0] if 'image_tensor_raw' in locals() and image_tensor_raw.ndim >= 1 else (ego_history_features_tensor.shape[0] if 'ego_history_features_tensor' in locals() and ego_history_features_tensor.ndim >= 1 else next(iter(features.values())).shape[0])
             except Exception:
                  batch_size = 1 # Default to 1 if batch size cannot be determined
             zero_preds = torch.zeros(batch_size, self._trajectory_sampling.num_poses, 3, dtype=torch.float32, device=device)
             return {"trajectory": zero_preds}


        # Preprocess image using Hugging Face processor
        processed_pixel_values = None
        try:
            # The processor expects PIL Images or numpy arrays (HWC uint8 typically) or torch tensors.
            # Assuming image_tensor_raw is BCHW float [0, 255], convert to BHWC uint8 numpy for processor
            if image_tensor_raw.dtype != torch.float32:
                 logger.warning(f"Image tensor dtype is {image_tensor_raw.dtype}, expected float32. Attempting conversion.")
                 image_tensor_raw = image_tensor_raw.float()
            if image_tensor_raw.max() > 1.0: # Assuming max value indicates range [0, 255]
                 image_batch_np = image_tensor_raw.permute(0, 2, 3, 1).byte().cpu().numpy() # Convert to BHWC uint8
            else: # Assuming range [0, 1]
                 image_batch_np = (image_tensor_raw * 255.0).permute(0, 2, 3, 1).byte().cpu().numpy()


            image_list_pil = [Image.fromarray(img) for img in image_batch_np]
            # Processor expects RGB PIL images. Assuming input is RGB.
            processor_output = self._processor(images=image_list_pil, return_tensors="pt")
            processed_pixel_values = processor_output['pixel_values'].to(device)
            if processed_pixel_values is None: raise ValueError("Processor returned None pixel_values.")
        except Exception as e:
            logger.error(f"Error during image processing: {e}. Returning zero predictions.", exc_info=True)
            # Need a way to determine batch size if image_tensor_raw processing failed early
            batch_size = image_tensor_raw.shape[0] if 'image_tensor_raw' in locals() and image_tensor_raw.ndim >= 1 else 1
            zero_preds = torch.zeros(batch_size, self._trajectory_sampling.num_poses, 3, dtype=torch.float32, device=device)
            return {"trajectory": zero_preds}

        # Extract features with Frozen I-JEPA
        visual_features = None
        with torch.no_grad(): # Ensure no gradients are calculated for the frozen encoder
            try:
                ijepa_outputs = self._ijepa_encoder(pixel_values=processed_pixel_values)
                if self._feature_extraction_method == "Pooler Output":
                    visual_features = ijepa_outputs.pooler_output
                    if visual_features is None: raise ValueError("Pooler output is None.")
                elif self._feature_extraction_method == "CLS Token":
                    # take the [CLS] embedding
                    # Ensure last_hidden_state exists and has at least the CLS token
                    if hasattr(ijepa_outputs, 'last_hidden_state') and ijepa_outputs.last_hidden_state is not None and ijepa_outputs.last_hidden_state.shape[1] > 0:
                        visual_features = ijepa_outputs.last_hidden_state[:, 0, :]
                    else:
                        # Fallback if last_hidden_state is missing or empty (shouldn't happen for standard models but safe)
                        logger.warning("I-JEPA last_hidden_state missing or empty, cannot get CLS token. Using Mean Pooling fallback.")
                        if hasattr(ijepa_outputs, 'last_hidden_state') and ijepa_outputs.last_hidden_state is not None:
                            visual_features = ijepa_outputs.last_hidden_state.mean(dim=1)
                        else:
                            raise ValueError("I-JEPA last_hidden_state is missing.")
                else:  # Mean Pooling
                    if hasattr(ijepa_outputs, 'last_hidden_state') and ijepa_outputs.last_hidden_state is not None:
                        visual_features = ijepa_outputs.last_hidden_state.mean(dim=1)
                    else:
                         raise ValueError("I-JEPA last_hidden_state is missing.")

                # Final check on the dimension of visual features
                if visual_features.shape[-1] != self.IJEP_DIM:
                    raise ValueError(f"I-JEPA feature extraction resulted in incorrect final dim {visual_features.shape[-1]}. Expected {self.IJEP_DIM}.")

            except Exception as e:
                logger.error(f"Error during I-JEPA feature extraction: {e}. Returning zero predictions.", exc_info=True)
                # Need a way to determine batch size from processed_pixel_values
                batch_size = processed_pixel_values.shape[0] if 'processed_pixel_values' in locals() and processed_pixel_values is not None and processed_pixel_values.ndim >= 1 else 1
                zero_preds = torch.zeros(batch_size, self._trajectory_sampling.num_poses, 3, dtype=torch.float32, device=device)
                return {"trajectory": zero_preds}


        # Predict with MLP
        try:
            # Adjusted to use Ego History Feature dimension
            expected_ego_dim = self.EGO_DIM # Use the agent's calculated EGO_DIM
            if ego_history_features_tensor.shape[-1] != expected_ego_dim:
                 raise ValueError(f"Ego history feature dim mismatch: {ego_history_features_tensor.shape[-1]} vs expected {expected_ego_dim}")
            if ego_history_features_tensor.shape[0] != visual_features.shape[0]:
                 raise ValueError(f"Batch size mismatch between visual features ({visual_features.shape[0]}) and ego features ({ego_history_features_tensor.shape[0]}).")


            combined_features = torch.cat([visual_features, ego_history_features_tensor], dim=1)
            flat_predictions = self.mlp(combined_features)
            # Ensure prediction shape matches expected output (Batch, Time, XYZ)
            predicted_relative_poses = flat_predictions.view(-1, self._trajectory_sampling.num_poses, 3)

            # Final check on prediction shape
            expected_pred_shape = (combined_features.shape[0], self._trajectory_sampling.num_poses, 3)
            if predicted_relative_poses.shape != expected_pred_shape:
                 raise ValueError(f"MLP output shape mismatch: {predicted_relative_poses.shape} vs expected {expected_pred_shape}.")

        except Exception as e:
            logger.error(f"Error during MLP forward: {e}. Returning zero predictions.", exc_info=True)
            # Need a way to determine batch size if MLP forward failed early
            batch_size = combined_features.shape[0] if 'combined_features' in locals() and combined_features.ndim >= 1 else (visual_features.shape[0] if 'visual_features' in locals() and visual_features is not None and visual_features.ndim >= 1 else 1)
            zero_preds = torch.zeros(batch_size, self._trajectory_sampling.num_poses, 3, dtype=torch.float32, device=device)
            return {"trajectory": zero_preds}

        return {"trajectory": predicted_relative_poses}


    def compute_loss(
        self,
        features: Dict[str, torch.Tensor], # Features are available if requires_scene is True, often not used directly in loss
        targets:  Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Calculates the composite loss including base position loss, acceleration penalty, and jerk penalty.
        """
        if self.base_criterion is None:
            raise RuntimeError("Base loss criterion not initialized.")
        if self._interval_length is None or self._interval_length <= 0:
             raise RuntimeError("Trajectory interval length not set or invalid for derivative calculations.")

        device = self.device

        # ---------------------------------------------------------------
        # 1) fetch tensors
        # ---------------------------------------------------------------
        try:
            gt_abs   = targets["trajectory_gt"].to(device)       # (B,T,3) absolute
            pred_rel = predictions["trajectory"].to(device)      # (B,T,3) relative
        except KeyError as e:
            logger.error(f"Missing key for loss computation: {e}. Expected 'trajectory_gt' and 'trajectory'.", exc_info=True)
            # Return a zero loss tensor that requires grad
            return torch.tensor(0.0, device=device, requires_grad=True)

        if pred_rel.shape != gt_abs.shape:
            logger.warning(f"Shape mismatch in compute_loss: pred {pred_rel.shape}, gt {gt_abs.shape}. Returning zero loss.")
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Ensure there are enough time steps for derivative calculations
        num_poses = self._trajectory_sampling.num_poses
        if num_poses < 3 and (self._accel_loss_weight > 0 or self._jerk_loss_weight > 0):
             logger.warning(f"Not enough trajectory points ({num_poses}) for acceleration or jerk loss. Need at least 3 for accel, 4 for jerk. Accel/Jerk loss weights will be ignored.")
             accel_loss_weight = 0.0
             jerk_loss_weight = 0.0
        elif num_poses < 4 and self._jerk_loss_weight > 0:
             logger.warning(f"Not enough trajectory points ({num_poses}) for jerk loss. Need at least 4. Jerk loss weight will be ignored.")
             accel_loss_weight = self._accel_loss_weight # Use original if only jerk is affected
             jerk_loss_weight = 0.0
        else:
             # Use configured weights if sufficient points exist
             accel_loss_weight = self._accel_loss_weight
             jerk_loss_weight = self._jerk_loss_weight


        # ---------------------------------------------------------------
        # 2) add current-frame (x,y) origin to convert rel → abs
        #    Origin comes from GT itself (first pose) - assuming GT is always available
        # ---------------------------------------------------------------
        try:
            # Take the first pose from GT as the origin for the relative prediction
            # Safely handle case where gt_abs might have < 1 pose (already checked above, but extra safe)
            if gt_abs.shape[1] > 0:
                 origin = gt_abs[:, 0:1, :]       # (B,1,3)
                 pred_abs = pred_rel + origin     # broadcasts over time (B, T, 3)
            else:
                 logger.warning("GT trajectory has no poses. Cannot compute absolute predictions or loss. Returning zero loss.")
                 return torch.tensor(0.0, device=device, requires_grad=True)
        except Exception as e:
            logger.error(f"Error computing absolute predictions from relative and origin: {e}", exc_info=True)
            return torch.tensor(0.0, device=device, requires_grad=True)


        # ---------------------------------------------------------------
        # 3) Calculate Base Loss (e.g., L1 or MSE on absolute poses)
        # ---------------------------------------------------------------
        try:
            base_loss = self.base_criterion(pred_abs, gt_abs)
        except Exception as e:
             logger.error(f"Error computing base loss: {e}", exc_info=True)
             return torch.tensor(0.0, device=device, requires_grad=True)


        # ---------------------------------------------------------------
        # 4) Calculate Acceleration and Jerk Penalties
        #    Focus on X and Y dimensions (first 2)
        #    Velocity (B, T-1, 2)
        #    Acceleration (B, T-2, 2)
        #    Jerk (B, T-3, 2)
        # ---------------------------------------------------------------
        accel_penalty = torch.tensor(0.0, device=device)
        jerk_penalty = torch.tensor(0.0, device=device)

        if accel_loss_weight > 0 or jerk_loss_weight > 0:
            try:
                # Calculate velocity from predicted absolute poses (XY only)
                # pred_abs[:, 1:, :2] are points from t=1 to t=T-1
                # pred_abs[:, :-1, :2] are points from t=0 to t=T-2
                # Shape (B, T-1, 2)
                pred_vel = (pred_abs[:, 1:, :2] - pred_abs[:, :-1, :2]) / self._interval_length

                if accel_loss_weight > 0 or jerk_loss_weight > 0:
                     # Calculate acceleration from predicted velocity (XY only)
                     # Shape (B, T-2, 2)
                     if pred_vel.shape[1] > 0: # Need at least 1 velocity vector to calculate accel
                         pred_accel = (pred_vel[:, 1:, :] - pred_vel[:, :-1, :]) / self._interval_length
                         if accel_loss_weight > 0:
                              # L2 norm of acceleration, averaged over batch and time dimensions
                              # torch.norm(pred_accel, dim=-1) -> (B, T-2)
                              # torch.mean(...) -> scalar
                              accel_penalty = torch.mean(torch.norm(pred_accel, dim=-1)) * accel_loss_weight

                         if jerk_loss_weight > 0:
                              # Calculate jerk from predicted acceleration (XY only)
                              # Shape (B, T-3, 2)
                              if pred_accel.shape[1] > 0: # Need at least 1 acceleration vector to calculate jerk
                                  pred_jerk = (pred_accel[:, 1:, :] - pred_accel[:, :-1, :]) / self._interval_length
                                  if pred_jerk.shape[1] > 0: # Need at least 1 jerk vector
                                      # L2 norm of jerk, averaged over batch and time dimensions
                                      # torch.norm(pred_jerk, dim=-1) -> (B, T-3)
                                      # torch.mean(...) -> scalar
                                      jerk_penalty = torch.mean(torch.norm(pred_jerk, dim=-1)) * jerk_loss_weight
                                  else:
                                       logger.warning(f"Insufficient acceleration points ({pred_accel.shape[1]}) to calculate Jerk penalty (need > 1 accel points). Jerk loss weight ignored.")
                              else:
                                   logger.warning(f"Insufficient velocity points ({pred_vel.shape[1]}) to calculate Acceleration/Jerk penalty (need > 1 vel points). Accel/Jerk loss weights ignored.")
                     else:
                          logger.warning(f"Insufficient trajectory points ({num_poses}) to calculate Velocity (need > 1 pose). Skipping accel/jerk loss calculation.")

            except Exception as e:
                 logger.error(f"Error calculating acceleration or jerk penalty: {e}", exc_info=True)
                 # Ensure penalties remain zero on error
                 accel_penalty = torch.tensor(0.0, device=device)
                 jerk_penalty = torch.tensor(0.0, device=device)


        # ---------------------------------------------------------------
        # 5) Combine Loss Terms
        # ---------------------------------------------------------------
        total_loss = base_loss + accel_penalty + jerk_penalty

        # Log individual loss components (optional, for debugging)
        # logger.info(f"Loss components: Base={base_loss.item():.4f}, Accel={accel_penalty.item():.4f}, Jerk={jerk_penalty.item():.4f}, Total={total_loss.item():.4f}")
        self.log('base_loss', base_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        if accel_loss_weight > 0:
             self.log('accel_penalty', accel_penalty, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        if jerk_loss_weight > 0:
             self.log('jerk_penalty', jerk_penalty, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        return total_loss

    # --- PyTorch Lightning Methods ---

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Performs a single training step.
        """
        # Ensure agent components are initialized before the first step
        if not self._ijepa_initialized:
             self.initialize()

        features, targets = batch
        predictions = self.forward(features)
        loss = self.compute_loss(features, targets, predictions) # compute_loss now returns the combined loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Performs a single validation step.
        """
        # Ensure agent components are initialized (should be if training started, but safe)
        if not self._ijepa_initialized:
             self.initialize()

        features, targets = batch
        # Call forward in inference mode for validation
        self.eval() # Set agent to eval mode (disables dropout, batch norm tracking etc.)
        with torch.no_grad():
             predictions = self.forward(features)
             loss = self.compute_loss(features, targets, predictions)
        self.train() # Set agent back to train mode

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True) # Log only at epoch end for validation
        return loss

    def configure_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        """
        Configures the optimizer and optional scheduler for the LightningModule.
        """
        # Ensure MLP is defined before configuring optimizer
        if self.mlp is None:
             raise RuntimeError("MLP head not initialized before configuring optimizer.")

        # Optimizer (only MLP parameters, added weight decay)
        optimizer = torch.optim.AdamW(self.mlp.parameters(),
                                      lr=self._learning_rate,
                                      weight_decay=self._weight_decay)

        # --- Configure Scheduler ---
        if self._use_cosine_scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self._max_epochs, # Total number of training epochs
                eta_min=self._cosine_eta_min
            )
            # Return as a dictionary for Lightning with scheduler
            opt_dict = {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch', # Call scheduler.step() at the end of every epoch
                    'frequency': 1,
                    # 'monitor': 'val_loss' # Optional: monitor a metric for ReduceLROnPlateau etc.
                }
            }
            logger.info(f"Configuring AdamW optimizer (LR={self._learning_rate}, WeightDecay={self._weight_decay}) with CosineAnnealingLR (T_max={self._max_epochs}, eta_min={self._cosine_eta_min}).")
            return opt_dict
        else:
             # Return just the optimizer if no scheduler
             logger.info(f"Configuring AdamW optimizer (LR={self._learning_rate}, WeightDecay={self._weight_decay}) with no scheduler.")
             return optimizer

    def get_training_callbacks(self) -> List[pl.Callback]:
        """
        Returns a list of PyTorch Lightning callbacks to use during training.
        """
        callbacks: List[pl.Callback] = [
             FirstBatchDebugger(),
             # BatchInspector(), # Uncomment if you want to inspect batches periodically
        ]
        # You can add other callbacks here, e.g., ModelCheckpoint, EarlyStopping
        return callbacks