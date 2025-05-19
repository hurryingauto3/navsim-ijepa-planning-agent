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
import numpy.typing as npt # Added import for npt
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
    SensorConfig, # Corrected indentation
    Scene,
)

# Assuming convert_absolute_to_relative_se2_array is available, e.g., from a utils module
from nuplan.common.actor_state.state_representation import StateSE2
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
    convert_absolute_to_relative_se2_array,
)

from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)

# --- Feature Builders ---

def rel_to_abs(pred_rel, gt_abs):
    origin = gt_abs[:, 0:1, :]             # (B,1,3)
    return pred_rel + origin     

import pytorch_lightning as pl

class FirstBatchDebugger(pl.Callback):
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # Only print on the very first optimisation step
        if trainer.global_step == 0 and batch_idx == 0:
            features, targets = batch

            # ----- ground-truth (absolute) -----
            gt_abs = targets["trajectory_gt"][0].cpu().numpy()        # (T,3)
            print("\n[DEBUG] FIRST-BATCH  GT  (first 3 poses):", gt_abs[:3])

            # ----- model output (relative) -----
            with torch.no_grad():
                pred_rel = pl_module.agent.forward(features)["trajectory"][0]  # (T,3)
                pred_rel = pred_rel.cpu().numpy()
            print("[DEBUG] FIRST-BATCH  PRED (relative, first 3):", pred_rel[:3])

            # ----- convert to absolute for easy comparison -----
            origin = gt_abs[0:1, :]                    # (1,3)  current pose
            pred_abs = pred_rel + origin               # broadcast over T

            print("[DEBUG] FIRST-BATCH  PRED (ABS, first 3):", pred_abs[:3])

class BatchInspector(pl.Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % 100 == 0: # Log every 100 batches
            features, targets = batch
            loss = outputs

            # Example: Print shapes
            print(f"\n[Batch {batch_idx}] Images shape: {features['front_camera_image']}")
            print(f"[Batch {batch_idx}] Ego features shape: {features['ego_features']}")
            print(f"[Batch {batch_idx}] Targets shape: {targets['trajectory_gt']}")
            print(f"[Batch {batch_idx}] Loss shape: {loss}")

            # Example: Print parts of first sample (move to CPU/Numpy if needed)
            print(f"[Batch {batch_idx}] First ego features: {features['ego_features'][0,:].cpu().numpy()}")
            # ... and so on


class CameraImageFeatureBuilder(AbstractFeatureBuilder):
    """
    Feature builder for extracting the front camera image.
    Returns the raw image data as a tensor (CHW, float) to be processed by the agent's forward.
    """
    def get_unique_name(self) -> str:
        return "front_camera_image"

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        # Check if cameras list is non-empty and has cam_f0 data
        if not agent_input.cameras or len(agent_input.cameras) == 0 or \
           not hasattr(agent_input.cameras[-1], "cam_f0") or \
           agent_input.cameras[-1].cam_f0.image is None:
            print(f"Warning: {self.get_unique_name()}: Front camera image missing in last history frame. Returning zero placeholder.")
            dummy_image = np.zeros((224, 224, 3), dtype=np.uint8)
            return {self.get_unique_name(): torch.from_numpy(dummy_image).permute(2, 0, 1).float()}

        # Assuming the relevant camera is the last one in the history list
        image_np = agent_input.cameras[-1].cam_f0.image
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float()
        return {self.get_unique_name(): image_tensor}

class EgoFeatureBuilder(AbstractFeatureBuilder):
    """
    Feature builder for extracting ego status features (velocity, acceleration, driving command).
    Formats the driving command (expected as npt.NDArray[np.int] or int) into a one-hot vector.
    """
    def __init__(self, num_history_frames: int = 4): # Add num_history_frames parameter
        """
        Initializes the EgoFeatureBuilder.
        :param num_history_frames: The number of history frames to include in the feature vector.
        """
        super().__init__()
        self._num_history_frames = num_history_frames
        self.NUM_DRIVING_COMMANDS = 4 # Keep this constant here or make it a class constant if preferred

    def get_unique_name(self) -> str:
        return "ego_features"

    # Ego feature builder for 4 num history frames
    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        # Check if there's enough history
        if not agent_input.ego_statuses or len(agent_input.ego_statuses) < self._num_history_frames:
            print(f"Warning: {self.get_unique_name()}: Insufficient ego history frames ({len(agent_input.ego_statuses)} vs {self._num_history_frames}). Returning zero placeholder.")
            # Calculate the expected output size: (velocity + acceleration + command) * num_history_frames
            expected_output_size = (2 + 2 + self.NUM_DRIVING_COMMANDS) * self._num_history_frames
            return {self.get_unique_name(): torch.zeros(expected_output_size, dtype=torch.float32)}

        # Process the last num_history_frames
        history_ego_features_list = []
        # Iterate backward through the history list, or slice the last N frames
        # Slicing [-self._num_history_frames:] gets the last N frames
        # We iterate forward through this slice to maintain chronological order in the concatenated vector
        for i, ego_status in enumerate(agent_input.ego_statuses[-self._num_history_frames:]):
            velocity = torch.tensor(ego_status.ego_velocity, dtype=torch.float32) # Shape [2]
            acceleration = torch.tensor(ego_status.ego_acceleration, dtype=torch.float32) # Shape [2]
            command_raw = ego_status.driving_command
            command_one_hot = torch.zeros(self.NUM_DRIVING_COMMANDS, dtype=torch.float32) # Shape [NUM_DRIVING_COMMANDS]

            # --- Robust command handling (same logic as before) ---
            try:
                if isinstance(command_raw, np.ndarray):
                    if command_raw.size == 1:
                        command_index = int(command_raw.item())
                        if 0 <= command_index < self.NUM_DRIVING_COMMANDS: command_one_hot[command_index] = 1.0
                    elif command_raw.size == self.NUM_DRIVING_COMMANDS:
                        command_one_hot = torch.from_numpy(command_raw).float()
                    else: print(f"Warning: Builder: Unexpected numpy array size {command_raw.shape} for history frame {i}.")
                elif isinstance(command_raw, (int, float)):
                    command_index = int(command_raw)
                    if 0 <= command_index < self.NUM_DRIVING_COMMANDS: command_one_hot[command_index] = 1.0
                    else: print(f"Warning: Builder: Invalid command index {command_index} for history frame {i}.")
                elif isinstance(command_raw, (list, tuple)) and len(command_raw) == self.NUM_DRIVING_COMMANDS:
                    command_one_hot = torch.tensor(command_raw, dtype=torch.float32)
                else: print(f"Warning: Builder: Unexpected command format {type(command_raw)} for history frame {i}.")
            except Exception as e:
                print(f"Error processing driving command '{command_raw}' for history frame {i}: {e}. Using zero vector.")
                command_one_hot = torch.zeros(self.NUM_DRIVING_COMMANDS, dtype=torch.float32)
            # --- End robust command handling ---

            # Concatenate features for the current historical frame [vel_x, vel_y, acc_x, acc_y, cmd_0, cmd_1, cmd_2, cmd_3]
            current_frame_features = torch.cat([velocity, acceleration, command_one_hot], dim=-1) # Shape [2+2+NUM_DRIVING_COMMANDS = 8]
            history_ego_features_list.append(current_frame_features)

        # Concatenate features ACROSS the history dimension (flattening the history)
        ego_features = torch.cat(history_ego_features_list, dim=-1) # Shape [num_history_frames * (2+2+NUM_DRIVING_COMMANDS)]

        # Final safety check on shape - should match the expected concatenated size
        expected_output_size = (2 + 2 + self.NUM_DRIVING_COMMANDS) * self._num_history_frames
        if ego_features.shape[-1] != expected_output_size:
            print(f"Warning: Builder: Final ego features dim mismatch {ego_features.shape[-1]} vs expected {expected_output_size}. Returning zero placeholder.")
            return {self.get_unique_name(): torch.zeros(expected_output_size, dtype=torch.float32)}

        return {self.get_unique_name(): ego_features}

    '''def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        if not agent_input.ego_statuses or len(agent_input.ego_statuses) == 0:
            print(f"Warning: {self.get_unique_name()}: Ego status missing. Returning zero placeholder.")
            return {self.get_unique_name(): torch.zeros(self.NUM_DRIVING_COMMANDS + 4, dtype=torch.float32)}

        # Use the last ego_status in the history list (current frame)
        ego_status: EgoStatus = agent_input.ego_statuses[-1]

        velocity = torch.tensor(ego_status.ego_velocity, dtype=torch.float32)
        acceleration = torch.tensor(ego_status.ego_acceleration, dtype=torch.float32)
        command_raw = ego_status.driving_command
        command_one_hot = torch.zeros(self.NUM_DRIVING_COMMANDS, dtype=torch.float32)

        try:
            if isinstance(command_raw, np.ndarray):
                if command_raw.size == 1:
                    command_index = int(command_raw.item())
                    if 0 <= command_index < self.NUM_DRIVING_COMMANDS: command_one_hot[command_index] = 1.0
                    else: print(f"Warning: Builder: Invalid command index {command_index} in array.")
                elif command_raw.size == self.NUM_DRIVING_COMMANDS:
                    command_one_hot = torch.from_numpy(command_raw).float()
                else: print(f"Warning: Builder: Unexpected numpy array size {command_raw.shape}.")
            elif isinstance(command_raw, (int, float)):
                command_index = int(command_raw)
                if 0 <= command_index < self.NUM_DRIVING_COMMANDS: command_one_hot[command_index] = 1.0
                else: print(f"Warning: Builder: Invalid command index {command_index}.")
            elif isinstance(command_raw, (list, tuple)) and len(command_raw) == self.NUM_DRIVING_COMMANDS:
                 command_one_hot = torch.tensor(command_raw, dtype=torch.float32)
            else: print(f"Warning: Builder: Unexpected command format {type(command_raw)}.")
        except Exception as e:
            print(f"Error processing driving command '{command_raw}': {e}. Using zero vector.")
            command_one_hot = torch.zeros(self.NUM_DRIVING_COMMANDS, dtype=torch.float32)

        ego_features = torch.cat([velocity, acceleration, command_one_hot], dim=-1)

        if ego_features.shape[-1] != (2 + 2 + self.NUM_DRIVING_COMMANDS):
             print(f"Warning: Builder: Final ego features dim mismatch {ego_features.shape[-1]}.")
             return {self.get_unique_name(): torch.zeros(self.NUM_DRIVING_COMMANDS + 4, dtype=torch.float32)}

        return {self.get_unique_name(): ego_features}'''

# --- Target Builder (Corrected) ---

class TrajectoryTargetBuilderGT(AbstractTargetBuilder):
    """
    Target builder for extracting the ground truth future trajectory poses.
    Converts absolute poses to relative poses using the specified number of history frames.
    """
    # CHANGE: Accept num_history_frames in constructor
    def __init__(self, trajectory_sampling: TrajectorySampling, num_history_frames: int):
        """
        Initializes the target builder.
        :param trajectory_sampling: Trajectory sampling specification (for num_poses).
        :param num_history_frames: Number of history frames to determine the current pose index.
        """
        super().__init__()
        self._trajectory_sampling = trajectory_sampling
        # CHANGE: Store num_history_frames directly
        self._num_history_frames = num_history_frames
        if self._num_history_frames < 1:
            raise ValueError("TrajectoryTargetBuilderGT requires num_history_frames >= 1")

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
            print(f"{self.get_unique_name()}: insufficient future poses "
                f"({0 if future_traj is None else future_traj.poses.shape[0]} "
                f"vs {self._trajectory_sampling.num_poses}). Returning zeros.")
            return {self.get_unique_name():
                    torch.zeros(self._trajectory_sampling.num_poses, 3,
                                dtype=torch.float32)}

        # 3) use absolute poses directly
        gt_abs = torch.as_tensor(future_traj.poses, dtype=torch.float32)

        # 4) final safety-check on shape
        if gt_abs.shape != (self._trajectory_sampling.num_poses, 3):
            print(f"{self.get_unique_name()}: shape mismatch {gt_abs.shape}, "
                "returning zeros.")
            gt_abs = torch.zeros(self._trajectory_sampling.num_poses, 3,
                                dtype=torch.float32)

        return {self.get_unique_name(): gt_abs}


class IJEPAPlanningAgent(AbstractAgent):
    """
    NAVSIM Agent combining I-JEPA encoding and an MLP planning head,
    compatible with the standard NAVSIM training framework.
    """
    # --- Constants ---
    NUM_FUTURE_FRAMES = 8 # Should match trajectory_sampling num_poses
    IJEP_DIM = 1280
    
    HIDDEN_DIM = 256
    NUM_DRIVING_COMMANDS = 4
    # --- End Constants ---

    def __init__(
        self,
        mlp_weights_path: Optional[str] = None,
        ijepa_model_id: str = "facebook/ijepa_vith14_1k",
        trajectory_sampling: TrajectorySampling = TrajectorySampling(
            time_horizon=4, interval_length=0.5
        ),
        # CHANGE: Added num_history_frames parameter
        num_history_frames: int = 4,
        use_cls_token_if_available: bool = True,
        requires_scene: bool = False,
        learning_rate: float = 1e-4,
        loss_criterion: str = "l1",
        max_epochs: int = 50, # Add max_epochs parameter with a default
    ):
        """
        Initializes the IJEPAPlanningAgent.
        :param num_history_frames: Number of history frames used (passed to target builder).
        # ... (rest of docstring)
        """
        super().__init__(
            trajectory_sampling=trajectory_sampling, requires_scene=requires_scene
        )

        # CHANGE: Store num_history_frames
        self._num_history_frames = num_history_frames
        self.EGO_DIM = (2 + 2 + self.NUM_DRIVING_COMMANDS) * self._num_history_frames


        self._mlp_weights_path_config = mlp_weights_path
        self._ijepa_model_id = ijepa_model_id
        self._use_cls_token = use_cls_token_if_available
        self._learning_rate = learning_rate
        self._loss_criterion_type = loss_criterion
        self._max_epochs = max_epochs # Store max_epochs


        self._processor: AutoProcessor = None
        self._ijepa_encoder: AutoModel = None
        self._feature_extraction_method = "Mean Pooling"

        # Define MLP
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.IJEP_DIM + self.EGO_DIM, self.HIDDEN_DIM),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(self.HIDDEN_DIM),
            torch.nn.Linear(self.HIDDEN_DIM, self.HIDDEN_DIM),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(self.HIDDEN_DIM),
            torch.nn.Linear(self.HIDDEN_DIM, self.HIDDEN_DIM),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(self.HIDDEN_DIM),
            torch.nn.Linear(self.HIDDEN_DIM, self.HIDDEN_DIM),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(self.HIDDEN_DIM),
            torch.nn.Linear(self.HIDDEN_DIM, self._trajectory_sampling.num_poses * 3),
        )

        # Define Loss
        if self._loss_criterion_type.lower() == "l1":
            self.criterion = torch.nn.L1Loss()
        elif self._loss_criterion_type.lower() == "mse":
            self.criterion = torch.nn.MSELoss()
        else:
            raise ValueError(f"Unsupported loss: {self._loss_criterion_type}")

    def name(self) -> str:
        return self.__class__.__name__

    def initialize(self) -> None:
        print(f"Initializing {self.name()}...")
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        self.to(device)

        # Load I-JEPA Processor and *Frozen* Encoder
        try:
            self._processor = AutoProcessor.from_pretrained(self._ijepa_model_id, use_fast=True)
            self._ijepa_encoder = AutoModel.from_pretrained(self._ijepa_model_id).to(device)
            for param in self._ijepa_encoder.parameters():
                param.requires_grad = False
            self._ijepa_encoder.eval()
            print(f"Loaded I-JEPA encoder: {self._ijepa_model_id}")
        except Exception as e: # Catch specific exceptions if possible
            raise RuntimeError(f"Failed to load I-JEPA model/processor: {e}") from e

        # Determine Feature Extraction Method (prefer CLS-token from last_hidden_state)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224).to(device)
            outputs = self._ijepa_encoder(pixel_values=dummy)
            hidden = getattr(outputs, "last_hidden_state", None)
            pooler = getattr(outputs, "pooler_output", None)

            if self._use_cls_token:
                # 1) real pooler
                if pooler is not None and pooler.shape[-1] == self.IJEP_DIM:
                    self._feature_extraction_method = "Pooler Output"
                # 2) fallback to first token
                elif hidden is not None and hidden.shape[-1] == self.IJEP_DIM:
                    self._feature_extraction_method = "CLS Token"
                else:
                    print("Warning: I-JEPA CLS token unavailable; defaulting to Mean Pooling")
                    self._feature_extraction_method = "Mean Pooling"
            else:
                # forced mean-pool
                self._feature_extraction_method = "Mean Pooling"

        print(f"Using I-JEPA feature extraction: {self._feature_extraction_method}")

        # ── Optional MLP-weight loading ─────────────────────────────────────────────
        if self._mlp_weights_path_config:
            weights_path = Path(self._mlp_weights_path_config)

            if weights_path.is_file():
                print(f"Loading MLP weights from: {weights_path}")

                try:
                    raw = torch.load(weights_path, map_location=device)
                    state = raw["state_dict"] if "state_dict" in raw else raw     # .ckpt or .pth

                    mlp_state = {
                        k.replace("agent.mlp.", ""): v
                        for k, v in state.items()
                        if k.startswith("agent.mlp.")
                    }

                    if not mlp_state:
                        raise ValueError("No 'agent.mlp.' keys found in checkpoint")

                    self.mlp.load_state_dict(mlp_state, strict=False)
                    print(f"MLP weights loaded: {len(mlp_state)} tensors.")
                except Exception as e:
                    print(f"Warning: failed to load MLP weights ({e}). Starting fresh.")
            else:
                print(f"Warning: MLP weights file not found: {weights_path}. Starting fresh.")
        else:
            print("No MLP weights path provided. Starting training from scratch.")

        self.mlp.to(device)
        print(f"{self.name()} initialization complete.")

    def get_sensor_config(self) -> SensorConfig:
        # Only front camera needed by builders
        return SensorConfig(cam_f0=True, cam_l0=False, cam_l1=False, cam_l2=False, cam_r0=False, cam_r1=False, cam_r2=False, cam_b0=False, lidar_pc=False)

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        return [CameraImageFeatureBuilder(), EgoFeatureBuilder(num_history_frames=self._num_history_frames)]

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        return [TrajectoryTargetBuilderGT(trajectory_sampling=self._trajectory_sampling,
                                         num_history_frames=self._num_history_frames)]

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self._ijepa_encoder is None or self._processor is None or self.mlp is None:
            raise RuntimeError("Agent not initialized.")

        device = next(self.parameters()).device
        self.mlp.to(device)
        self._ijepa_encoder.to(device)

        # Get features from builders' output
        try:
            image_tensor_raw = features["front_camera_image"].to(device)
            ego_features_tensor = features["ego_features"].to(device)
        except KeyError as e: raise KeyError(f"Missing feature key: {e}") from e

        # Preprocess image using Hugging Face processor
        processed_pixel_values = None
        try:
            image_batch_np = image_tensor_raw.mul(255).permute(0, 2, 3, 1).byte().cpu().numpy()
            image_list_pil = [Image.fromarray(img) for img in image_batch_np]
            processor_output = self._processor(images=image_list_pil, return_tensors="pt")
            processed_pixel_values = processor_output['pixel_values'].to(device)
            if processed_pixel_values is None: raise ValueError("Processor returned None")
        except Exception as e:
            print(f"Error during image processing: {e}. Returning zero predictions.")
            batch_size = image_tensor_raw.shape[0] if image_tensor_raw.ndim == 4 else 1
            zero_preds = torch.zeros(batch_size, self._trajectory_sampling.num_poses, 3, dtype=torch.float32, device=device)
            return {"trajectory": zero_preds}

        # Extract features with Frozen I-JEPA
        visual_features = None
        with torch.no_grad():
            ijepa_outputs = self._ijepa_encoder(pixel_values=processed_pixel_values)
            if self._feature_extraction_method == "Pooler Output":
                visual_features = ijepa_outputs.pooler_output
            elif self._feature_extraction_method == "CLS Token":
                # take the [CLS] embedding
                visual_features = ijepa_outputs.last_hidden_state[:, 0, :]
            else:  # Mean Pooling
                visual_features = ijepa_outputs.last_hidden_state.mean(dim=1)
            if visual_features is None or visual_features.shape[-1] != self.IJEP_DIM:
                raise ValueError("I-JEPA feature extraction failed.")

        # Predict with MLP
        try:
            if ego_features_tensor.shape[-1] != self.EGO_DIM: raise ValueError(f"Ego feature dim mismatch: {ego_features_tensor.shape[-1]} vs {self.EGO_DIM}")
            combined_features = torch.cat([visual_features, ego_features_tensor], dim=1)
            flat_predictions = self.mlp(combined_features)
            predicted_relative_poses = flat_predictions.view(-1, self._trajectory_sampling.num_poses, 3)
        except Exception as e:
            print(f"Error during MLP forward: {e}. Returning zero predictions.")
            batch_size = visual_features.shape[0] if visual_features is not None else 1
            zero_preds = torch.zeros(batch_size, self._trajectory_sampling.num_poses, 3, dtype=torch.float32, device=device)
            return {"trajectory": zero_preds}

        return {"trajectory": predicted_relative_poses}


    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets:  Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if self.criterion is None:
            raise RuntimeError("Loss criterion not initialized.")

        device = next(self.parameters()).device       # <- reliable device

        # ---------------------------------------------------------------
        # 1) fetch tensors
        # ---------------------------------------------------------------
        try:
            gt_abs   = targets["trajectory_gt"].to(device)       # (B,T,3) absolute
            pred_rel = predictions["trajectory"].to(device)      # (B,T,3) relative
        except KeyError as e:
            raise KeyError(f"Missing key for loss computation: {e}") from e

        if pred_rel.shape != gt_abs.shape:
            print(f"Shape mismatch: pred {pred_rel.shape}, gt {gt_abs.shape}")
            return torch.tensor(0.0, device=device, requires_grad=True)

        # ---------------------------------------------------------------
        # 2) add current-frame (x,y) origin to convert rel → abs
        #    origin comes from ego_features: [vx, vy, ax, ay, cmd0..3]
        #    we stored *velocities* there, not positions, so instead we
        #    derive origin from GT itself (first pose) — safest & frame-correct
        # ---------------------------------------------------------------
        origin = gt_abs[:, 0:1, :]       # (B,1,3)
        pred_abs = pred_rel + origin     # broadcasts over time

        # ---------------------------------------------------------------
        # 3) loss between absolute predictions and absolute GT
        # ---------------------------------------------------------------
        loss = self.criterion(pred_abs, gt_abs)
        return loss

    def get_optimizers(self) -> Dict[str, Union[Optimizer, LRScheduler]]:
        if self.mlp is None: raise RuntimeError("MLP head not initialized.")

        # Optimizer (only MLP parameters)
        optimizer = torch.optim.AdamW(self.mlp.parameters(), lr=self._learning_rate)

        # --- ADD Scheduler ---
        # Use the stored max_epochs for T_max
        # Cosine Annealing scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self._max_epochs, # Use the value passed via config
            eta_min=self._learning_rate * 0.01 # Decay to 1% of initial LR (common)
            #eta_min=1e-6 # Alternative fixed minimum
        )
        # --- END ADD Scheduler ---

        # Return as a dictionary for Lightning
        opt_dict = {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch', # Call scheduler.step() at the end of every epoch
                'frequency': 1,
            }
        }

        # Add a print statement here for verification if needed
        print(f"INFO: Using the following optimizer setup {opt_dict}")

        return opt_dict

    def get_training_callbacks(self) -> List[pl.Callback]:
        return []
    



# def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
    #     future_trajectory: Trajectory = scene.get_future_trajectory(
    #         num_trajectory_frames=self._trajectory_sampling.num_poses
    #     )

    #     if future_trajectory is None or future_trajectory.poses is None or future_trajectory.poses.shape[0] < self._trajectory_sampling.num_poses:
    #         print(f"{self.get_unique_name()}: Insufficient future poses found ({future_trajectory.poses.shape[0] if future_trajectory.poses is not None else 0} vs {self._trajectory_sampling.num_poses}). Returning zero placeholder.")
    #         return {self.get_unique_name(): torch.zeros(self._trajectory_sampling.num_poses, 3, dtype=torch.float32)}

    #     gt_poses_abs = future_trajectory.poses

    #     current_frame_idx = self._num_history_frames - 1
    #     current_ego_pose_for_conversion: Optional[StateSE2] = None # Use Optional and type hint
    #     gt_poses_rel = gt_poses_abs # Default fallback

    #     if not scene.frames or len(scene.frames) <= current_frame_idx: # Use <= for correct index check
    #         print(f"{self.get_unique_name()}: Scene has insufficient frames ({len(scene.frames)}) to get current pose at index {current_frame_idx}. Cannot convert to relative. Returning absolute poses.")
    #     elif scene.frames[current_frame_idx].ego_status is None:
    #         print(f"{self.get_unique_name()}: Ego status missing for current frame {current_frame_idx}. Cannot convert to relative. Returning absolute poses.")
    #     elif scene.frames[current_frame_idx].ego_status.ego_pose is None:
    #          print(f"{self.get_unique_name()}: Ego pose data missing for current frame {current_frame_idx}. Cannot convert to relative. Returning absolute poses.")
    #     else:
    #         raw_ego_pose_data = scene.frames[current_frame_idx].ego_status.ego_pose

    #         # --- Robust conversion: take first three elements if it's an array ---
    #         if isinstance(raw_ego_pose_data, np.ndarray) and raw_ego_pose_data.size >= 3:
    #             x, y, heading = float(raw_ego_pose_data.flat[0]), float(raw_ego_pose_data.flat[1]), float(raw_ego_pose_data.flat[2])
    #             current_ego_pose_for_conversion = StateSE2(x=x, y=y, heading=heading)
    #         elif isinstance(raw_ego_pose_data, StateSE2):
    #             current_ego_pose_for_conversion = raw_ego_pose_data
    #         else:
    #             print(f"{self.get_unique_name()}: Unexpected ego pose format: {type(raw_ego_pose_data)}. Cannot convert to relative. Returning absolute poses.")                

    #         # --- MODIFICATION ENDS HERE ---

    #         # Convert absolute future poses to relative poses if we have a valid current pose object
    #         if current_ego_pose_for_conversion is not None:
    #              try:
    #                  # Now call the conversion function with the correctly formatted StateSE2 object
    #                  gt_poses_rel = convert_absolute_to_relative_se2_array(current_ego_pose_for_conversion, gt_poses_abs)
    #              except Exception as e: # Catch generic exception during conversion
    #                  print(f"{self.get_unique_name()}: Failed to convert future poses to relative using StateSE2 object: {e}. Returning absolute poses.")
    #                  gt_poses_rel = gt_poses_abs # Fallback in case conversion fails even with correct object type

    #     # Convert to tensor and ensure correct shape
    #     gt_poses_tensor = torch.tensor(gt_poses_rel, dtype=torch.float32)

    #     # Final shape check (optional, but good practice)
    #     # The utility function convert_absolute_to_relative_se2_array returns an array
    #     # of shape (N, 3) if the input state_se2_array is (N, 3). This shape check
    #     # confirms the number of poses matches what was requested.
    #     if gt_poses_tensor.shape != (self._trajectory_sampling.num_poses, 3):
    #          print(f"{self.get_unique_name()}: Final poses tensor shape mismatch after processing ({gt_poses_tensor.shape} vs {(self._trajectory_sampling.num_poses, 3)}). Returning zero placeholder.")
    #          return {self.get_unique_name(): torch.zeros(self._trajectory_sampling.num_poses, 3, dtype=torch.float32)}


    #     return {self.get_unique_name(): gt_poses_tensor}