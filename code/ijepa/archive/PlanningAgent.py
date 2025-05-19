# ijepa_planner.py

# --- Imports ---
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoModel
from typing import Dict, Tuple, List # Added List

# NAVSIM imports
# --- CHANGE Base class import ---
from navsim.agents.abstract_agent import AbstractAgent
# from navsim.planning.simulation.planner.abstract_planner import AbstractPlanner, PlannerInitialization, PlannerInput # Remove or comment out
from navsim.common.dataclasses import AgentInput, Trajectory, EgoStatus, TrajectorySampling, SensorConfig # Added SensorConfig
# --- Need these for placeholder methods ---
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
from navsim.agents.transfuser.transfuser_callback import TransfuserCallback
from navsim.agents.ijepa.PlanningHead import PlanningHead

# --- Constants (MUST MATCH TRAINING CONFIGURATION) ---
NUM_FUTURE_FRAMES = 8   # Number of poses in the output trajectory (4 seconds at 0.5s interval)
IJEP_DIM = 1280         # Dimension of I-JEPA features (1280 for ViT-H/14)
EGO_DIM = 8             # Dimension of Ego features (vel_x, vel_y, acc_x, acc_y, cmd_0, cmd_1, cmd_2, cmd_3)
HIDDEN_DIM = 256        # Hidden dimension used in the MLP head during training
NUM_DRIVING_COMMANDS = 4 # Number of driving commands expected (must match data)
# --- End Constants ---

class IJEPAPlanningAgent(AbstractAgent):
    """
    NAVSIM Agent using a frozen I-JEPA encoder and a pre-trained MLP head.
    Inherits from AbstractAgent for compatibility with evaluation script structure.
    """
    def __init__(
        self,
        mlp_weights_path: str,
        ijepa_model_id: str,
        trajectory_sampling: TrajectorySampling, # AbstractAgent expects this
        use_cls_token_if_available: bool = True,
    ):
        # --- Pass trajectory_sampling to super().__init__ ---
        super().__init__(trajectory_sampling=trajectory_sampling)
        self._ijepa_model_id = ijepa_model_id
        self._mlp_weights_path_config = mlp_weights_path
        self._use_cls_token = use_cls_token_if_available
        self._num_driving_commands = NUM_DRIVING_COMMANDS
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"DEBUG: Forcing device to {self._device}")
        self._processor: AutoProcessor = None
        self._ijepa_encoder: AutoModel = None
        self._mlp_head: PlanningHead = None
        self._feature_extraction_method = "Mean Pooling"

    # --- Implement initialize (no PlannerInitialization needed) ---
    def initialize(self): # Changed signature
        """ Loads models and weights """
        print(f"Initializing {self.name()}...")
        print(f"Using device: {self._device}")
        # --- Resolve Weights Path ---
        weights_path = Path(self._mlp_weights_path_config)
        # (Keep your existing path resolution logic here)
        if not weights_path.is_file():
             raise FileNotFoundError(f"MLP weights file not found")

        print(f"Loading MLP weights from resolved path: {weights_path}")
        # --- Load Models and Processor ---
        self._processor = AutoProcessor.from_pretrained(self._ijepa_model_id, use_fast=True)
        self._ijepa_encoder = AutoModel.from_pretrained(self._ijepa_model_id).to(self._device)
        for param in self._ijepa_encoder.parameters():
            param.requires_grad = False
        self._ijepa_encoder.eval()
        # --- Determine feature extraction (keep your logic) ---
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224).to(self._device)
            try:
                 outputs = self._ijepa_encoder(pixel_values=dummy_input)
                 if self._use_cls_token and hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                     self._feature_extraction_method = "Pooler Output (CLS Token)"
                 elif hasattr(outputs, "last_hidden_state"):
                     self._feature_extraction_method = "Mean Pooling"
                 else:
                     self._feature_extraction_method = "Unknown"
                 print(f"Using I-JEPA feature extraction method: {self._feature_extraction_method}")
            except Exception as e:
                 self._feature_extraction_method = "Mean Pooling"
                 print(f"Warning: Could not check I-JEPA output: {e}. Defaulting to {self._feature_extraction_method}.")
        # --- Load MLP Head (keep your logic) ---
        self._mlp_head = PlanningHead(IJEP_DIM, EGO_DIM, HIDDEN_DIM, output_dim=NUM_FUTURE_FRAMES).to(self._device)
        self._mlp_head.load_state_dict(torch.load(weights_path, map_location=self._device))
        self._mlp_head.eval()
        print(f"{self.name()} initialization complete.")

    def name(self) -> str:
        return "IJEPAPlanningAgent"

    # --- Implement get_sensor_config (as added before) ---
    def get_sensor_config(self) -> SensorConfig:
        """ Returns the sensor configuration required by this agent. """
        return SensorConfig(cam_f0=True, cam_l0=False, cam_l1=False, cam_l2=False,
                            cam_r0=False, cam_r1=False, cam_r2=False, cam_b0=False,
                            lidar_pc=False)

    # --- RENAME compute_planner_trajectory to compute_trajectory ---
    # --- Ensure input type matches AbstractAgent (AgentInput likely correct) ---
    @torch.no_grad()
    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        """ Computes the predicted trajectory for the current timestep. """
        # --- The logic inside this method remains EXACTLY THE SAME ---
        # --- as your previous compute_planner_trajectory method ---
        history: AgentInput = agent_input # Use the direct input

        if not history.ego_statuses: raise ValueError("Ego status history empty")
        if not history.cameras: raise ValueError("Camera history empty")
        current_ego_status = history.ego_statuses[-1]
        current_cameras = history.cameras[-1]

        # Prepare Image Input (same as before)
        if not hasattr(current_cameras, 'cam_f0') or current_cameras.cam_f0.image is None:
            print("Warning: CAM_F0 image missing...")
            zero_poses = np.zeros((NUM_FUTURE_FRAMES, 3), dtype=np.float32)
            return Trajectory(poses=zero_poses, trajectory_sampling=self._trajectory_sampling)
        image_np = current_cameras.cam_f0.image
        image_pil = Image.fromarray(image_np)
        image_inputs = self._processor(images=image_pil, return_tensors="pt")
        pixel_values = image_inputs['pixel_values'].to(self._device)

        # Prepare Ego Status Input (same as before)
        velocity = torch.tensor(current_ego_status.ego_velocity, dtype=torch.float32)
        acceleration = torch.tensor(current_ego_status.ego_acceleration, dtype=torch.float32)
        command_raw = current_ego_status.driving_command
        command_one_hot = None
        # (Keep the robust command handling logic here)
        if isinstance(command_raw, (int, float)) or \
           (hasattr(command_raw, 'size') and np.size(command_raw) == 1) or \
           (isinstance(command_raw, list) and len(command_raw) == 1):
             try:
                 command_index = torch.tensor(int(command_raw), dtype=torch.long)
                 if 0 <= command_index < self._num_driving_commands: command_one_hot = nn.functional.one_hot(command_index, num_classes=self._num_driving_commands).float()
                 else: command_one_hot = torch.zeros(self._num_driving_commands, dtype=torch.float32)
             except: command_one_hot = torch.zeros(self._num_driving_commands, dtype=torch.float32)
        elif hasattr(command_raw, '__len__') and len(command_raw) == self._num_driving_commands:
             try: command_one_hot = torch.tensor(command_raw, dtype=torch.float32)
             except: command_one_hot = torch.zeros(self._num_driving_commands, dtype=torch.float32)
        else: command_one_hot = torch.zeros(self._num_driving_commands, dtype=torch.float32)
        ego_features = torch.cat([velocity, acceleration, command_one_hot]).unsqueeze(0).to(self._device)

        # Model Inference (same as before)
        ijepa_outputs = self._ijepa_encoder(pixel_values=pixel_values)
        if self._feature_extraction_method == "Pooler Output (CLS Token)": visual_features = ijepa_outputs.pooler_output
        elif self._feature_extraction_method == "Mean Pooling": visual_features = ijepa_outputs.last_hidden_state.mean(dim=1)
        else: visual_features = ijepa_outputs.last_hidden_state.mean(dim=1) # Fallback
        predicted_relative_poses_tensor = self._mlp_head(visual_features, ego_features)

        # Format Output (same as before)
        predicted_relative_poses = predicted_relative_poses_tensor.squeeze(0).cpu().numpy()
        trajectory = Trajectory(poses=predicted_relative_poses, trajectory_sampling=self._trajectory_sampling)
        return trajectory

    # --- ADD Placeholders for other AbstractAgent methods ---
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """ Forward pass used potentially during training (not needed for PDM eval) """
        # This agent is pre-trained; evaluation uses compute_trajectory directly.
        raise NotImplementedError("Forward method not implemented for inference-only agent.")

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        """ Feature builders for training (not needed for PDM eval) """
        return [] # Return empty list as we are not using the training framework here

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        """ Target builders for training (not needed for PDM eval) """
        return [] # Return empty list