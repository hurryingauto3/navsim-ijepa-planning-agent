import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig, Scene, Camera, EgoStatus, Trajectory
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import convert_absolute_to_relative_se2_array

from PIL import Image

# --- Define Custom Dataset ---
class NavsimTrajectoryDataset(Dataset):
    def __init__(self, scene_loader: SceneLoader, processor, num_history_frames, num_future_frames, device):
        self.scene_loader = scene_loader
        self.tokens = scene_loader.tokens
        self.processor = processor
        self.num_history_frames = num_history_frames
        self.num_future_frames = num_future_frames # For ground truth extraction
        self.device = device
        # Precompute number of driving commands for one-hot encoding
        # Assuming commands are 0, 1, 2 (left, straight, right) or similar integer range
        # Find the max command value if unsure, or hardcode if known (e.g., 3)
        self.num_driving_commands = 4 # Match the data format (e.g., [1,0,0,0])

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        token = self.tokens[idx]
        try:
            scene = self.scene_loader.get_scene_from_token(token)
            current_frame_idx = self.num_history_frames - 1
            current_frame = scene.frames[current_frame_idx]

            # 1. Get Front Camera Image
            if not current_frame.cameras or \
               not hasattr(current_frame.cameras, 'cam_f0') or \
               current_frame.cameras.cam_f0.image is None:
                 raise ValueError(f"Front camera image missing for token {token} at index {idx}")

            image_np = current_frame.cameras.cam_f0.image
            image_pil = Image.fromarray(image_np)
            image_inputs = self.processor(images=image_pil, return_tensors="pt")
            pixel_values = image_inputs['pixel_values'].squeeze(0)

            # 2. Get Ego Status (velocity, acceleration, command)
            ego_status: EgoStatus = current_frame.ego_status
            velocity = torch.tensor(ego_status.ego_velocity, dtype=torch.float32) # Shape [2]
            acceleration = torch.tensor(ego_status.ego_acceleration, dtype=torch.float32) # Shape [2]

            # --- ROBUST COMMAND HANDLING ---
            command_raw = ego_status.driving_command
            command_one_hot = None

            # Check if it looks like a scalar index (int, float, or size-1 array/list)
            if isinstance(command_raw, (int, float)) or \
               (hasattr(command_raw, 'size') and np.size(command_raw) == 1) or \
               (isinstance(command_raw, list) and len(command_raw) == 1):
                try:
                    # Treat as index
                    command_index = torch.tensor(int(command_raw), dtype=torch.long)
                    # Ensure index is valid before applying one_hot
                    if 0 <= command_index < self.num_driving_commands:
                         command_one_hot = nn.functional.one_hot(command_index, num_classes=self.num_driving_commands).float()
                    else:
                        print(f"Warning: Invalid command index {command_index} for token {token}. Using zero vector.")
                        command_one_hot = torch.zeros(self.num_driving_commands, dtype=torch.float32)
                except (ValueError, TypeError) as e:
                    print(f"Warning: Could not process command index {command_raw} for token {token}: {e}. Using zero vector.")
                    command_one_hot = torch.zeros(self.num_driving_commands, dtype=torch.float32)

            # Check if it looks like a pre-encoded vector (list/array of correct length)
            elif hasattr(command_raw, '__len__') and len(command_raw) == self.num_driving_commands:
                 try:
                    # Treat as existing vector
                    command_one_hot = torch.tensor(command_raw, dtype=torch.float32)
                    # Optional validation: Check if it's roughly one-hot
                    if not (torch.isclose(torch.sum(command_one_hot), torch.tensor(1.0)) and \
                            torch.all((command_one_hot >= 0) & (command_one_hot <= 1))):
                        print(f"Warning: Command vector {command_raw} for token {token} not strictly one-hot, using as is.")
                 except (ValueError, TypeError) as e:
                    print(f"Warning: Could not process command vector {command_raw} for token {token}: {e}. Using zero vector.")
                    command_one_hot = torch.zeros(self.num_driving_commands, dtype=torch.float32)
            else:
                 # Fallback for unexpected format
                 print(f"Warning: Unexpected command format {command_raw} (type: {type(command_raw)}) for token {token}. Using zero vector.")
                 command_one_hot = torch.zeros(self.num_driving_commands, dtype=torch.float32)

            # --- END ROBUST COMMAND HANDLING ---

            # Combine ego features
            ego_features = torch.cat([velocity, acceleration, command_one_hot]) # Shape [7]

            # 3. Get Ground Truth Future Trajectory
            gt_trajectory: Trajectory = scene.get_future_trajectory(num_trajectory_frames=self.num_future_frames)
            gt_poses = torch.tensor(gt_trajectory.poses, dtype=torch.float32) # [num_future_frames, 3]

            return pixel_values, ego_features, gt_poses

        except Exception as e:
            # Print error and return None to allow dataloader to skip this item
            # Use traceback to get more detailed error info if needed
            # import traceback
            # print(f"Error loading item {idx} (token: {token}): {e}\n{traceback.format_exc()}")
            print(f"Error loading item {idx} (token: {token}): {e}")
            return None # Remember to use the collate_fn_skip_none

def collate_fn_skip_none(batch):
    # Filter out None values from the batch
    batch = [item for item in batch if item is not None]
    # If the whole batch was None (e.g., batch_size=1 and item failed)
    if not batch:
        return None # Return None to signal skipping this batch
    # Use default collate for the filtered batch
    return torch.utils.data.dataloader.default_collate(batch)
