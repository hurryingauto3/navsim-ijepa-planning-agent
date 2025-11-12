Okay, let's break down the provided information about agents in the `navsim` library.

**Core Concept: The Agent**

In `navsim`, an "Agent" is a module responsible for predicting the future trajectory of the ego vehicle based on various inputs. Think of it as the "brain" making driving decisions.

**The Foundation: `AbstractAgent`**

Every agent you create *must* inherit from `navsim.agents.abstract_agent.AbstractAgent`. This base class defines the essential contract that all agents need to fulfill. It's built upon `torch.nn.Module`, meaning agents are standard PyTorch modules.

**Key Methods for *All* Agents:**

1.  `__init__(self, trajectory_sampling: TrajectorySampling, requires_scene: bool = False)`:
    *   The constructor.
    *   Crucially takes `trajectory_sampling`, which defines the time horizon and frequency (e.g., 4 seconds at 10Hz) of the trajectory the agent should output. This ensures consistency for evaluation.
    *   `requires_scene`: A flag indicating if the agent needs access to the ground-truth `Scene` object during inference (like the `HumanAgent`). Most agents will keep this `False`.

2.  `name(self) -> str`:
    *   Returns a unique string name for the agent.
    *   Used for identification and naming output files (like evaluation results).

3.  `initialize(self) -> None`:
    *   Called *before* the first inference call for the agent, potentially within each parallel worker if used.
    *   **Important:** Use this method for loading model weights (`state_dict`) or performing computationally intensive setup, *not* the `__init__` constructor. This ensures proper initialization, especially in multi-process environments.

4.  `get_sensor_config(self) -> SensorConfig`:
    *   Defines *which* sensor data the agent needs and *when* (how much history).
    *   Returns a `SensorConfig` dataclass. You can specify lists of history frame indices for each sensor (e.g., `cam_f0=[0, 1, 2, 3]` for 4 history frames) or use `True` to load all available history, or `False` to load none.
    *   `SensorConfig.build_all_sensors()` and `SensorConfig.build_no_sensors()` are convenient helpers.
    *   **Performance Tip:** Only request the sensors you *actually* use. Loading unnecessary sensor data significantly impacts runtime.

5.  `compute_trajectory(self, agent_input: AgentInput) -> Trajectory`:
    *   The core inference function for *non-learning-based* agents (or agents where you want explicit control).
    *   Takes `AgentInput` (containing ego status history and the sensor data requested via `get_sensor_config`).
    *   Must return a `Trajectory` object, which holds an array of future poses (x, y, heading in local coordinates) and the `TrajectorySampling` config.

**Learning-Based Agents: Additional Requirements**

If your agent uses machine learning and you want to leverage `navsim`'s training infrastructure, you need to implement these *additional* methods (and typically *don't* override `compute_trajectory` directly):

1.  `get_feature_builders(self) -> List[AbstractFeatureBuilder]`:
    *   Returns a list of `AbstractFeatureBuilder` objects.
    *   Feature Builders take the `AgentInput` (sensor data, ego status) and transform it into feature tensors (e.g., BEV maps, processed images, feature vectors) suitable for input to the neural network.
    *   Examples: `EgoStatusFeatureBuilder`, `TransfuserFeatureBuilder`.

2.  `get_target_builders(self) -> List[AbstractTargetBuilder]`:
    *   Returns a list of `AbstractTargetBuilder` objects.
    *   Target Builders take the full `Scene` object (which includes ground-truth information like future trajectory, agent annotations, map data) and generate the target tensors needed for calculating the loss during training.
    *   Examples: `TrajectoryTargetBuilder`, `TransfuserTargetBuilder`.

3.  `forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]`:
    *   The standard PyTorch `forward` pass.
    *   Receives a dictionary of batched feature tensors (output from the feature builders).
    *   Must return a dictionary of prediction tensors. Critically, this dictionary *must* contain a key `"trajectory"` with the predicted trajectory tensor of shape `[B, T, 3]` (Batch, Time, [x, y, heading]).

4.  `compute_loss(self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor]) -> torch.Tensor`:
    *   Calculates the loss value used for training.
    *   Takes the input features, ground-truth targets (from target builders), and the model's predictions (from `forward`).
    *   Must return a single scalar `torch.Tensor` representing the loss.

5.  `get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]`:
    *   Defines the optimizer(s) and optional learning rate scheduler(s) for training.
    *   Can return just an optimizer or a dictionary containing `"optimizer"` and `"lr_scheduler"`.

6.  `get_training_callbacks(self) -> List[pl.Callback]`:
    *   Returns a list of PyTorch Lightning Callbacks (`pl.Callback`).
    *   Used for monitoring, logging, visualization, or other actions during the training loop (e.g., `TransfuserCallback` for visualizing predictions).

**How `compute_trajectory` works for Learning Agents:**

The `AbstractAgent` provides a default implementation of `compute_trajectory` that is used if you don't override it. For learning agents, this default implementation automatically:
1.  Calls the agent's `get_feature_builders()` to get the feature builders.
2.  Runs each feature builder on the `AgentInput` to generate feature tensors.
3.  Batches these features (adds a batch dimension of 1).
4.  Calls the agent's `forward()` method with the features in `torch.no_grad()` mode.
5.  Extracts the `"trajectory"` tensor from the `forward()` output.
6.  Packages the trajectory tensor into a `Trajectory` object using the agent's `_trajectory_sampling`.

**Inputs Provided to the Agent:**

*   **Sensor Data:** Defined by `get_sensor_config`. For OpenScene, this includes 8 cameras and 1 merged LiDAR point cloud, each with up to 2 seconds of history at 2Hz (4 frames total). *Crucially, only this sensor data (and ego status/command) will be available during leaderboard evaluation.* Maps, GT tracks etc., might be used for training via `Scene` but not for test-time inference.
*   **Ego Status:** Past and current ego pose, velocity, and acceleration in the local coordinate frame.
*   **Driving Command:** A discrete indicator (left, straight, right) based *only* on the planned route, not obstacles. Helps disambiguate intent.

**Output Required from the Agent:**

*   A `Trajectory` object containing:
    *   An array of future ego poses (x, y, heading) in local coordinates.
    *   The `TrajectorySampling` config specifying duration and frequency.
*   Evaluation (PDM score) is typically done over a 4-second horizon at 10Hz. The `TrajectorySampling` allows interpolation if the agent outputs at a different frequency.

**Provided Baselines (Examples):**

1.  **`ConstantVelocityAgent`:**
    *   *Type:* Non-learning based.
    *   *Logic:* Assumes constant speed and heading, predicting a straight line.
    *   *Sensors:* None (`SensorConfig.build_no_sensors()`).
    *   *Use:* Simplest baseline, good for understanding the basic `AbstractAgent` interface.

2.  **`EgoStatusMLPAgent`:**
    *   *Type:* Learning-based (MLP).
    *   *Logic:* Uses only ego status (velocity, acceleration, driving command) as input to an MLP to predict the trajectory. It's "blind" to the environment.
    *   *Sensors:* None.
    *   *Use:* Simple learning example, shows feature/target builders, training setup. Represents performance achievable by just extrapolating ego motion.

3.  **`HumanAgent`:**
    *   *Type:* Non-learning based (Privileged).
    *   *Logic:* Directly outputs the ground-truth future trajectory recorded by the human driver.
    *   *Sensors:* None.
    *   *Special:* Requires `requires_scene = True` and accesses the `Scene` object in `compute_trajectory`.
    *   *Use:* Provides an upper bound or oracle performance.

4.  **`TransfuserAgent`:**
    *   *Type:* Learning-based (CNNs + Transformers).
    *   *Logic:* Fuses features from front cameras and LiDAR BEV using a Transformer-based backbone (`TransfuserBackbone`). Predicts trajectory and performs auxiliary tasks (BEV semantic segmentation, agent detection).
    *   *Sensors:* Front cameras (`cam_f0`, `cam_l0`, `cam_r0`) and LiDAR (`lidar_pc`) from the *current* timestep.
    *   *Use:* Complex, state-of-the-art sensor-based agent example. Demonstrates advanced feature/target building, custom losses (Hungarian matching), model architecture, and visualization callbacks. Uses a detailed `TransfuserConfig`.

**In Summary:**

The `navsim` library provides a flexible framework for defining driving agents. The `AbstractAgent` class mandates a core interface, while learning-based agents extend this with methods for feature extraction (`get_feature_builders`), target generation (`get_target_builders`), model definition (`forward`), loss calculation (`compute_loss`), optimization (`get_optimizers`), and training callbacks (`get_training_callbacks`). The provided baselines showcase different approaches, from simple heuristics to complex deep learning models, offering starting points for development. Careful sensor selection via `get_sensor_config` is crucial for performance.