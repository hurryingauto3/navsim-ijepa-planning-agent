Okay, let's break down the `navsim.planning` directory based on the provided code snippets.

This directory seems to handle both the **training** of planning models (agents) and the **simulation/evaluation** of planners, with a significant focus on a specific planner type called "PDM" (likely Probabilistic Driving Model or similar). It also includes scripts for running these processes and utilities for handling data and scenarios.

Here's a breakdown by submodule:

**1. `navsim.planning.training`**: Focuses on setting up and executing the training process for planning agents.

*   **`__init__.py`**: Likely empty, just marks the directory as a Python package.
*   **`abstract_feature_target_builder.py`**:
    *   `AbstractFeatureBuilder`: Defines the interface for classes that compute input *features* for a planning model. These features must be computable *without* ground truth future information (using `AgentInput`).
    *   `AbstractTargetBuilder`: Defines the interface for classes that compute *targets* (ground truth labels) for training. These can access the full ground truth `Scene`.
    *   **Purpose**: These abstract classes enforce a standard way to define what data the planning model receives as input and what it's trying to predict during training.
*   **`dataset.py`**:
    *   `load_feature_target_from_pickle`, `dump_feature_target_to_pickle`: Helper functions to load/save computed features/targets using compressed pickle files (`.gz`).
    *   `CacheOnlyDataset`: A PyTorch `Dataset` that *only* loads pre-computed features/targets from a specified cache directory. It checks if all required feature/target files (identified by builder names) exist for a given sample (token).
    *   `Dataset`: A more flexible PyTorch `Dataset`. It can:
        *   Compute features/targets on-the-fly if no `cache_path` is provided.
        *   Use cached data if available.
        *   Compute and *save* features/targets to the cache if `cache_path` is provided (either forced or if the cache doesn't exist). It uses the `AbstractFeatureBuilder` and `AbstractTargetBuilder` instances passed to it.
    *   **Purpose**: Provides the mechanism to feed data (features, targets) to the training loop, with efficient caching capabilities.
*   **`agent_lightning_module.py`**:
    *   `AgentLightningModule`: A wrapper class using `pytorch_lightning` to train an `AbstractAgent`. It takes an agent instance, handles the forward pass (`agent.forward`), loss calculation (`agent.compute_loss`), logging, and optimizer configuration (`agent.get_optimizers`).
    *   **Purpose**: Integrates a NAVSIM agent into the standard PyTorch Lightning training framework, simplifying the training loop implementation.
*   **`callbacks/time_logging_callback.py`**:
    *   `TimeLoggingCallback`: A simple PyTorch Lightning callback to log the time taken for each training, validation, and test epoch.
    *   **Purpose**: Provides basic timing information during the training process.

**2. `navsim.planning.simulation`**: Contains code related to running planning simulations, particularly the PDM planner family.

*   **`__init__.py`**: Likely empty package marker.
*   **`planner/__init__.py`**: Likely empty package marker for planners.
*   **`planner/pdm_planner`**: This is a substantial submodule implementing the PDM planner.
    *   **`__init__.py`**: Package marker.
    *   **`abstract_pdm_planner.py` (`AbstractPDMPlanner`)**:
        *   A base class for all PDM planner variants.
        *   Handles common functionalities like:
            *   Loading and correcting the route (roadblock IDs) based on the ego's current position.
            *   Finding the current lane the ego is on.
            *   Extracting a discrete centerline path from the map using Dijkstra's algorithm on the lane graph (`_get_discrete_centerline`).
            *   Requires a `PDMDrivableMap`.
    *   **`abstract_pdm_closed_planner.py` (`AbstractPDMClosedPlanner`)**:
        *   Inherits from `AbstractPDMPlanner`.
        *   Specific interface for *closed-loop* PDM planners (like PDM-Closed, PDM-Hybrid).
        *   Introduces the core PDM loop components:
            *   `PDMObservation`: Handles environment perception/prediction.
            *   `PDMGenerator`: Generates candidate trajectories (proposals).
            *   `PDMSimulator`: Simulates the proposals forward in time.
            *   `PDMScorer`: Evaluates the simulated proposals.
            *   `PDMProposalManager`: Manages the different proposals (combinations of paths and policies).
            *   `BatchIDMPolicy`: Used for longitudinal control within proposals.
        *   Defines the `_get_closed_loop_trajectory` method outlining the planning cycle: update observation -> update proposals -> generate -> simulate -> score -> select best.
    *   **`pdm_closed_planner.py` (`PDMClosedPlanner`)**:
        *   Concrete implementation inheriting from `AbstractPDMClosedPlanner`.
        *   Implements the `initialize` and `compute_planner_trajectory` methods required by the `nuplan` planner interface.
        *   Uses `PDMDrivableMap` for checking drivable areas during planning.
    *   **`observation/`**: Handles how the PDM planner perceives the environment.
        *   **`pdm_observation.py` (`PDMObservation`)**: Creates time-series of forecasted occupancy maps (`PDMOccupancyMap`). It processes tracked objects (from `DetectionsTracks`) and traffic light data, potentially predicting future positions (implicitly via `PDMObjectManager` or explicitly via log replay/traffic policies). It manages unique objects and handles temporal sampling/resolution differences. Can be updated from live simulation data or replay data.
        *   **`pdm_occupancy_map.py` (`PDMOccupancyMap`, `PDMDrivableMap`)**: Implements spatial indexing (using `STRtree`) for efficient querying of objects/map elements based on polygons. `PDMDrivableMap` is specialized for map elements, storing their semantic types and providing methods like `points_in_polygons` and `is_in_layer`. `from_simulation` builds this map from the `AbstractMap` around the ego.
        *   **(Inferred `pdm_object_manager.py` `PDMObjectManager`)**: Likely filters and manages tracked objects based on proximity and collision status. (Not provided, but referenced implicitly).
    *   **`proposal/`**: Deals with generating candidate trajectories.
        *   **`pdm_generator.py` (`PDMGenerator`)**: Generates multiple trajectory proposals by unrolling longitudinal policies (`BatchIDMPolicy`) along different lateral paths (`PDMPath`). It uses the `PDMObservation` to find leading agents/obstacles for the IDM calculations. Uses array representations for efficiency.
        *   **`pdm_proposal.py` (`PDMProposal`, `PDMProposalManager`)**: `PDMProposal` stores info about one proposal (lateral path index, longitudinal policy index, the `PDMPath`). `PDMProposalManager` holds a collection of all proposals and manages updates to the longitudinal policies (e.g., based on speed limits).
        *   **`batch_idm_policy.py` (`BatchIDMPolicy`)**: An efficient batch implementation of the Intelligent Driver Model (IDM) for longitudinal control. It can handle multiple parameter sets (creating multiple policies) and propagates the state (progress, velocity) forward based on leading agent information.
    *   **`simulation/`**: Contains code for simulating the generated proposals.
        *   **`pdm_simulator.py` (`PDMSimulator`)**: Takes the generated proposal states (as arrays) and simulates them forward using a motion model (`BatchKinematicBicycleModel`) and a trajectory tracker (`BatchLQRTracker`). Re-implements parts of nuPlan's simulation pipeline for batch processing.
        *   **`batch_kinematic_bicycle.py` (`BatchKinematicBicycleModel`)**: Implements a standard kinematic bicycle model, but modified to operate efficiently on batches (arrays) of ego states. Includes simple first-order delays for control inputs.
        *   **`batch_lqr.py` (`BatchLQRTracker`)**: Implements a Linear Quadratic Regulator (LQR) designed to make the kinematic model follow the reference proposal trajectories. It operates on batches, decouples lateral and longitudinal control, and uses utilities from `batch_lqr_utils.py` to estimate velocity/curvature profiles needed for linearization. Includes a fallback P-controller for stopping.
        *   **`batch_lqr_utils.py`**: Provides helper functions for `BatchLQRTracker`, primarily focused on estimating kinematic states (velocity, acceleration, curvature, curvature rate) from a sequence of poses using regularized least squares.
    *   **`scoring/`**: Evaluates the quality of the simulated proposals.
        *   **`pdm_scorer.py` (`PDMScorer`, `PDMScorerConfig`)**: The core scoring class. Takes simulated proposal trajectories (as arrays), the `PDMObservation`, centerline path, route info, and drivable area map. It calculates various metrics re-implementing nuPlan closed-loop metrics: collision checking (`_calculate_no_at_fault_collision`), drivable area compliance, driving direction, traffic light compliance, progress along centerline, time-to-collision (TTC), lane keeping, and comfort (`_calculate_history_comfort`). `PDMScorerConfig` holds weights and thresholds.
        *   **`pdm_and_traffic_scorer.py` (`PDMTrafficScorer`)**: Extends `PDMScorer`. Its main purpose seems to be evaluating *both* the ego's proposed trajectory *and* the simulated trajectories of surrounding traffic agents (comparing them against log replay). It involves creating agent-centric observations and scoring the agents' simulated behavior using the standard PDM metrics.
        *   **`pdm_comfort_metrics.py`**: Defines functions to calculate comfort metrics like longitudinal/lateral acceleration/jerk and yaw rate/acceleration based on state arrays. Includes thresholds from nuPlan/other works. Also implements `ego_is_two_frame_extended_comfort` for comparing comfort consistency between subsequent planning steps.
        *   **`pdm_scorer_utils.py`**: Contains utility functions specifically for the scorer, like `get_collision_type` to classify collisions based on ego/track state.
    *   **`utils/`**: General utilities for the PDM planner.
        *   **`pdm_enums.py`**: Crucial enumerations (`StateIndex`, `SE2Index`, `PointIndex`, `BBCoordsIndex`, etc.) defining the structure (indices) of the NumPy arrays used throughout the PDM planner for representing states, poses, bounding boxes, etc. This enables efficient batch operations.
        *   **`pdm_array_representation.py`**: Functions for converting between nuPlan/navsim object representations (`EgoState`, `StateSE2`) and the NumPy array representations defined by `pdm_enums.py`. Essential for interfacing between nuPlan data and the PDM planner's internal batch processing.
        *   **`pdm_geometry_utils.py`**: Basic geometric functions like angle normalization, calculating parallel paths, coordinate transformations (e.g., absolute to relative), calculating progress along a path, and calculating velocity/acceleration at shifted points on a rigid body.
        *   **`pdm_path.py` (`PDMPath`)**: Represents a path (like a centerline) defined by discrete `StateSE2` waypoints. Provides methods for interpolation (`interpolate`), calculating length, getting a `LineString` representation, and extracting substrings.
        *   **`route_utils.py`**: Functions for finding the current roadblock ego is likely on and correcting the route plan if ego deviates, using BFS on the roadblock graph. Includes logic to remove loops.
        *   **`pdm_emergency_brake.py` (`PDMEmergencyBrake`)**: Implements logic to generate a hard braking trajectory if the scorer detects an imminent collision or TTC violation below a threshold.
        *   **`graph_search/dijkstra.py` (`Dijkstra`)**: Implements Dijkstra's algorithm for finding the shortest path on the *lane graph*, used by `AbstractPDMPlanner` to find the centerline.
        *   **`graph_search/bfs_roadblock.py` (`BreadthFirstSearchRoadBlock`)**: Implements Breadth-First Search on the *roadblock graph*, used by `route_utils` for route correction.

**3. `navsim.planning.metric_caching`**: Pre-computes and stores data needed for efficient evaluation, heavily leveraging the PDM planner components.

*   **`__init__.py`**: Package marker.
*   **`metric_cache.py` (`MetricCache`, `MapParameters`)**: Defines the `MetricCache` dataclass. This structure holds pre-computed information for a specific scenario timestep (token), including the PDM-Closed planner's output trajectory, interpolated observations (`PDMObservation`), the centerline (`PDMPath`), route info, drivable area map, past/current/future tracked objects, and map parameters. `MapParameters` stores map info.
*   **`metric_caching_utils.py` (`StateInterpolator`)**: A utility class to interpolate state arrays over time, used during the caching process.
*   **`caching.py` (`cache_scenarios`, `cache_data`)**: Orchestrates the metric caching process. `cache_data` sets up the scene loader and distributes the caching tasks using a `WorkerPool`. `cache_scenarios` is the function executed by each worker, setting up a `MetricCacheProcessor` and calling it for each assigned scenario. It saves metadata about the cached files.
*   **`metric_cache_processor.py` (`MetricCacheProcessor`)**: The core class responsible for *computing* the data stored in `MetricCache`. For a given scenario, it initializes and runs the internal `PDMClosedPlanner` to get the reference trajectory, centerline, and drivable map. It interpolates ground truth observations using `_interpolate_gt_observation` and `_interpolate_traffic_light_status`, then uses these to build the `PDMObservation`. Finally, it bundles everything into a `MetricCache` object and saves it.

**4. `navsim.planning.scenario_builder`**: Adapts NAVSIM data to the nuPlan scenario interface.

*   **`__init__.py`**: Package marker.
*   **`navsim_scenario.py` (`NavSimScenario`)**: Acts as a wrapper around a NAVSIM `Scene` object, implementing the `nuplan.planning.scenario_builder.abstract_scenario.AbstractScenario` interface. This allows NAVSIM data to be used within frameworks expecting nuPlan scenarios (like the PDM planner, metric caching, and potentially nuPlan's simulation/evaluation tools). It handles requests for ego states, tracked objects, timestamps, etc., at different iterations by accessing the underlying `Scene` data.
*   **`navsim_scenario_utils.py`**: Contains crucial helper functions used by `NavSimScenario` to convert NAVSIM data types (like `EgoStatus`, `Annotations`) into nuPlan data types (`EgoState`, `DetectionsTracks`, `OrientedBox`).

**5. `navsim.planning.script`**: Contains executable scripts and configuration files for running training, caching, and evaluation.

*   **`__init__.py`**: Package marker.
*   **`run_training.py`**: The main script to train a planning agent. It uses Hydra for configuration, sets up the agent, dataset(s) (either `Dataset` or `CacheOnlyDataset`), dataloaders, the `AgentLightningModule`, and the PyTorch Lightning `Trainer`, then starts the `trainer.fit()` process.
*   **`run_dataset_caching.py`**: A script dedicated to running the feature/target caching for training (using `navsim.planning.training.dataset.Dataset`). It distributes the work using a `WorkerPool`.
*   **`run_metric_caching.py`**: A script dedicated to running the metric caching process (using `navsim.planning.metric_caching.caching.cache_data`).
*   **`run_pdm_score_from_submission.py`**: This script takes a pre-computed submission file (a pickle containing agent trajectories for various tokens) and evaluates it using the PDM scoring mechanism (`pdm_score` function, `PDMTrafficScorer`). It loads the corresponding `MetricCache` for each token, runs the scoring, and aggregates the results. It includes logic for handling two-stage evaluation (original + synthetic frames) by inferring/using a mapping and calculating weighted scores. It also computes the two-frame extended comfort metric.
*   **`utils.py`**: Provides common utility functions used by the run scripts, leveraging nuPlan's scripting utilities for setting up logging, worker pools (`WorkerPool`), and handling simulation runners/reports.
*   **`builders/`**: Contains builder functions (using `hydra.instantiate`) to create core components like planners (`build_planners`), observations (`build_observations`), simulation runners (`build_simulations`), and worker pools (`build_worker`) based on Hydra configurations. These are heavily used within the run scripts and potentially within `MetricCacheProcessor`.
*   **`config/`**: Holds Hydra configuration files (`.yaml`).
    *   `common/agent/`: Configs for different agent types (MLP, Human, Constant Velocity, Transfuser).
    *   `common/traffic_agents_policy/`: Configs for simulating traffic agents (IDM, Log Replay, Constant Velocity).
    *   `common/worker/`: Configs for different parallel execution backends (Ray, Sequential, ThreadPool).
    *   Other files (like `default_training.yaml`, `default_metric_caching.yaml`, `default_run_pdm_score_from_submission.yaml`) define the main configurations for the respective scripts, importing common components and setting specific parameters.

**In Summary:**

*   The `planning` module provides a comprehensive framework for both training planning agents and evaluating them, primarily through the lens of the PDM planner and its associated scoring metrics.
*   It heavily utilizes **caching** (for training data and evaluation metrics) to improve efficiency.
*   The **PDM planner** is a core component, involving proposal generation (lateral paths + longitudinal IDM policies), simulation (using LQR tracking on a kinematic model), and scoring (based on collisions, comfort, progress, etc.).
*   **Array representations** (`pdm_enums`, `pdm_array_representation`) are key for efficient batch processing within the PDM components.
*   The code is tightly integrated with **nuPlan's ecosystem**, using its scenario interfaces (`AbstractScenario`), data types (`EgoState`, `DetectionsTracks`), map API, and scripting/builder patterns. `NavSimScenario` acts as the bridge.
*   **Hydra** is used extensively for configuration management across scripts and components.
*   **PyTorch Lightning** is used for structuring the training process.
*   Evaluation seems to focus on the **PDM score**, which aggregates multiple metrics, and can be run on agent outputs directly or via submission files. A specialized scorer (`PDMTrafficScorer`) also evaluates traffic agent behavior. Two-stage (original + synthetic) evaluation is supported.