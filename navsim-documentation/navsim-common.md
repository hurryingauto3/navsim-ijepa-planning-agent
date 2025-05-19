Okay, let's break down the components within the `navsim.common` package based on the provided files. This package seems to contain fundamental definitions and utilities used throughout the `navsim` library.

**1. `navsim.common.enums.py`**

*   **Purpose:** Defines standardized integer enumerations (`IntEnum`) to represent indices or types for various data structures. Using enums makes the code more readable and less prone to errors compared to using raw integer indices ("magic numbers").
*   **Key Enums:**
    *   **`SceneFrameType`**: Simple enum to distinguish between `ORIGINAL` (0) recorded data and `SYNTHETIC` (1) generated/modified data.
    *   **`StateSE2Index`**: Defines indices for accessing components of a 2D pose (x, y, heading) typically stored in a NumPy array.
        *   Indices: `X` (0), `Y` (1), `HEADING` (2).
        *   Convenience: Provides properties like `.X`, `.Y`, `.HEADING` for readability and `.POINT` (slice(0, 2)), `.STATE_SE2` (slice(0, 3)) for easy slicing of pose arrays.
        *   `size()`: Returns the number of elements (3).
    *   **`BoundingBoxIndex`**: Defines indices for accessing components of a 3D bounding box (x, y, z, length, width, height, heading).
        *   Indices: `X` (0) to `HEADING` (6).
        *   Convenience: Properties for each component and slices like `.POINT2D` (xy), `.POSITION` (xyz), `.DIMENSION` (lwh).
        *   `size()`: Returns the number of elements (7).
    *   **`LidarIndex`**: Defines indices for the different channels/features within a LiDAR point cloud array (typically shaped `(num_features, num_points)`).
        *   Indices: `X` (0) to `ID` (5). Includes position, intensity, ring index, and lidar sensor ID.
        *   Convenience: Properties for each component and slices like `.POINT2D` (xy), `.POSITION` (xyz).
        *   `size()`: Returns the number of features (6).
*   **Overall:** This file provides essential constants for consistently accessing data within arrays representing poses, bounding boxes, and LiDAR points.

**2. `navsim.common.__init__.py`**

*   **Purpose:** This is an empty file that marks the `navsim/common/` directory as a Python package, allowing modules within it to be imported using dot notation (e.g., `from navsim.common.enums import StateSE2Index`).

**3. `navsim.common.dataloader.py`**

*   **Purpose:** Contains functions and classes responsible for finding, filtering, and loading scene data from disk, preparing it for use by agents or evaluation metrics. It handles both original log data and potentially synthesized scene data.
*   **Key Components:**
    *   **`filter_scenes` function:**
        *   Loads raw scene data (pickled lists of frame dictionaries) from original log files.
        *   Applies filtering based on a `SceneFilter` object (e.g., filter by log names, specific scene tokens, minimum length, presence of a route, maximum number of scenes).
        *   Splits long logs into smaller, overlapping scenes according to `SceneFilter`'s frame counts and interval.
        *   Returns a dictionary mapping scene tokens (specifically, the token of the frame at the end of the history) to the list of raw frame dictionaries for that scene, and also a list of the *final* frame tokens of these loaded scenes (used to link synthetic scenes).
    *   **`filter_synthetic_scenes` function:**
        *   Loads *synthetic* scene data.
        *   Filters based on `SceneFilter` (log names, specific synthetic tokens).
        *   Crucially, it can link synthetic scenes to the original scenes loaded previously by checking if the synthetic scene's `corresponding_original_scene` metadata matches one of the `stage1_scenes_final_frames_tokens` from `filter_scenes`.
        *   Returns a dictionary mapping synthetic scene tokens to their file path and log name.
    *   **`SceneLoader` class:**
        *   The main interface for loading scene data.
        *   In `__init__`, it orchestrates calls to `filter_scenes` and potentially `filter_synthetic_scenes` based on the `SceneFilter`.
        *   Stores references to data paths and configurations (`SceneFilter`, `SensorConfig`).
        *   Provides methods to:
            *   Get all loadable `tokens` (`.tokens`).
            *   Get the total number of scenes (`__len__`).
            *   Get a specific token by index (`__getitem__`).
            *   **`get_scene_from_token`**: Loads and parses a *full* `Scene` object (including history, future, map API, sensors based on `SensorConfig`) given a token. Handles whether it's an original or synthetic scene.
            *   **`get_agent_input_from_token`**: Loads only the `AgentInput` part of a scene (history ego status + sensors based on `SensorConfig`). More efficient if future ground truth/map isn't needed.
            *   `get_tokens_list_per_log`: Groups loaded tokens by their log file.
    *   **`MetricCacheLoader` class:**
        *   Specialized loader for reading pre-computed metric results (likely PDM scores) stored in a specific cached format (compressed pickles).
        *   Provides a standard loading interface (`tokens`, `__len__`, `__getitem__`, `get_from_token`).
        *   Includes a utility `to_pickle` to aggregate individual cache files into one large pickle.
*   **Overall:** This file abstracts the data loading process, allowing users to easily get `Scene` or `AgentInput` objects based on filtering criteria, without needing to manage the raw file formats or filtering logic directly. It intelligently handles both original and synthetic data sources.

**4. `navsim.common.dataclasses.py`**

*   **Purpose:** Defines the core data structures used to represent scenes, sensor data, agent inputs/outputs, configurations, and results, using Python's `@dataclass`. These provide type hints and structure to the data passed around in `navsim`.
*   **Key Dataclasses:**
    *   **`Camera`**: Represents data for a single camera (image, calibration, path).
    *   **`Cameras`**: Aggregates all 8 `Camera` objects. Includes logic (`from_camera_dict`) to load requested camera images from disk.
    *   **`Lidar`**: Represents the merged LiDAR point cloud data and its path. Includes loading logic (`from_paths`).
    *   **`EgoStatus`**: State of the ego vehicle at one time step (pose, velocity, acceleration, driving command). Includes a flag `in_global_frame`.
    *   **`AgentInput`**: The input provided to an agent for inference. Contains *lists* (history) of `EgoStatus` (in *local* frame relative to current), `Cameras`, and `Lidars`. Classmethod `from_scene_dict_list` handles creation from raw logs, including coordinate conversion and sensor loading based on `SensorConfig`.
    *   **`Annotations`**: Ground truth annotations for a frame (bounding boxes, names, velocities, tokens). Includes validation (`__post_init__`).
    *   **`Trajectory`**: Represents a sequence of future poses (in *local* coordinates) and the sampling information (`TrajectorySampling`). Expected agent output format. Includes validation.
    *   **`SceneMetadata`**: Static information about a scene (log/scene/map names, tokens, frame counts, synthetic scene linkage info).
    *   **`Frame`**: Represents *all* data for a single timestep within a `Scene`, including privileged information like ground truth annotations, global ego status, sensors, route/traffic light info.
    *   **`Scene`**: The main dataclass representing a full scenario slice. Contains `SceneMetadata`, the `AbstractMap` API object, a list of `Frame` objects (history + future), and optional extended future data (for synthetic scenes).
        *   Provides methods `get_future_trajectory`, `get_history_trajectory` (returning `Trajectory` objects in local coordinates) and `get_agent_input` (extracting the non-privileged `AgentInput`).
        *   Includes extensive classmethods (`_build_*`, `from_scene_dict_list`, `load_from_disk`) to handle construction from raw logs or saved pickles.
        *   Includes `save_to_disk` for serializing (primarily synthetic) scenes.
    *   **`SceneFilter`**: Configuration for the `SceneLoader`, defining how scenes are selected and structured (frame counts, interval, filtering rules).
    *   **`SensorConfig`**: Configuration defining which sensors and which history frames an agent requires. Crucial for efficient data loading. Provides `get_sensors_at_iteration` helper and `build_all_sensors`/`build_no_sensors` constructors.
    *   **`PDMResults`**: Stores the detailed results from the PDM evaluation metric.
*   **Overall:** This file is central to `navsim`, defining the structure and types of data used throughout the library. It clearly separates concepts like agent input vs. full scene ground truth and provides methods for data manipulation (e.g., coordinate transforms, extracting trajectories) and construction.

In summary, the `navsim.common` package lays the groundwork by defining standard data types (`enums.py`), structuring data representation (`dataclasses.py`), and providing the tools to load and prepare this data for use (`dataloader.py`).