Okay, let's analyze the documentation files (`.md`) provided. These files offer crucial context about the NAVSIM dataset, evaluation metrics, agent development, and submission process.

**1. `docs/splits.md`**:
*   **Dataset vs. Training/Test Splits:** Clarifies the distinction. *Dataset splits* (`mini`, `trainval`, `test`, `private_test_e2e`) refer to downloadable log/sensor data collections (derived from nuPlan/OpenScene). *Training/Test splits* (`navtrain`, `navtest`, `warmup_test_e2e`) are implemented as *scene filters* applied to the dataset splits, often selecting more challenging or relevant scenarios.
*   **NAVSIM Splits:**
    *   `navtrain`: A filtered subset of `trainval`, recommended for training. Significantly smaller sensor data requirement than full `trainval`. Separate sensor download script provided.
    *   `navtest`: A filtered subset of `test`, recommended for evaluation.
    *   These NAVSIM splits contain overlapping scenes, unlike the standard splits.
*   **Competition Splits:**
    *   `warmup_test_e2e`: A filter applied to the `mini` dataset for warmup/validation.
    *   `private_test_e2e`: A separate data download for the official challenge leaderboard.
*   **Configuration:** Specifies the `train_test_split=...` argument used in the shell scripts/Hydra configs to select the desired split.
*   **Troubleshooting:** Provides MD5 checksums for the `navtrain` download files to verify integrity.

**2. `docs/install.md`**:
*   Provides step-by-step instructions for setting up the NAVSIM environment.
*   **Steps:**
    1.  Clone the `navsim-devkit` repository.
    2.  Download data: Maps (`./download_maps`), and desired dataset splits (`./download_mini`, `./download_trainval`, etc.). Mentions the separate `./download_navtrain` script.
    3.  Organize the downloaded data into a specific workspace structure (`~/navsim_workspace/dataset/maps`, `.../navsim_logs`, `.../sensor_blobs`, etc.).
    4.  Set required environment variables (`NUPLAN_MAP_VERSION`, `NUPLAN_MAPS_ROOT`, `NAVSIM_EXP_ROOT`, `NAVSIM_DEVKIT_ROOT`, `OPENSCENE_DATA_ROOT`) in `~/.bashrc`.
    5.  Create a conda environment using `environment.yml` and install the devkit using `pip install -e .`.

**3. `docs/traffic_agents.md`**:
*   **Reactive Agents (New in v2):** Explains that NAVSIM v2 supports simulating traffic agents that react to the ego vehicle's planned trajectory. **Crucially, the ego agent itself remains non-reactive** (its plan doesn't change based on the simulated future).
*   **Available Policies:** Lists the policies implemented in `navsim.traffic_agents_policies`:
    *   `Log-Replay`: Non-reactive, follows recorded data (same as v1).
    *   `Constant-Velocity`: Non-reactive baseline for debugging.
    *   `IDM`: Simulates *vehicles* reactively using the Intelligent Driver Model. Non-vehicle agents still follow log data.
*   **Selection:** Shows how to select a policy via Hydra override (e.g., `traffic_agents_policy=navsim_IDM_traffic_agents`).
*   **Future Work:** Mentions plans for learning-based traffic simulation models.

**4. `docs/metrics.md`**:
*   **EPDMS (Extended Predictive Driver Model Score):** This is the core evaluation metric for NAVSIM v2.
    *   It extends the original PDMS from NAVSIM v1.
    *   **New Weighted Metrics:**
        *   Lane Keeping (LK): Penalizes large lateral deviation over time (disabled in intersections).
        *   Extended Comfort (EC): Compares dynamic states (accel, jerk, yaw) between *subsequent* planning frames to penalize inconsistent/uncomfortable changes in the plan over time.
    *   **New Multiplier Metrics:**
        *   Driving Direction Compliance (DDC): Penalizes driving against the flow of traffic.
        *   Traffic Light Compliance (TLC): Penalizes running red lights.
    *   **False-Positive Penalty Filtering:** A key improvement. If the *human* driver in the log also violated a rule (resulting in a score of 0 for that metric), the agent's score for that metric is forced to 1. This prevents penalizing the agent for necessary violations (e.g., briefly entering the oncoming lane to avoid an obstacle).
    *   Provides the full EPDMS formula, including the filtering logic and weights for each component. Compares it to the original PDMS formula.
*   **Pseudo Closed-Loop Aggregation:** Explains the two-stage evaluation process designed to approximate closed-loop testing in an open-loop setting:
    1.  **Stage 1:** Evaluate the planner on an initial scene (e.g., 4s horizon) with standard log-replay or IDM traffic agents (EPDMS calculation).
    2.  **Stage 2:** Evaluate the planner on *multiple pre-computed follow-up scenes*. These scenes start ~4s after the initial scene but at slightly different states (simulating results of different possible plans from stage 1). Crucially, these second-stage evaluations use **reactive background traffic**.
    3.  **Weighting:** The scores from the second-stage scenes are weighted based on the proximity of their starting state to the *actual* end state achieved by the planner in the first stage. Closer start states get higher weights.
    4.  **Aggregation:** A weighted average of the second-stage scores is computed, and potentially combined with the first-stage score, to produce the final aggregated metric.
*   **Running Evaluation:** Points to the example evaluation scripts (`run_cv_pdm_score_evaluation.sh`) and explains how to add and run custom agents.

**5. `docs/cache.md`**:
*   **Data Format:** Briefly explains that NAVSIM uses `navsim.common.dataclasses.Scene` objects, containing `Frame` objects, built from the OpenScene dataset (a 2Hz version of nuPlan).
*   **Metric Caching:** Highlights the need for caching preprocessed data (`MetricCache`) due to the computational cost of accessing maps and transforming coordinates for every frame during evaluation. Points to the `run_metric_caching.sh` script for generating this cache.

**6. `docs/agents.md`**:
*   **Agent Interface:** Explains how to create custom agents by inheriting from `navsim.agents.abstract_agent.AbstractAgent`.
*   **Required Methods (All Agents):**
    *   `__init__`, `name`, `initialize`, `get_sensor_config`, `compute_trajectory`.
*   **Required Methods (Learning-Based Agents):**
    *   `get_feature_builders`, `get_target_builders`, `forward`, `compute_loss`, `get_optimizers`, `get_training_callbacks`. Explains the role of Feature/Target Builders.
*   **Inputs:** Details the available inputs: 9 sensor modalities (8 cameras, 1 merged LiDAR) with 2s history at 2Hz, ego status (pose, velocity, accel), and a discrete driving command (left/straight/right based on route only). Emphasizes the potential compute savings of using LiDAR.
*   **Output:** Explains the required output format (`navsim.common.dataclasses.Trajectory`) containing relative future poses and sampling info.
*   **Baselines:** Describes the provided baseline agents: `ConstantVelocityAgent`, `EgoStatusMLPAgent` (blind MLP), and `TransfuserAgent` (multi-modal camera+LiDAR). Links to implementations and Hugging Face weights.

**7. `docs/submission.md`**:
*   **Leaderboard:** Points to the official leaderboards on Hugging Face.
*   **Rules:**
    *   **Open Source Requirement:** Submissions must eventually provide open-source code and checkpoints for reproducibility. Links are embedded in the `TEAM_NAME`.
    *   **Multi-Seed Submissions:** Encouraged (min 3 seeds) to show robustness. Leaderboard displays mean/std/max for these.
*   **Submission Creation:**
    *   Use `run_create_submission_pickle.sh` (or similar) to generate a `submission.pkl` file containing trajectory predictions for each test scenario. Requires setting team metadata variables.
    *   Use `run_merge_submission_pickles.sh` for multi-seed submissions.
    *   Specifies using `train_test_split=warmup_navsafe_two_stage_extended` for the 2025 warmup.
*   **Submission Process:** Upload the `submission.pkl` as a (private) Hugging Face *model*, then submit the model link to the competition space. Mentions submission limits.
*   **FAQ:** Covers viewing submissions, privacy, troubleshooting failed submissions, and login issues.

**Overall Connection:**

These documents provide the necessary user-facing information to understand the NAVSIM dataset structure, the evaluation methodology (EPDMS and pseudo closed-loop), how to develop and train agents within the framework, how to run standard workflows using the provided scripts, and finally, how to participate in the leaderboard challenge. They complement the code by explaining the "why" and "how" from a user's perspective.