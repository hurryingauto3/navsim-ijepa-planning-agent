Okay, let's analyze these shell scripts. They reside in the `scripts` directory and provide convenient command-line interfaces to run the core functionalities defined in the `navsim.planning.script` Python modules you provided earlier.

**Overall Purpose:**

These scripts automate the execution of common tasks within the NAVSIM planning framework, such as:

1.  **Training:** Running the training loop for different planning agents.
2.  **Evaluation:** Evaluating the performance of different agents using the PDM-Score metric. This includes preparing the necessary metric caches.
3.  **Submission:** Generating prediction files in the format required for submission to a challenge or benchmark.

**Key Observations & Patterns:**

*   **Python Script Execution:** Each script executes a specific Python script from `navsim.planning.script/` (e.g., `run_training.py`, `run_pdm_score.py`, `run_metric_caching.py`, `run_create_submission_pickle.py`).
*   **Hydra Configuration:** They heavily rely on passing command-line arguments in the `key=value` format. This strongly indicates the Python scripts use the Hydra framework for configuration management. These arguments override default settings defined in the `.yaml` config files (located likely in `navsim/planning/script/config/`).
*   **Environment Variables:** They use environment variables like `$NAVSIM_DEVKIT_ROOT` (presumably pointing to the root of the codebase) and `$NAVSIM_EXP_ROOT` (likely for experiment outputs). `CHECKPOINT` is used to pass model paths. Submission scripts use variables for team metadata.
*   **Data Splits:** The `TRAIN_TEST_SPLIT` variable (set to `navtrain` or `navtest`) is consistently used to specify which dataset split configuration (defined in the Hydra configs, e.g., `navsim/planning/script/config/common/train_test_split/navtrain.yaml`) should be used.
*   **Experiment Naming:** `experiment_name` is often set, likely used by Hydra or the Python scripts to create unique output directories for logs and results.

**Breakdown by Subdirectory:**

1.  **`scripts/training/`**:
    *   `run_transfuser_training.sh`: Executes `run_training.py` to train the `transfuser_agent` using the `navtrain` data split.
    *   `run_ego_mlp_agent_training.sh`: Executes `run_training.py` to train the default agent (likely `EgoStatusMLPAgent` based on other scripts) using the `navtrain` split. It overrides the `max_epochs` parameter.

2.  **`scripts/evaluation/`**:
    *   `run_metric_caching.sh`: Executes `run_metric_caching.py` to pre-compute and cache the data needed for PDM-Score evaluation (`MetricCache` objects) for the `navtest` split, saving to `$NAVSIM_EXP_ROOT/metric_cache`. This is likely a prerequisite for running `run_pdm_score.py`.
    *   `run_ego_mlp_agent_pdm_score_evaluation.sh`: Executes `run_pdm_score.py`. It evaluates the `ego_status_mlp_agent` (loaded from the specified `CHECKPOINT`) on the `navtest` split using the PDM score.
    *   `run_transfuser.sh`: Executes `run_pdm_score.py`. Evaluates the `transfuser_agent` (loaded from `CHECKPOINT`) on `navtest` using PDM score. Specifies `single_machine_thread_pool` for the worker configuration.
    *   `run_cv_pdm_score_evaluation.sh`: Executes `run_pdm_score.py`. Evaluates the default agent (likely `constant_velocity_agent`) on `navtest` using PDM score. No checkpoint is needed as it's a rule-based agent.
    *   `run_human_agent_pdm_score_evaluation.sh`: Executes `run_pdm_score.py`. Evaluates the `human_agent` (which likely replays the ground truth trajectory) on `navtest` using PDM score.

3.  **`scripts/submission/`**:
    *   `run_cv_create_submission_pickle.sh`: Executes `run_create_submission_pickle.py` (code not shown, but likely runs an agent and saves trajectories). It uses the `constant_velocity_agent` on the `navtest` split and includes required team metadata to generate a submission file.
    *   `run_merge_submission_pickles.sh`: Executes `run_merge_submission_pickles.py` (code not shown). Takes a list of existing submission pickles (`SUBMISSION_PICKLES`) and merges them, adding the required team metadata. Useful for ensembling or combining results.

**In Relation to the Python Code:**

*   These scripts are the command-line entry points that trigger the execution of the Python scripts analyzed previously (`run_training.py`, `run_metric_caching.py`, `run_pdm_score.py`).
*   The arguments passed (e.g., `agent=transfuser_agent`, `train_test_split=navtest`, `agent.checkpoint_path=$CHECKPOINT`) directly configure the behavior of those Python scripts via Hydra, controlling which agent is used, which data is processed, where models are loaded from, and where results are saved.
*   They demonstrate the practical workflows enabled by the underlying Python codebase: train models, cache evaluation data, evaluate models with PDM-Score, and prepare submissions.