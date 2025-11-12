Okay, let's analyze the scripts in the `download/` directory.

These are shell scripts designed to automate the download and extraction of the necessary data components for running NAVSIM, including the nuPlan maps and various splits of the OpenScene dataset (which NAVSIM uses).

**Key Actions Performed by the Scripts:**

1.  **Downloading:** They use `wget` to download compressed data files (`.tgz`, `.zip`) from specific URLs. The primary sources are:
    *   Hugging Face (`huggingface.co/datasets/OpenDriveLab/OpenScene/` and `huggingface.co/datasets/AGC2025/warmup_synthetic_scenes/`): For the main OpenScene dataset splits (metadata/logs and sensor data) and the synthetic scenes.
    *   AWS S3 (`s3.amazonaws.com`): Specifically for the curated sensor blobs required for the `navtrain` split and for the nuPlan maps.
2.  **Extracting:** They use `tar -xzf` (for `.tgz`) or `unzip` (for `.zip`) to decompress the downloaded archives.
3.  **Cleaning Up:** They use `rm` to delete the downloaded compressed archives after extraction to save space.
4.  **Organizing:** They use `mv` and occasionally `rsync` to rename and move the extracted directories into a standardized structure, often matching the layout described in `docs/install.md`.

**Specific Scripts and Data:**

*   **`download_maps.sh`**: Downloads and extracts the nuPlan v1.0 maps, placing them in a directory named `maps`.
*   **`download_mini.sh`**: Downloads and extracts the OpenScene `mini` split. It downloads metadata (`openscene_metadata_mini.tgz`) and sensor data (camera and lidar separately, in multiple parts). Organizes them into `mini_navsim_logs` and `mini_sensor_blobs`.
*   **`download_test.sh`**: Downloads and extracts the OpenScene `test` split, following the same pattern as `download_mini.sh` but for the test data. Organizes into `test_navsim_logs` and `test_sensor_blobs`.
*   **`download_trainval.sh`**: Downloads and extracts the large OpenScene `trainval` split. Similar pattern, downloading metadata and many sensor data parts (200 parts each for camera and lidar). Organizes into `trainval_navsim_logs` and `trainval_sensor_blobs`.
*   **`download_navtrain.sh`**: This script is specific to the `navtrain` split mentioned in `docs/splits.md`.
    *   It first downloads the *metadata* (logs) for the full `trainval` split from Hugging Face (`openscene_metadata_trainval.tgz`) and places it in `trainval_navsim_logs`. This confirms `navtrain` uses the same logs as `trainval`.
    *   Then, it downloads specific, smaller sensor data archives (`navtrain_current_*.tgz`, `navtrain_history_*.tgz`) from an AWS S3 bucket.
    *   It extracts these and uses `rsync` to merge them into the `trainval_sensor_blobs/trainval/` directory. This means the curated `navtrain` sensor data lives *within* the standard `trainval` sensor directory structure, allowing the system to find it when using the `navtrain` filter.
*   **`download_private_test_e2e.sh`**: Downloads and extracts the data for the private leaderboard evaluation (`private_test_e2e` split). Organizes into `private_test_e2e_navsim_logs` and `private_test_e2e_sensor_blobs`.
*   **`download_warmup_synthetic_scenes.sh`**: Downloads and extracts the synthetic scenes used for the warmup phase of a competition. This likely creates the `synthetic_scenes` directory structure expected by the dataloaders.

**In Relation to Documentation:**

These scripts directly implement the download steps outlined in `docs/install.md`. They fetch the data corresponding to the different splits described in `docs/splits.md` and organize them into the required `~/navsim_workspace/dataset/` structure. They automate a potentially tedious process of downloading numerous large files.