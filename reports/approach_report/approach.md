**Project Report Outline: Applying I-JEPA to NAVSIM Planning**

**Abstract:**
*   Briefly introduce the challenge of data-efficient representation learning for autonomous driving planning in NAVSIM.
*   Introduce I-JEPA as a promising self-supervised method for learning semantic visual features.
*   State the project's objective: To pre-train an I-JEPA model on visual data extracted from NAVSIM scenarios and subsequently integrate and fine-tune this model within a NAVSIM agent for the end-to-end trajectory planning task.
*   Mention the evaluation methodology based on NAVSIM's PDM-Score and comparison against baselines, with a focus on assessing potential improvements in performance and label efficiency.

**1. Introduction:**
*   (Adapt from your lit review) Motivation for SSL in Autonomous Driving.
*   Introduce NAVSIM as the target simulation environment and benchmark for planning. Mention its key characteristics (sensor data, planning task, evaluation metric).
*   Introduce I-JEPA and its core idea (predicting representations in latent space). Briefly contrast with MAE/contrastive methods.
*   State the **Problem:** Existing planning agents in NAVSIM (like MLP or Transfuser) are either simplistic or require significant supervised training/complex architectures. Can SSL pre-training, specifically I-JEPA, improve performance or data efficiency for a vision-based planner?
*   **Objective:** Develop and evaluate a NAVSIM planning agent that leverages visual features pre-trained using the I-JEPA self-supervised learning approach on data derived from the NAVSIM environment.
*   **Outline:** Briefly describe the structure of the report (Background, Proposed Approach, Evaluation, etc.).

**2. Background:**
*   **2.1 NAVSIM Environment:**
    *   Goal: Benchmarking autonomous driving planning agents.
    *   Data: Sensor suite (mention cameras, LiDAR), map data, ego status, ground truth trajectories.
    *   Task: Predict future ego-trajectory (e.g., 4 seconds).
    *   Evaluation: PDM-Score (mention its purpose - assessing safety, comfort, progress).
*   **2.2 I-JEPA:**
    *   (Adapt from your lit review) Core concept: Joint-Embedding Predictive Architecture.
    *   Key Components: Context Encoder, Target Encoder (EMA), Predictor.
    *   Mechanism: Predict target patch *representations* (latent space) from context patch representations.
    *   Masking Strategy: Importance of large targets and informative context (cite I-JEPA paper).
    *   Advantages: Learns semantic features, computationally efficient (no decoder/negatives), no reliance on hand-crafted augmentations.

**3. Proposed Approach:**

This is the core methodological section detailing *how* you will conduct the project.

*   **3.1 Phase 1: Data Preparation for I-JEPA Pre-training**
    *   **Goal:** Create a dataset of images suitable for I-JEPA pre-training, derived *only* from the visual data available in NAVSIM logs.
    *   **Source Data:** Utilize the NAVSIM training logs (`navtrain` split).
    *   **Extraction Process:**
        *   Develop a script that uses `navsim.common.dataloader.SceneLoader` to iterate through training scenarios/tokens.
        *   For each relevant timestep within a scene (e.g., every `k` frames or specific keyframes), access the `AgentInput` or `Scene` object.
        *   Extract raw image data from the desired camera(s) (e.g., `cam_f0`, potentially `cam_l0`, `cam_r0` if planning to use a stitched view). Decide if you'll use only the "current" frame (index 3 in history by default) or potentially multiple history frames as separate images. *Recommendation: Start with only the current frame's front camera (`cam_f0`).*
        *   Save these raw images (e.g., as PNG or JPG files) into a standard directory structure suitable for PyTorch's `ImageFolder` dataset (e.g., `navsim_images/train/class_placeholder/image_xxx.png`). Since it's SSL, the class label doesn't matter.
    *   **Dataset Size & Split:** Aim for a substantial number of images (tens or hundreds of thousands, depending on resources). Create a small validation split from the extracted images (e.g., 95% train, 5% val) for monitoring pre-training loss.
    *   **Rationale:** I-JEPA pre-training only requires unlabeled images. Extracting them from NAVSIM ensures the pre-training distribution is relevant to the downstream task environment.

*   **3.2 Phase 2: I-JEPA Pre-training Implementation**
    *   **Goal:** Train the I-JEPA model using the prepared NAVSIM image dataset.
    *   **Framework:** PyTorch and potentially PyTorch Lightning for structuring the training loop.
    *   **Model Architecture:**
        *   **Context Encoder (`f_theta`):** Vision Transformer (ViT). Choose an architecture (e.g., `ViT-S/16`, `ViT-B/16` – start smaller like ViT-Small based on resources) using a library like `timm`. Ensure patch size and expected input resolution are defined.
        *   **Target Encoder (`f_theta_bar`):** Same ViT architecture as the context encoder. Its weights will be an Exponential Moving Average (EMA) of the context encoder's weights.
        *   **Predictor (`g_phi`):** A shallower/narrower Transformer (or potentially MLP) that takes the context encoder's output patch embeddings and predicts the target encoder's output patch embeddings for specific masked locations. Define its depth and width (e.g., following I-JEPA paper's recommendations based on the main encoder size).
    *   **Input/Preprocessing:**
        *   Use the custom `ImageFolder` dataset created in Phase 1.
        *   Apply image transformations *before* the model: Resizing to the ViT's expected input size (e.g., 224x224), normalization (use standard ImageNet stats or calculate from NAVSIM data if feasible).
        *   Implement the **I-JEPA Masking Strategy** within the dataloader's collate function or as a dataset transformation:
            *   Sample `M` target blocks (e.g., M=4) with appropriate scale/aspect ratio (as per paper).
            *   Sample 1 context block (larger scale).
            *   Remove target regions from the context block.
            *   Generate masks indicating which patches belong to context and which to targets.
    *   **Loss Function:** Mean Squared Error (MSE or L2 loss) between the predictor's output embeddings (`s_hat_y`) and the target encoder's output embeddings (`s_y`) for the target patches.
    *   **Optimization:**
        *   Optimizer: AdamW.
        *   Learning Rate Scheduler: Cosine annealing schedule with linear warmup (typical for ViTs/I-JEPA).
        *   EMA Update: Implement the EMA update rule for the target encoder weights (`theta_bar = m * theta_bar + (1-m) * theta`). Choose momentum `m` (e.g., 0.996, potentially scheduled towards 1.0).
    *   **Training Procedure:** Train for a specified number of epochs (e.g., 100-300, monitor validation loss). Use appropriate batch size based on GPU memory. Log the training loss.
    *   **Output:** The saved state dictionary of the trained **Context Encoder (`f_theta`)**.

*   **3.3 Phase 3: NAVSIM Agent Implementation (`IJEPAPlanningAgent`)**
    *   **Goal:** Create the `navsim.agents.abstract_agent.AbstractAgent` subclass that integrates the pre-trained I-JEPA encoder.
    *   **Structure:** (Refer to the detailed Python code provided in the previous answer).
        *   `__init__`: Takes configuration (path to pre-trained encoder, planning head architecture, freeze flag, LR, etc.). Instantiates the ViT encoder (`self._encoder`) and the planning head (`self._planning_head`).
        *   `initialize`: Loads the pre-trained **Context Encoder** weights (from Phase 2 output) into `self._encoder`. Implements logic to freeze/unfreeze encoder parameters based on the config.
        *   `get_sensor_config`: Specifies *only* the camera inputs needed (matching the pre-training and feature builder).
        *   `get_feature_builders`: Returns `IJEPACameraFeatureBuilder` (processes camera images to match encoder input) and `EgoStatusFeatureBuilder` (standard).
        *   `get_target_builders`: Returns `TrajectoryTargetBuilder` (standard).
        *   `forward`:
            1.  Extract camera features using `self._encoder` (potentially within `torch.no_grad()` if frozen). Process encoder output (e.g., take CLS token or average pool patch embeddings).
            2.  Extract ego status features.
            3.  Concatenate image features and ego status features.
            4.  Pass combined features through `self._planning_head` to predict trajectory `[B, T, 3]`.
        *   `compute_loss`: Standard L1 loss between predicted and target trajectories.
        *   `get_optimizers`: Adam optimizer configured for the trainable parameters (planning head only if encoder is frozen, or both if fine-tuning).

*   **3.4 Phase 4: Fine-tuning Strategy**
    *   **Goal:** Train the `IJEPAPlanningAgent` on the NAVSIM planning task using the NAVSIM training infrastructure.
    *   **Framework:** Use `navsim.planning.script.run_training.py`.
    *   **Dataset:** The standard NAVSIM training dataset (`navsim.planning.training.dataset.Dataset`), which provides `Scene` objects. The agent's feature/target builders will process these.
    *   **Training:**
        *   Run `run_training.py` script, specifying your `IJEPAPlanningAgent` configuration (using Hydra).
        *   The script will handle the batching, forward pass (`agent.forward`), loss calculation (`agent.compute_loss`), and optimization (`agent.get_optimizers`).
    *   **Strategy:**
        *   **Initial Fine-tuning:** Start with the I-JEPA encoder *frozen* (`freeze_encoder: true`). Train only the planning head for a number of epochs (e.g., 20-50). This adapts the randomly initialized head to the pre-trained features.
        *   **(Optional) Full Fine-tuning:** After initial head training, potentially unfreeze the encoder (`freeze_encoder: false`) and continue training the entire network (encoder + head) for more epochs, possibly with a lower learning rate for the encoder parameters to avoid catastrophic forgetting of the pre-trained features.
    *   **Label Efficiency Experiment:** To explicitly test label efficiency, modify the `run_training.py` script or the `Dataset` setup to only use a fraction (e.g., 1%, 10%, 50%) of the available NAVSIM training scenarios. Train separate models for each fraction and compare their final performance.

**4. Evaluation Plan:**
*   **4.1 Metrics:**
    *   Primary: Overall PDM-Score on the NAVSIM `navtest` set.
    *   Secondary: Analyze sub-components of the PDM-Score (Collision, Comfort, Progress, Off-Route, etc.) to understand specific strengths/weaknesses.
*   **4.2 Baselines:**
    *   `ConstantVelocityAgent`: Simple non-learning baseline.
    *   `EgoStatusMLPAgent`: Simple learning baseline using only ego state (train this similarly).
    *   **(Optional/Stretch Goal)** `TransfuserAgent`: State-of-the-art multi-modal baseline (use pre-trained weights if available or acknowledge the training complexity).
*   **4.3 Procedure:**
    *   Use the `navsim.planning.script.run_pdm_score.py` script.
    *   Ensure the required `MetricCache` is generated first (`run_metric_caching.sh`).
    *   Run evaluation for the best checkpoint obtained for the `IJEPAPlanningAgent` (after fine-tuning) and the baseline agents.
*   **4.4 Label Efficiency Analysis:** Plot the final PDM-Score (on the full test set) vs. the percentage of training data used during fine-tuning (from the experiment in 3.4) for both the I-JEPA agent and the `EgoStatusMLPAgent`. Compare the curves.

**5. Expected Outcomes and Contributions:**
*   Demonstrate the feasibility of using I-JEPA pre-trained features for end-to-end planning in NAVSIM.
*   Quantify the performance of the I-JEPA-based agent compared to non-SSL and potentially other SSL-inspired (if time permits) baselines using the PDM-Score.
*   Provide evidence for or against the hypothesis that I-JEPA pre-training improves label efficiency for the NAVSIM planning task.
*   Contribute insights into the transferability of semantic features learned via I-JEPA to complex, sequential decision-making tasks like autonomous driving.

**6. Timeline/Milestones (Example - Adjust based on your timeframe):**
*   Weeks 1-2: Setup NAVSIM environment, Data Extraction script development & execution.
*   Weeks 3-6: I-JEPA model implementation, Pre-training execution & debugging.
*   Weeks 7-8: `IJEPAPlanningAgent` implementation & integration. Initial fine-tuning (frozen encoder).
*   Weeks 9-11: Fine-tuning experiments (frozen/unfrozen), Label efficiency experiments (training on subsets).
*   Week 12: Evaluation runs, Baseline runs.
*   Weeks 13-14: Results analysis, Report writing.

**7. Conclusion:**
*   Reiterate the goal and the proposed I-JEPA-based approach for NAVSIM planning.
*   Summarize the expected impact – potentially more data-efficient and performant vision-based planning agents.

---

This structured approach provides a clear roadmap. Remember to break down each phase into smaller, manageable coding and experimentation tasks. Good luck!