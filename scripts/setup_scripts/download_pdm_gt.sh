#!/bin/bash
# =============================================================================
# Download PDM Ground Truth Scores for Hydra-MDP Training
# 
# This downloads pre-computed PDM scores for each trajectory in the vocabulary.
# These are needed for multi-target distillation during training.
# =============================================================================

set -e

echo "=============================================="
echo "Downloading PDM Ground Truth Scores"
echo "=============================================="
echo ""

# Check environment
if [[ -z "${OPENSCENE_DATA_ROOT}" ]]; then
    echo "Error: OPENSCENE_DATA_ROOT not set"
    echo "Please set: export OPENSCENE_DATA_ROOT=/scratch/ah7072/navsim_workspace/dataset"
    exit 1
fi

# Create directory structure
PDM_ROOT="${OPENSCENE_DATA_ROOT}/traj_pdm_v2"
mkdir -p "${PDM_ROOT}/ori"
mkdir -p "${PDM_ROOT}/random_aug"

echo "Creating directory: ${PDM_ROOT}"
echo ""

# Download original (non-augmented) PDM ground truths
echo "Downloading PDM ground truths (no augmentation)..."
cd "${PDM_ROOT}/ori"

echo "  → Downloading navtrain_8192.pkl (8K vocab)..."
if [[ ! -f "navtrain_8192.pkl" ]]; then
    wget -q --show-progress https://huggingface.co/Zzxxxxxxxx/gtrs/resolve/main/navtrain_8192.pkl
    echo "    ✓ Downloaded navtrain_8192.pkl"
else
    echo "    ⊙ navtrain_8192.pkl already exists, skipping"
fi

echo "  → Downloading navtrain_16384.pkl (16K vocab)..."
if [[ ! -f "navtrain_16384.pkl" ]]; then
    wget -q --show-progress https://huggingface.co/Zzxxxxxxxx/gtrs/resolve/main/navtrain_16384.pkl
    echo "    ✓ Downloaded navtrain_16384.pkl"
else
    echo "    ⊙ navtrain_16384.pkl already exists, skipping"
fi

echo ""
echo "Downloading PDM ground truths (with data augmentation)..."
cd "${PDM_ROOT}/random_aug"

echo "  → Downloading augmentation config..."
if [[ ! -f "rot_30-trans_0-va_0-p_0.5-ensemble.json" ]]; then
    wget -q --show-progress https://huggingface.co/Zzxxxxxxxx/gtrs/resolve/main/rot_30-trans_0-va_0-p_0.5-ensemble.json
    echo "    ✓ Downloaded augmentation config"
else
    echo "    ⊙ Augmentation config already exists, skipping"
fi

echo "  → Downloading augmented PDM scores (this may take a while)..."
if [[ ! -f "aug_traj_pdm.zip" && ! -d "rot_30-trans_0-va_0-p_0.5-ensemble" ]]; then
    wget -q --show-progress https://huggingface.co/Zzxxxxxxxx/gtrs/resolve/main/aug_traj_pdm.zip
    echo "    ✓ Downloaded aug_traj_pdm.zip"
    echo "  → Extracting augmented PDM scores..."
    unzip -q aug_traj_pdm.zip
    rm aug_traj_pdm.zip
    echo "    ✓ Extracted and cleaned up"
else
    echo "    ⊙ Augmented PDM scores already exist, skipping"
fi

echo ""
echo "=============================================="
echo "PDM Ground Truth Download Complete!"
echo "=============================================="
echo ""
echo "Directory structure:"
echo "${PDM_ROOT}/"
echo "├── ori/"
echo "│   ├── navtrain_8192.pkl      (for 8K vocabulary)"
echo "│   └── navtrain_16384.pkl     (for 16K vocabulary)"
echo "└── random_aug/"
echo "    ├── rot_30-trans_0-va_0-p_0.5-ensemble.json"
echo "    └── rot_30-trans_0-va_0-p_0.5-ensemble/"
echo "        └── split_pickles/"
echo ""
echo "These files are referenced in agent configs via:"
echo "  pdm_gt_path: \${oc.env:OPENSCENE_DATA_ROOT}/traj_pdm_v2/ori/navtrain_16384.pkl"
echo ""
