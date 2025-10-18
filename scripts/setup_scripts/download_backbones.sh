#!/bin/bash
# =============================================================================
# Download Pretrained Vision Backbones for GTRS/Hydra-MDP
# 
# Downloads the DD3D pretrained VoV backbone needed for training.
# =============================================================================

set -e

echo "=============================================="
echo "Downloading Pretrained Vision Backbones"
echo "=============================================="
echo ""

# Check environment
if [[ -z "${OPENSCENE_DATA_ROOT}" ]]; then
    echo "Error: OPENSCENE_DATA_ROOT not set"
    echo "Please set: export OPENSCENE_DATA_ROOT=/scratch/ah7072/navsim_workspace/dataset"
    exit 1
fi

# Create models directory
MODELS_DIR="${OPENSCENE_DATA_ROOT}/models"
mkdir -p "${MODELS_DIR}"

echo "Creating directory: ${MODELS_DIR}"
echo ""

# Download DD3D VoV backbone
cd "${MODELS_DIR}"

echo "Downloading DD3D pretrained backbone (VoV-99)..."
if [[ ! -f "dd3d_det_final.pth" ]]; then
    wget -q --show-progress https://huggingface.co/Zzxxxxxxxx/gtrs/resolve/main/dd3d_det_final.pth
    echo "  ✓ Downloaded dd3d_det_final.pth"
else
    echo "  ⊙ dd3d_det_final.pth already exists, skipping"
fi

echo ""
echo "=============================================="
echo "Backbone Download Complete!"
echo "=============================================="
echo ""
echo "Location: ${MODELS_DIR}/dd3d_det_final.pth"
echo ""
echo "This file is referenced in agent configs via:"
echo "  vov_ckpt: \${oc.env:OPENSCENE_DATA_ROOT}/models/dd3d_det_final.pth"
echo ""
