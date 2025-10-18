#!/bin/bash
# =============================================================================
# Diagnostic Script - Check What's Available on Torch
# Run this to see what options you have
# =============================================================================

echo "=============================================="
echo "Torch Environment Diagnostic"
echo "=============================================="
echo ""

echo "1. Checking for Module System"
echo "----------------------------------------"
if command -v module &> /dev/null; then
    echo "✓ Module command found"
    echo ""
    echo "Available Python/Conda/PyTorch modules:"
    module avail 2>&1 | grep -iE "python|conda|pytorch|cuda" || echo "  None found"
else
    echo "✗ No module system found"
fi
echo ""

echo "2. Checking for Conda"
echo "----------------------------------------"
if command -v conda &> /dev/null; then
    echo "✓ Conda found: $(which conda)"
    echo "  Version: $(conda --version)"
    echo ""
    echo "  Conda environments:"
    conda env list
else
    echo "✗ Conda not in PATH"
    echo "  Checking standard locations..."
    if [ -d "$HOME/.conda" ]; then
        echo "  ✓ Found: $HOME/.conda"
    fi
    if [ -d "/opt/conda" ]; then
        echo "  ✓ Found: /opt/conda"
    fi
    if [ -d "/gpfsnyu/home/$USER/.conda" ]; then
        echo "  ✓ Found (Greene): /gpfsnyu/home/$USER/.conda"
    fi
fi
echo ""

echo "3. Checking for Python"
echo "----------------------------------------"
if command -v python &> /dev/null; then
    echo "✓ Python found: $(which python)"
    python --version
    echo ""
    echo "  Checking for PyTorch:"
    python -c "import torch; print(f'  ✓ PyTorch {torch.__version__}'); print(f'  CUDA available: {torch.cuda.is_available()}')" 2>/dev/null || echo "  ✗ PyTorch not installed"
else
    echo "✗ Python not in PATH"
fi
echo ""

echo "4. Checking Greene Home Accessibility"
echo "----------------------------------------"
if [ -d "/gpfsnyu/home/$USER" ]; then
    echo "✓ Greene home accessible: /gpfsnyu/home/$USER"
    if [ -d "/gpfsnyu/home/$USER/.conda" ]; then
        echo "  ✓ Greene conda environments found!"
        ls -la /gpfsnyu/home/$USER/.conda/envs/ 2>/dev/null || echo "  No envs directory"
    fi
else
    echo "✗ Greene home not accessible from Torch"
fi
echo ""

echo "5. Checking Available Containers"
echo "----------------------------------------"
echo "Containers in /share/apps/images/:"
ls /share/apps/images/*.sif 2>/dev/null | head -10
echo "  ... (showing first 10)"
echo ""

echo "6. Checking NAVSIM Installation"
echo "----------------------------------------"
NAVSIM_DIR="/scratch/$USER/navsim_workspace/navsim"
if [ -d "$NAVSIM_DIR" ]; then
    echo "✓ NAVSIM directory found: $NAVSIM_DIR"
    if [ -f "$NAVSIM_DIR/environment.yml" ]; then
        echo "  ✓ environment.yml found"
    fi
    if [ -f "$NAVSIM_DIR/setup.py" ]; then
        echo "  ✓ setup.py found"
    fi
else
    echo "✗ NAVSIM directory not found"
fi
echo ""

echo "=============================================="
echo "Recommendations"
echo "=============================================="
echo ""

# Provide recommendation based on findings
HAS_CONDA=false
HAS_PYTHON=false
HAS_GREENE_HOME=false

command -v conda &> /dev/null && HAS_CONDA=true
command -v python &> /dev/null && HAS_PYTHON=true
[ -d "/gpfsnyu/home/$USER" ] && HAS_GREENE_HOME=true

if $HAS_CONDA; then
    echo "✅ BEST: Use existing conda"
    echo "   cd /scratch/$USER/navsim_workspace/navsim"
    echo "   conda env create -n navsim -f environment.yml"
    echo "   conda activate navsim"
elif $HAS_GREENE_HOME; then
    echo "✅ GOOD: Use Greene conda environment"
    echo "   Set up on Greene first, then:"
    echo "   export PATH=\"/gpfsnyu/home/$USER/.conda/bin:\$PATH\""
    echo "   conda activate navsim"
elif $HAS_PYTHON; then
    echo "⚠️  RISKY: Python found but no conda"
    echo "   You can try pip install, but Torch docs warn against host installs"
    echo "   Better to use a container approach"
else
    echo "❌ CONTAINER REQUIRED"
    echo "   No Python/Conda found. You need to:"
    echo "   1. Build a container (see build_navsim_container.sh)"
    echo "   2. Or copy from Greene (see setup_greene_conda_transfer.sh)"
    echo "   3. Or contact HPC support: hpc@nyu.edu"
fi
echo ""
