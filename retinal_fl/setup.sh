#!/bin/bash
# =====================================================
#  SETUP SCRIPT — Privacy-Preserving FL for Retinal Disease
#
#  Tested on: Ubuntu 22.04, Python 3.10+, CUDA 12.x
#  Works on : Local (RTX 3060), Kaggle, Colab
# =====================================================

set -e

echo "================================================="
echo "  Setting up environment …"
echo "================================================="

# 1. Create virtual environment (skip on Kaggle/Colab)
if [ -z "$KAGGLE_KERNEL_RUN_TYPE" ] && [ -z "$COLAB_GPU" ]; then
    echo "Creating virtual environment …"
    python3 -m venv venv
    source venv/bin/activate
fi

# 2. Upgrade pip
pip install --upgrade pip

# 3. Install PyTorch (adjust CUDA version if needed)
echo "Installing PyTorch …"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Install all dependencies
echo "Installing dependencies …"
pip install -r requirements.txt

# 5. TenSEAL (may need special handling on some systems)
echo "Installing TenSEAL for Homomorphic Encryption …"
pip install tenseal || echo "[WARN] TenSEAL install failed — HE will be disabled"

# 6. Create output directories
mkdir -p outputs/model_cache

echo ""
echo "================================================="
echo "  Setup complete!"
echo ""
echo "  Next steps:"
echo "    1. Edit config.py → set DATA_ROOT to your dataset path"
echo "    2. Run:  python train_federated.py"
echo "================================================="
