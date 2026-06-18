#!/bin/bash
# Setup script for video-to-3dgs.
# Installs all dependencies with correct versions to avoid conflicts.
#
# Usage:
#   bash setup.sh                     # auto-detect CUDA
#   bash setup.sh --cuda 12.4         # specify CUDA version
#   bash setup.sh --venv /path/.venv  # custom venv location

set -e

CUDA_VERSION=""
VENV_DIR=".venv"

while [[ $# -gt 0 ]]; do
    case $1 in
        --cuda) CUDA_VERSION="$2"; shift 2 ;;
        --venv) VENV_DIR="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=== video-to-3dgs setup ==="

# Check Python version
PYTHON=${PYTHON:-python3}
PY_VERSION=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$($PYTHON -c "import sys; print(sys.version_info.major)")
PY_MINOR=$($PYTHON -c "import sys; print(sys.version_info.minor)")

if [ "$PY_MAJOR" -lt 3 ] || ([ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]); then
    echo "ERROR: Python >= 3.10 required (found $PY_VERSION)"
    exit 1
fi
if [ "$PY_MINOR" -gt 12 ]; then
    echo "WARNING: Python $PY_VERSION detected. PyTorch+CUDA wheels may not be available."
    echo "         Recommended: Python 3.10-3.12"
fi
echo "Python: $PY_VERSION"

# Detect CUDA if not specified
if [ -z "$CUDA_VERSION" ]; then
    if command -v nvidia-smi &>/dev/null; then
        DRIVER_CUDA=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
        echo "NVIDIA driver: $DRIVER_CUDA"
        # Default to cu124 (widely compatible)
        CUDA_VERSION="12.4"
    else
        echo "No NVIDIA GPU detected. Installing CPU-only torch."
        CUDA_VERSION="cpu"
    fi
fi
echo "CUDA target: $CUDA_VERSION"

# Create venv
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR ..."
    $PYTHON -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
pip install --upgrade pip

# Install PyTorch with correct CUDA
if [ "$CUDA_VERSION" = "cpu" ]; then
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
else
    CUDA_TAG="cu$(echo $CUDA_VERSION | tr -d '.')"
    echo "Installing PyTorch with $CUDA_TAG ..."
    pip install torch torchvision --index-url "https://download.pytorch.org/whl/$CUDA_TAG"
fi

# Install gsplat (needs torch installed first for CUDA extension compilation)
pip install "gsplat>=1.0"

# Install remaining dependencies
pip install \
    numpy opencv-python Pillow plyfile pycolmap scipy scikit-learn \
    torchmetrics[image] tqdm "imageio[ffmpeg]"

# Verify installation
echo ""
echo "=== Verifying installation ==="
python -c "
import torch
print(f'  PyTorch:  {torch.__version__}')
print(f'  CUDA:     {torch.cuda.is_available()} ({torch.version.cuda})')
import gsplat
print(f'  gsplat:   {gsplat.__version__}')
import pycolmap
print(f'  pycolmap: OK')
import cv2
print(f'  OpenCV:   {cv2.__version__}')
print()
print('Setup complete! Run:')
print('  python run.py --video room.mp4 --output results/')
"
