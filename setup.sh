#!/bin/bash
# Setup script for video-to-3dgs.
# Uses uv for fast, reproducible dependency management.
#
# Usage:
#   bash setup.sh                          # auto-detect everything
#   bash setup.sh --cuda 12.4              # specify CUDA version
#   bash setup.sh --venv /transfer/.venv   # custom venv location (for disk quota)
#   bash setup.sh --cache /transfer/.cache # custom uv cache (for disk quota)

set -e

CUDA_VERSION=""
VENV_DIR=".venv"
CACHE_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --cuda)  CUDA_VERSION="$2"; shift 2 ;;
        --venv)  VENV_DIR="$2"; shift 2 ;;
        --cache) CACHE_DIR="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Export cache dir if specified (avoids home disk quota issues)
if [ -n "$CACHE_DIR" ]; then
    export UV_CACHE_DIR="$CACHE_DIR"
    echo "UV cache: $CACHE_DIR"
fi

echo "=== video-to-3dgs setup ==="

# ── Check uv ─────────────────────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "uv: $(uv --version)"

# ── Detect CUDA ──────────────────────────────────────────────────────────────
if [ -z "$CUDA_VERSION" ]; then
    if command -v nvidia-smi &>/dev/null; then
        DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
        echo "NVIDIA driver: $DRIVER"
        CUDA_VERSION="12.4"
    else
        echo "No NVIDIA GPU detected. Installing CPU-only torch."
        CUDA_VERSION="cpu"
    fi
fi
echo "CUDA target: $CUDA_VERSION"

# ── Create venv with Python 3.10-3.12 ────────────────────────────────────────
# Python 3.13+ lacks PyTorch CUDA wheels; force 3.10 for maximum compatibility.
echo "Creating venv at $VENV_DIR (Python 3.10)..."
uv venv "$VENV_DIR" --python 3.10 2>/dev/null || uv venv "$VENV_DIR" --python 3.11 2>/dev/null || uv venv "$VENV_DIR" --python 3.12 2>/dev/null || uv venv "$VENV_DIR"
echo "Python: $("$VENV_DIR/bin/python" --version)"

# ── Install PyTorch with correct CUDA ─────────────────────────────────────────
if [ "$CUDA_VERSION" = "cpu" ]; then
    echo "Installing PyTorch (CPU)..."
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
else
    CUDA_TAG="cu$(echo $CUDA_VERSION | tr -d '.')"
    echo "Installing PyTorch + CUDA $CUDA_VERSION ($CUDA_TAG)..."
    uv pip install torch torchvision --index-url "https://download.pytorch.org/whl/$CUDA_TAG"
fi

# ── Install gsplat (needs torch for CUDA kernels) ────────────────────────────
echo "Installing gsplat..."
uv pip install "gsplat>=1.0"

# ── Install remaining deps ───────────────────────────────────────────────────
echo "Installing dependencies..."
uv pip install numpy opencv-python Pillow plyfile pycolmap scipy scikit-learn \
    "torchmetrics[image]" tqdm "imageio[ffmpeg]"

# ── Verify ───────────────────────────────────────────────────────────────────
echo ""
echo "=== Verifying installation ==="
"$VENV_DIR/bin/python" -c "
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
print(f'  {\"$VENV_DIR\"}/bin/python run.py --video room.mp4 --output results/')
"
