#!/bin/bash

# Rebuilds venv and installs all dependencies

set -euo pipefail

echo ""

echo "==========> Checking for uv..."
if ! command -v uv &> /dev/null; then
    echo "Error: uv not found. Please install uv first:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
echo ""

echo "==========> Cleaning up old environment..."
rm -rf .venv
rm -rf __pycache__
echo ""

echo "==========> Creating fresh virtual environment..."
uv venv
source .venv/bin/activate
echo ""

echo "==========> Installing core dependencies..."
uv pip install wheel setuptools ninja packaging kernels
echo ""

echo "==========> Installing PyTorch..."
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
echo ""

echo "==========> Installing diffusers..."
uv pip install --upgrade "git+https://github.com/huggingface/diffusers.git"
echo ""

echo "==========> Installing transformers..."
uv pip install --upgrade "git+https://github.com/huggingface/transformers.git"
echo ""

echo "==========> Installing packages..."
uv pip install tokenizers accelerate gradio peft
echo ""

echo "==========> Installing flash_attn (you may need to tweak the wheel to use)..."
# uv pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.4.22/flash_attn-2.8.1+cu130torch2.9-cp310-cp310-linux_x86_64.whl
# uv pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.4.18/flash_attn-2.8.3+cu130torch2.9-cp312-cp312-linux_x86_64.whl
uv pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.0/flash_attn-2.8.3+cu130torch2.9-cp311-cp311-linux_x86_64.whl
echo ""

echo "=========================================="""
echo "Verifying installation..."
echo "=========================================="
echo ""

python - <<'VERIFY'
import sys
import torch
import gradio

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
print(f"Gradio: {gradio.__version__}")
VERIFY
echo ""

echo "========================================="
echo "Installation complete!"
echo "========================================="
echo ""
echo "To use:"
echo "  ./start.sh"
echo ""
