#!/usr/bin/env bash
# Setup script for SHMQ-Ultimate on a GPU machine.
#
# Installs all dependencies needed to:
#   1. Run the full SHMQ-Ultimate pipeline
#   2. Compile and run the custom CUDA kernel
#   3. Evaluate on Qwen2.5-7B-Instruct
#
# Requirements:
#   - NVIDIA GPU with CUDA >= 11.8 (compute capability >= 7.0)
#   - CUDA toolkit (nvcc) installed: https://developer.nvidia.com/cuda-toolkit
#   - Python 3.10+
#   - ~30GB free disk for model download
#
# Usage:
#   chmod +x scripts/gpu/setup_gpu.sh
#   ./scripts/gpu/setup_gpu.sh
#
set -e

echo "======================================================================"
echo "SHMQ-Ultimate GPU Setup"
echo "======================================================================"

# Check CUDA
if ! command -v nvcc &> /dev/null; then
    echo "[ERROR] nvcc not found. Install CUDA toolkit first:"
    echo "  https://developer.nvidia.com/cuda-toolkit"
    exit 1
fi

NVCC_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
echo "nvcc version: $NVCC_VERSION"

# Determine CUDA version for PyTorch
CUDA_MAJOR=$(echo $NVCC_VERSION | cut -d. -f1)
CUDA_MINOR=$(echo $NVCC_VERSION | cut -d. -f2)
if [ "$CUDA_MAJOR" -ge "12" ]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu121"
else
    TORCH_INDEX="https://download.pytorch.org/whl/cu118"
fi
echo "PyTorch index: $TORCH_INDEX"

# Create virtualenv if not exists
VENV_DIR="${VENV_DIR:-.venv}"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtualenv: $VENV_DIR"
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

echo "Installing PyTorch with CUDA support..."
pip install --upgrade pip
pip install torch --index-url "$TORCH_INDEX"

echo "Installing other dependencies..."
pip install transformers accelerate datasets sentencepiece
pip install scipy numpy
pip install pulp  # ILP solver
pip install tqdm rich
pip install matplotlib  # for visualization scripts

# Optional: lm-eval for zero-shot evaluation
echo "Installing lm-eval (for zero-shot evaluation)..."
pip install lm-eval || echo "  [WARN] lm-eval install failed — zero-shot eval will be skipped"

# Verify installation
echo "Verifying installation..."
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Compute capability: {torch.cuda.get_device_capability(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# Build the CUDA kernel
echo ""
echo "Building SHMQ CUDA kernel..."
cd "$(dirname "$0")/../.."
python3 scripts/gpu/build_cuda_kernel.py

echo ""
echo "======================================================================"
echo "Setup complete!"
echo "======================================================================"
echo ""
echo "Next steps:"
echo "  1. Run the full pipeline on Qwen2.5-7B-Instruct:"
echo "     python scripts/gpu/benchmark_qwen7b.py"
echo ""
echo "  2. Evaluate perplexity:"
echo "     python scripts/gpu/eval_perplexity.py --model ./download/qwen7b_shmq_ultimate"
echo ""
echo "  3. Evaluate zero-shot accuracy:"
echo "     python scripts/gpu/eval_zeroshot.py --model ./download/qwen7b_shmq_ultimate"
