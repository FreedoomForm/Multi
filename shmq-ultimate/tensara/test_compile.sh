#!/usr/bin/env bash
# ============================================================================
# SHMQ-Ultimate — Tensara Kernel Compilation Test
# ============================================================================
#
# Verifies that all 6 Tensara .cu files compile cleanly with nvcc.
# Does NOT run them on GPU (Tensara does that for you).
#
# Requirements:
#   - CUDA toolkit (nvcc) installed
#   - Any GPU architecture supported (sm_75..sm_100)
#
# Usage:
#   ./tensara/test_compile.sh
#
# After successful compilation, the .o files are removed.
# This script only checks compilation, not correctness.
# ============================================================================

set -e

cd "$(dirname "$0")"

if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc not found. Install CUDA toolkit first."
    echo "  Ubuntu:  sudo apt install nvidia-cuda-toolkit"
    echo "  Conda:   conda install -c nvidia cuda-nvcc"
    exit 1
fi

echo "=========================================="
echo "  SHMQ-Ultimate Tensara Kernel Test"
echo "=========================================="
echo "nvcc version:"
nvcc --version | grep -E 'release|Build'
echo ""

# Detect compute capability from first available GPU, or default to sm_80 (A100)
GPU_ARCH="${GPU_ARCH:-sm_80}"
if command -v nvidia-smi &> /dev/null; then
    CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' ')
    if [ -n "$CC" ]; then
        # "8.0" → "sm_80"
        GPU_ARCH="sm_$(echo "$CC" | tr -d '.')"
    fi
fi
echo "Compiling for arch: $GPU_ARCH"
echo ""

KERRNELS=(
    "matmul.cu"
    "mxfp4_gemm.cu"
    "mxfp8_gemm.cu"
    "nvfp4_gemm.cu"
    "rmsnorm.cu"
    "softmax.cu"
)

PASS=0
FAIL=0

for k in "${KERRNELS[@]}"; do
    if [ ! -f "$k" ]; then
        echo "  [SKIP] $k (file not found)"
        FAIL=$((FAIL + 1))
        continue
    fi
    echo "  [TEST] compiling $k ..."
    if nvcc -O3 -arch=$GPU_ARCH -Xcompiler "-Wall -Wextra" \
            -c "$k" -o "/tmp/${k%.cu}.o" 2> "/tmp/${k%.cu}.err"; then
        echo "  [PASS] $k"
        PASS=$((PASS + 1))
        rm -f "/tmp/${k%.cu}.o"
    else
        echo "  [FAIL] $k — compile errors:"
        sed 's/^/         /' "/tmp/${k%.cu}.err" | head -30
        FAIL=$((FAIL + 1))
    fi
done

echo ""
echo "=========================================="
echo "  Result: $PASS passed, $FAIL failed"
echo "=========================================="

if [ $FAIL -gt 0 ]; then
    exit 1
fi
