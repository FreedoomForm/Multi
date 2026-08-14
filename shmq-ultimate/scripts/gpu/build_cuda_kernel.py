#!/usr/bin/env python3
"""Build script: compile & load the SHMQ custom CUDA kernel.

Runs on a GPU machine to verify the kernel compiles cleanly and produces
correct results. This is the first thing to run on a fresh GPU environment.

Usage:
    python scripts/gpu/build_cuda_kernel.py

Requirements:
    - NVIDIA GPU (compute capability >= 7.0, i.e., V100/T4/A100/30xx/40xx/H100)
    - CUDA toolkit (nvcc) >= 11.0
    - PyTorch with CUDA support: pip install torch --index-url https://download.pytorch.org/whl/cu121

Exit codes:
    0 = kernel compiled and passed correctness test
    1 = compilation failed
    2 = correctness test failed
    3 = no CUDA available
"""
from __future__ import annotations
import sys
import os
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


def main():
    print("=" * 70)
    print("SHMQ CUDA Kernel — Build & Verify")
    print("=" * 70)

    import torch
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available:  {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("\n[ERROR] No CUDA-capable GPU detected.")
        print("        Install PyTorch with CUDA support:")
        print("        pip install torch --index-url https://download.pytorch.org/whl/cu121")
        return 3

    print(f"GPU:             {torch.cuda.get_device_name(0)}")
    print(f"Compute cap:     {torch.cuda.get_device_capability(0)}")
    print(f"CUDA version:    {torch.version.cuda}")

    # Check nvcc
    import subprocess
    try:
        nvcc_out = subprocess.check_output(["nvcc", "--version"], stderr=subprocess.STDOUT).decode()
        for line in nvcc_out.splitlines():
            if "release" in line.lower():
                print(f"nvcc:            {line.strip()}")
                break
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("\n[WARNING] nvcc not found on PATH. The kernel cannot be JIT-compiled.")
        print("          Install CUDA toolkit: https://developer.nvidia.com/cuda-toolkit")
        return 1

    # ------------------------------------------------------------------
    # Step 1: JIT-compile the kernel
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Step 1: JIT-compile shmq_matmul_kernel.cu")
    print("-" * 70)

    from shmq.inference.kernel_loader import _try_load_cuda, _CU_SOURCE
    print(f"Source: {_CU_SOURCE}")
    t0 = time.time()
    mod = _try_load_cuda()
    t1 = time.time()

    if mod is None:
        print(f"\n[FAILED] Kernel compilation failed after {t1-t0:.1f}s")
        print("         Check the error output above.")
        return 1

    print(f"\n[OK] Kernel compiled & loaded in {t1-t0:.1f}s")
    print(f"     Module: {mod}")
    print(f"     Function: shmq_matmul_forward")

    # ------------------------------------------------------------------
    # Step 2: Correctness test against CPU reference
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Step 2: Correctness test (CUDA vs CPU reference)")
    print("-" * 70)

    from shmq.inference.weight_packing import (
        _symmetric_quantize_int, pack_int4, quantize_activation_int8
    )
    from shmq.inference.kernel_loader import shmq_matmul

    torch.manual_seed(42)
    device = "cuda"

    # Test parameters
    batch = 4
    cin = 512
    cout = 256
    K_s = 128          # 128 INT8 channels + 384 INT4 channels
    group_size = 128

    print(f"  Test shape: x=({batch},{cin}), W_int8=({cout},{K_s}), "
          f"W_int4=({cout},{(cin-K_s)//2})")
    print(f"  group_size={group_size}")

    # Create random weight and quantize
    W = torch.randn(cout, cin, dtype=torch.float32, device=device)
    W_sens = W[:, :K_s].contiguous()
    W_insens = W[:, K_s:].contiguous()

    codes8, scales8 = _symmetric_quantize_int(W_sens, n_bits=8, group_size=group_size)
    codes4, scales4 = _symmetric_quantize_int(W_insens, n_bits=4, group_size=group_size)
    packed4 = pack_int4(codes4)

    # Create random activation and quantize
    x = torch.randn(batch, cin, dtype=torch.float32, device=device)
    x_q, x_scale = quantize_activation_int8(x)

    # Move to CUDA
    codes8 = codes8.to(device)
    scales8 = scales8.to(device).to(torch.float16)
    packed4 = packed4.to(device)
    scales4 = scales4.to(device).to(torch.float16)
    x_q = x_q.to(device)
    x_scale = x_scale.to(device).to(torch.float16)

    # Run CUDA kernel
    torch.cuda.synchronize()
    t0 = time.time()
    y_cuda = shmq_matmul(x_q, x_scale, codes8, packed4, scales8, scales4, group_size)
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"\n  CUDA kernel time: {(t1-t0)*1000:.2f} ms")

    # Run CPU reference (move tensors to CPU)
    y_cpu = shmq_matmul(
        x_q.cpu(), x_scale.cpu(),
        codes8.cpu(), packed4.cpu(),
        scales8.cpu(), scales4.cpu(),
        group_size,
    ).to(device)

    # Compare
    max_diff = (y_cuda - y_cpu).abs().max().item()
    mean_diff = (y_cuda - y_cpu).abs().mean().item()
    print(f"\n  Max  |y_cuda - y_cpu| = {max_diff:.6f}")
    print(f"  Mean |y_cuda - y_cpu| = {mean_diff:.6f}")

    # Tolerance: the CUDA kernel uses FP32 accumulation, CPU uses FP32 too,
    # so they should match to within floating-point rounding (< 1e-3).
    if max_diff > 0.1:
        print(f"\n[FAILED] Correctness test failed (max_diff={max_diff} > 0.1)")
        return 2

    print(f"\n  [OK] CUDA kernel matches CPU reference within tolerance")

    # ------------------------------------------------------------------
    # Step 3: Performance benchmark
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Step 3: Performance benchmark")
    print("-" * 70)

    # Larger benchmark: realistic LLM dimensions
    batch = 32
    cin = 4096
    cout = 4096
    K_s = 512  # 12.5% of 4096

    print(f"  Benchmark shape: x=({batch},{cin}), cout={cout}, K_s={K_s}")

    W = torch.randn(cout, cin, dtype=torch.float32, device=device)
    W_sens = W[:, :K_s].contiguous()
    W_insens = W[:, K_s:].contiguous()
    codes8, scales8 = _symmetric_quantize_int(W_sens, n_bits=8, group_size=group_size)
    codes4, scales4 = _symmetric_quantize_int(W_insens, n_bits=4, group_size=group_size)
    packed4 = pack_int4(codes4)

    x = torch.randn(batch, cin, dtype=torch.float32, device=device)
    x_q, x_scale = quantize_activation_int8(x)

    codes8 = codes8.to(device)
    scales8 = scales8.to(device).to(torch.float16)
    packed4 = packed4.to(device)
    scales4 = scales4.to(device).to(torch.float16)
    x_q = x_q.to(device)
    x_scale = x_scale.to(device).to(torch.float16)

    # Warmup
    for _ in range(5):
        _ = shmq_matmul(x_q, x_scale, codes8, packed4, scales8, scales4, group_size)
    torch.cuda.synchronize()

    # Timed runs
    n_iters = 50
    t0 = time.time()
    for _ in range(n_iters):
        _ = shmq_matmul(x_q, x_scale, codes8, packed4, scales8, scales4, group_size)
    torch.cuda.synchronize()
    t1 = time.time()
    cuda_ms = (t1 - t0) / n_iters * 1000

    # FP16 baseline
    W_fp16 = W.to(torch.float16)
    x_fp16 = x.to(torch.float16)
    for _ in range(5):
        _ = x_fp16 @ W_fp16.T
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_iters):
        _ = x_fp16 @ W_fp16.T
    torch.cuda.synchronize()
    t1 = time.time()
    fp16_ms = (t1 - t0) / n_iters * 1000

    print(f"\n  SHMQ CUDA kernel: {cuda_ms:.2f} ms/iter")
    print(f"  FP16 cuBLAS:      {fp16_ms:.2f} ms/iter")
    print(f"  Speedup:          {fp16_ms / cuda_ms:.2f}x")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("BUILD & VERIFY: PASSED")
    print("=" * 70)
    print("\nThe SHMQ CUDA kernel is ready for inference.")
    print("Next step: python scripts/gpu/benchmark_qwen7b.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
