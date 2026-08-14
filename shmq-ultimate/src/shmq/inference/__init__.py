"""SHMQ Real INT4/INT8 Inference Module.

This module implements the SHMQ paper's parallel two-bit inference path
(§3.2 "MatMul is partitioned into W4A8 and W8A8 operations, similar to QUIK").

After running the SHMQ pipeline (steps 0-8), call `convert_model_to_real_int4`
to swap every nn.Linear with a SHMQQuantLinear that stores REAL packed
INT4+INT8 weights and dispatches to the custom CUDA kernel at inference time.

Components:
  - weight_packing.py    : INT4/INT8 weight packing with per-group scales
  - shmq_matmul_kernel.cu: Custom CUDA kernel for parallel INT4×INT8 + INT8×INT8 matmul
  - kernel_loader.py     : JIT-loads the CUDA kernel; falls back to CPU (PyTorch) if no GPU
  - shmq_quant_linear.py : nn.Module drop-in replacement for nn.Linear
  - model_converter.py   : swap nn.Linear → SHMQQuantLinear across the whole model

Key innovation (the SHMQ paper's 2.86× speedup):
  Both INT4 and INT8 are native CUDA integer formats, so a single custom kernel
  can compute  y = x_sens @ W_int8^T + x_insens @ W_int4^T  in ONE pass with
  NO dequantization overhead. This is the "parallel two-bit inference" idea.
"""
from .weight_packing import (
    pack_shmq_linear,
    pack_int4,
    unpack_int4,
    quantize_activation_int8,
)
from .kernel_loader import (
    shmq_matmul,
    is_cuda_kernel_available,
)
from .shmq_quant_linear import SHMQQuantLinear
from .model_converter import (
    convert_model_to_real_int4,
    convert_model_back_to_fake_quant,
)

__all__ = [
    "pack_shmq_linear",
    "pack_int4",
    "unpack_int4",
    "quantize_activation_int8",
    "shmq_matmul",
    "is_cuda_kernel_available",
    "SHMQQuantLinear",
    "convert_model_to_real_int4",
    "convert_model_back_to_fake_quant",
]
