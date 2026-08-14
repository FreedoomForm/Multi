"""Kernel loader: compile and load the SHMQ CUDA kernel, with CPU fallback.

If a CUDA-capable GPU is available, the .cu kernel is JIT-compiled via
`torch.utils.cpp_extension.load` and cached under ~/.cache/torch_extensions.
The compiled module exposes a single function:

    shmq_matmul_forward(x_q, x_scale, W_int8, W_int4, w_scale_8, w_scale_4, group_size)

If no GPU is available, we fall back to a pure-PyTorch CPU implementation
that simulates the same parallel two-bit matmul (INT8×INT8 + INT4×INT8).
This is functionally identical to the CUDA path but slower — its purpose is
to enable correctness testing on CPU-only environments (and to serve as a
reference implementation that documents the kernel's intended behavior).
"""
from __future__ import annotations
import os
import logging
from typing import Optional
import torch

log = logging.getLogger(__name__)

_KERNEL_DIR = os.path.dirname(os.path.abspath(__file__))
_CU_SOURCE  = os.path.join(_KERNEL_DIR, "shmq_matmul_kernel.cu")
_EXT_NAME   = "shmq_matmul_cuda"

_cuda_module = None
_cuda_load_attempted = False


def _try_load_cuda():
    """Attempt to JIT-compile and load the CUDA extension. Returns module or None."""
    global _cuda_module, _cuda_load_attempted
    if _cuda_load_attempted:
        return _cuda_module
    _cuda_load_attempted = True

    if not torch.cuda.is_available():
        log.info("[shmq_kernel] CUDA not available — using CPU fallback.")
        return None

    try:
        from torch.utils.cpp_extension import load
        log.info("[shmq_kernel] JIT-compiling CUDA kernel (first run takes ~30s)...")
        _cuda_module = load(
            name=_EXT_NAME,
            sources=[_CU_SOURCE],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-gencode=arch=compute_70,code=sm_70",  # V100
                "-gencode=arch=compute_75,code=sm_75",  # T4 / 20xx
                "-gencode=arch=compute_80,code=sm_80",  # A100
                "-gencode=arch=compute_86,code=sm_86",  # 30xx
                "-gencode=arch=compute_89,code=sm_89",  # 40xx
                "-gencode=arch=compute_90,code=sm_90",  # H100
            ],
            extra_cflags=["-O3"],
            verbose=False,
        )
        log.info("[shmq_kernel] CUDA kernel compiled & loaded successfully.")
        return _cuda_module
    except Exception as e:
        log.warning(f"[shmq_kernel] CUDA load failed ({e}); using CPU fallback.")
        return None


# ----------------------------------------------------------------------------
# CPU fallback — exact reference implementation of the same parallel matmul
# ----------------------------------------------------------------------------

def _cpu_shmq_matmul(
    x_q: torch.Tensor,         # (..., cin) int8
    x_scale: torch.Tensor,     # (..., 1)  float16
    W_int8: torch.Tensor,      # (cout, K_s) int8
    W_int4: torch.Tensor,      # (cout, (cin-K_s)/2) uint8
    w_scale_8: torch.Tensor,   # (cout, K_s/g) float16
    w_scale_4: torch.Tensor,   # (cout, (cin-K_s)/g) float16
    group_size: int,
) -> torch.Tensor:
    """Pure-PyTorch CPU implementation of the SHMQ parallel two-bit matmul.

    This is the reference for what the CUDA kernel does. It is intentionally
    written for clarity rather than speed.
    """
    from .weight_packing import unpack_int4
    assert x_q.dtype == torch.int8
    assert W_int8.dtype == torch.int8
    assert W_int4.dtype == torch.uint8
    assert x_scale.dtype == torch.float16
    assert w_scale_8.dtype == torch.float16
    assert w_scale_4.dtype == torch.float16

    leading_shape = x_q.shape[:-1]
    cin = x_q.shape[-1]
    cout = W_int8.shape[0]
    K_s = W_int8.shape[1]
    g = group_size

    # Flatten leading dims for ease
    x_q_flat = x_q.reshape(-1, cin)                    # (bs, cin)
    bs = x_q_flat.shape[0]

    # ---------- INT8 path: y8 = x_sens @ W_sens^T ----------
    if K_s > 0:
        x_sens = x_q_flat[:, :K_s].to(torch.int32)     # (bs, K_s)
        # Per-group scale expansion: each group of g columns shares one scale.
        # w_scale_8: (cout, K_s/g) -> expand to (cout, K_s)
        ws8_exp = w_scale_8.to(torch.float32).repeat_interleave(g, dim=1)  # (cout, K_s)
        # INT32 matmul
        acc8 = x_sens.to(torch.float32) @ (W_int8.to(torch.int32).to(torch.float32) * ws8_exp).T  # (bs, cout)
    else:
        acc8 = torch.zeros(bs, cout, dtype=torch.float32)

    # ---------- INT4 path: y4 = x_insens @ W_insens^T ----------
    if cin - K_s > 0:
        x_insens = x_q_flat[:, K_s:].to(torch.int32)   # (bs, cin-K_s)
        # Unpack INT4 -> int8 codes
        W4_codes = unpack_int4(W_int4).to(torch.int32)  # (cout, cin-K_s)
        ws4_exp = w_scale_4.to(torch.float32).repeat_interleave(g, dim=1)  # (cout, cin-K_s)
        acc4 = x_insens.to(torch.float32) @ (W4_codes.to(torch.float32) * ws4_exp).T  # (bs, cout)
    else:
        acc4 = torch.zeros(bs, cout, dtype=torch.float32)

    # ---------- Combine + activation scale ----------
    total = (acc8 + acc4) * x_scale.reshape(-1, 1).to(torch.float32)  # (bs, cout)
    return total.reshape(*leading_shape, cout).to(torch.float16)


# ----------------------------------------------------------------------------
# Public dispatch: CUDA if available, else CPU fallback
# ----------------------------------------------------------------------------

def shmq_matmul(
    x_q: torch.Tensor,
    x_scale: torch.Tensor,
    W_int8: torch.Tensor,
    W_int4: torch.Tensor,
    w_scale_8: torch.Tensor,
    w_scale_4: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """SHMQ parallel two-bit matmul. Dispatches to CUDA kernel or CPU fallback.

    Args:
        x_q        : (..., cin)        int8     — INT8 activation codes
        x_scale    : (..., 1)          float16  — per-token activation scale
        W_int8     : (cout, K_s)       int8     — sensitive weight codes
        W_int4     : (cout, (cin-K_s)/2) uint8  — packed insensitive weight codes
        w_scale_8  : (cout, K_s/g)     float16  — INT8 per-group weight scales
        w_scale_4  : (cout, (cin-K_s)/g) float16 — INT4 per-group weight scales
        group_size : int (g)

    Returns:
        y : (..., cout) float16
    """
    use_cuda = x_q.is_cuda and W_int8.is_cuda and torch.cuda.is_available()
    if use_cuda:
        mod = _try_load_cuda()
        if mod is not None:
            # Flatten leading dims of x to (bs, cin) for the kernel
            leading = x_q.shape[:-1]
            cin = x_q.shape[-1]
            x_q_flat = x_q.reshape(-1, cin).contiguous()
            x_scale_flat = x_scale.reshape(-1, 1).contiguous()
            y_flat = mod.shmq_matmul_forward(
                x_q_flat, x_scale_flat,
                W_int8.contiguous(), W_int4.contiguous(),
                w_scale_8.contiguous(), w_scale_4.contiguous(),
                int(group_size),
            )
            return y_flat.reshape(*leading, -1)
        # fall through to CPU if load failed
    return _cpu_shmq_matmul(x_q, x_scale, W_int8, W_int4, w_scale_8, w_scale_4, group_size)


def is_cuda_kernel_available() -> bool:
    """Returns True iff the CUDA kernel compiled and loaded successfully."""
    return _try_load_cuda() is not None
