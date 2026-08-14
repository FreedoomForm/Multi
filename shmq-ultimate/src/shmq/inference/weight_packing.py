"""Weight packing for SHMQ real INT4/INT8 inference.

After SHMQ permutation, each Linear layer has its input dimension split into:
  - First K channels:    sensitive   -> INT8 weights (W8A8 matmul path)
  - Remaining cin-K:     insensitive -> INT4 weights (W4A8 matmul path)

This module packs the two halves into compact integer buffers with per-group
scales, ready to be consumed by the custom CUDA kernel (or CPU fallback).

Layout conventions
------------------
INT8 weights:
  qweight_int8 : (out_features, K)        int8     (symmetric, signed)
  scales_int8  : (out_features, K // g)   float16  (per-group-of-g)

INT4 weights:
  qweight_int4 : (out_features, (cin-K) // 2)  uint8   (two signed 4-bit values packed per byte)
  scales_int4  : (out_features, (cin-K) // g)  float16 (per-group-of-g)

Permutation buffer:
  perm         : (cin,) int32  (the SHMQ input-channel permutation; the kernel
                                 expects activations ALREADY permuted by the
                                 fused RMSNorm, so this is stored only for
                                 reference / debug)

Activation quantization (per-token symmetric, INT8):
  x_q          : (B, S, cin)   int8
  x_scale      : (B, S, 1)     float16
"""
from __future__ import annotations
from typing import Tuple, Dict
import torch
import torch.nn as nn


# -----------------------------------------------------------------------------
# Per-element quantization helpers (return INTEGER codes, not fake-quant fp)
# -----------------------------------------------------------------------------

def _symmetric_quantize_int(w: torch.Tensor, n_bits: int, group_size: int
                            ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-group symmetric quantization that returns INTEGER codes + scales.

    Args:
        w: (out_features, in_features) float tensor
        n_bits: 4 or 8
        group_size: 128 (must divide in_features)

    Returns:
        codes: (out_features, in_features) int8 tensor holding the integer codes
               (values fit in [-2^(b-1), 2^(b-1)-1], stored in int8 for both
               4-bit and 8-bit cases — for 4-bit the upper nibble is unused)
        scales: (out_features, in_features // group_size) float16 tensor
    """
    out_features, in_features = w.shape
    assert in_features % group_size == 0, \
        f"in_features ({in_features}) must be divisible by group_size ({group_size})"
    n_groups = in_features // group_size
    max_q = 2 ** (n_bits - 1) - 1  # 7 for 4-bit, 127 for 8-bit

    w_g = w.reshape(out_features, n_groups, group_size)
    max_abs = w_g.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = (max_abs / max_q).to(torch.float16)
    # Integer codes (rounding to nearest, clamping to range)
    codes = (w_g / scale.to(w.dtype)).round().clamp(-max_q, max_q).to(torch.int8)
    codes = codes.reshape(out_features, in_features)
    scales = scale.squeeze(-1)  # (out_features, n_groups)
    return codes, scales


def pack_int4(codes_int8: torch.Tensor) -> torch.Tensor:
    """Pack int8 codes (low-nibble only, values in [-7, 7]) into uint8 (2 per byte).

    Layout: byte i (along the in_features dimension) holds:
        high nibble = codes[2*i]
        low  nibble = codes[2*i+1]

    Signed 4-bit values are stored in two's-complement nibble form.
    """
    assert codes_int8.dtype == torch.int8
    assert codes_int8.dim() == 2
    out_features, in_features = codes_int8.shape
    assert in_features % 2 == 0, "in_features must be even for INT4 packing"
    # Convert to uint8 by adding 8 (bias) so range [-8,7] -> [0,15], then pack.
    # Note: although our symmetric range is [-7,7] (max_q=7), GPTQ might push
    # values to -8 in principle; we handle the full [-8,7] range to be safe.
    biased = (codes_int8.to(torch.int16) & 0x0F).to(torch.uint8)  # mask low nibble
    even = biased[:, 0::2]   # (out_features, in_features/2)
    odd  = biased[:, 1::2]
    packed = (even << 4) | odd   # (out_features, in_features/2) uint8
    return packed.contiguous()


def unpack_int4(packed_uint8: torch.Tensor) -> torch.Tensor:
    """Inverse of pack_int4: returns int8 codes in [-8, 7]."""
    assert packed_uint8.dtype == torch.uint8
    even = ((packed_uint8 >> 4) & 0x0F).to(torch.int16)
    odd  = ( packed_uint8       & 0x0F).to(torch.int16)
    # Sign-extend from 4 bits: values 8..15 should become -8..-1
    even = torch.where(even >= 8, even - 16, even)
    odd  = torch.where(odd  >= 8, odd  - 16, odd)
    out = torch.stack([even, odd], dim=-1).flatten(start_dim=1)
    return out.to(torch.int8)


# -----------------------------------------------------------------------------
# SHMQ Linear packer: split permuted weight into INT8 / INT4 halves
# -----------------------------------------------------------------------------

def pack_shmq_linear(weight: torch.Tensor,
                     n_sensitive_channels: int,
                     group_size: int = 128,
                     perm: torch.Tensor = None,
                     precomputed_codes: torch.Tensor = None,
                     precomputed_scales: torch.Tensor = None,
                     precomputed_n_bits: int = None,
                     ) -> Dict[str, torch.Tensor]:
    """Pack a single Linear's weight into SHMQ real-INT4/INT8 buffers.

    Args:
        weight: (out_features, in_features) float16/float32 tensor — ALREADY
                permuted along in_features (i.e. the caller has applied the
                SHMQ permutation to the weight columns).
        n_sensitive_channels: K — number of leading input channels that get
                INT8 (the sensitive cluster Csen from SHMQ Eq.12). The
                remaining (in_features - K) channels get INT4.
        group_size: 128 (default)
        perm: (in_features,) int32 — the permutation applied (kept for debug)
        precomputed_codes: optional (out_features, in_features) int8 tensor of
                GPTQ-optimized integer codes from Step 8. If provided, the
                insensitive half reuses these codes directly (preserving GPTQ
                optimization). The sensitive half is re-quantized at 8-bit
                from the (already-quantized) weight.
        precomputed_scales: optional (out_features, in_features // g) float16
                scales matching `precomputed_codes`.
        precomputed_n_bits: the bit-width used for `precomputed_codes` (4 or 8).

    Returns:
        dict with keys:
          qweight_int8   : (out_features, K)               int8
          scales_int8    : (out_features, K // g)          float16
          qweight_int4   : (out_features, (cin-K) // 2)    uint8   (packed)
          scales_int4    : (out_features, (cin-K) // g)    float16
          n_sensitive    : int (K)
          group_size     : int (g)
          in_features    : int (cin)
          out_features   : int (cout)
          perm           : (cin,) int32 (or None)
    """
    out_features, in_features = weight.shape
    K = int(n_sensitive_channels)
    assert 0 <= K <= in_features, f"K={K} out of range for cin={in_features}"
    assert K % group_size == 0, \
        f"K ({K}) must be divisible by group_size ({group_size})"
    assert (in_features - K) % group_size == 0, \
        f"(cin-K) ({in_features-K}) must be divisible by group_size ({group_size})"

    # Split weight columns: sensitive (first K) -> INT8, insensitive (rest) -> INT4
    w_sens   = weight[:, :K].contiguous()
    w_insens = weight[:, K:].contiguous()

    # ---------- INT8 (sensitive) half ----------
    # Always re-quantize at 8-bit (since the GPTQ codes are 4-bit for 4-bit layers,
    # we need fresh 8-bit codes for the sensitive half). For 8-bit layers (K=cin),
    # we can reuse the precomputed codes if available.
    if K > 0:
        if precomputed_codes is not None and precomputed_n_bits == 8 and K == in_features:
            # Full 8-bit layer: reuse precomputed codes
            codes8 = precomputed_codes[:, :K].to(torch.int8)
            scales8 = precomputed_scales[:, :K // group_size].to(torch.float16)
        else:
            codes8, scales8 = _symmetric_quantize_int(w_sens, n_bits=8, group_size=group_size)
    else:
        codes8 = torch.zeros(out_features, 0, dtype=torch.int8)
        scales8 = torch.zeros(out_features, 0, dtype=torch.float16)

    # ---------- INT4 (insensitive) half ----------
    # Reuse precomputed 4-bit GPTQ codes if available (preserves GPTQ optimization);
    # otherwise RTN quantize from the weight.
    if in_features - K > 0:
        if (precomputed_codes is not None and precomputed_n_bits == 4
                and precomputed_scales is not None):
            codes4 = precomputed_codes[:, K:].to(torch.int8)
            scales4 = precomputed_scales[:, K // group_size:].to(torch.float16)
        else:
            codes4, scales4 = _symmetric_quantize_int(w_insens, n_bits=4, group_size=group_size)
        packed4 = pack_int4(codes4)
    else:
        packed4 = torch.zeros(out_features, 0, dtype=torch.uint8)
        scales4 = torch.zeros(out_features, 0, dtype=torch.float16)

    return {
        "qweight_int8":  codes8.to("cpu"),
        "scales_int8":   scales8.to("cpu"),
        "qweight_int4":  packed4.to("cpu"),
        "scales_int4":   scales4.to("cpu"),
        "n_sensitive":   K,
        "group_size":    group_size,
        "in_features":   in_features,
        "out_features":  out_features,
        "perm":          perm.to("cpu").to(torch.int32) if perm is not None else None,
    }


# -----------------------------------------------------------------------------
# Per-token INT8 activation quantization (returns integer codes + scales)
# -----------------------------------------------------------------------------

def quantize_activation_int8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-token symmetric INT8 activation quantization.

    Args:
        x: (..., in_features) float16/float32

    Returns:
        x_q    : (..., in_features) int8 — integer codes in [-127, 127]
        x_scale: (..., 1)          float16 — multiply x_q by this to dequantize
    """
    max_q = 127
    max_abs = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = (max_abs / max_q).to(torch.float16)
    x_q = (x / scale.to(x.dtype)).round().clamp(-max_q, max_q).to(torch.int8)
    return x_q, scale
