"""Core quantization primitives (per-group symmetric, SHMQ §4.1, g=128)."""
from __future__ import annotations

import torch


def qmax(bits: int) -> int:
    return (1 << (bits - 1)) - 1  # 127 for 8-bit, 7 for 4-bit


@torch.no_grad()
def group_scales(w: torch.Tensor, bits: int, group_size: int) -> torch.Tensor:
    """Symmetric per-group scales along cin. w: (cout, cin) -> (cout, cin//g)."""
    cout, cin = w.shape
    g = w.reshape(cout, cin // group_size, group_size)
    absmax = g.abs().amax(dim=-1)
    return (absmax / qmax(bits)).clamp_min(1e-8)


@torch.no_grad()
def quantize_group_sym(w: torch.Tensor, bits: int, group_size: int,
                       scales: torch.Tensor | None = None):
    cout, cin = w.shape
    if scales is None:
        scales = group_scales(w, bits, group_size)
    g = w.reshape(cout, cin // group_size, group_size)
    q = torch.round(g / scales.unsqueeze(-1)).clamp(-qmax(bits) - 1, qmax(bits))
    return q.reshape(cout, cin).to(torch.int16), scales


@torch.no_grad()
def dequantize_group_sym(codes: torch.Tensor, scales: torch.Tensor,
                         group_size: int) -> torch.Tensor:
    cout, cin = codes.shape
    g = codes.reshape(cout, cin // group_size, group_size).to(scales.dtype)
    return (g * scales.unsqueeze(-1)).reshape(cout, cin)


@torch.no_grad()
def fake_quantize_group_sym(w: torch.Tensor, bits: int, group_size: int,
                            scales: torch.Tensor | None = None) -> torch.Tensor:
    codes, s = quantize_group_sym(w, bits, group_size, scales)
    return dequantize_group_sym(codes, s, group_size)


@torch.no_grad()
def quantize_activation_per_token(x: torch.Tensor, bits: int = 8):
    """Per-token symmetric activation quantization -> (int8 codes, scales)."""
    absmax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    s = absmax / qmax(bits)
    q = torch.round(x / s).clamp(-qmax(bits) - 1, qmax(bits)).to(torch.int8)
    return q, s


@torch.no_grad()
def pack_int4(codes: torch.Tensor) -> torch.Tensor:
    """Pack int4 codes [-8,7] two per byte (high nibble = even index)."""
    c = codes.to(torch.int8)
    hi = (c[..., 0::2] & 0xF).to(torch.uint8)
    lo = (c[..., 1::2] & 0xF).to(torch.uint8)
    return (hi << 4) | lo


@torch.no_grad()
def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of pack_int4, sign-extended int8."""
    hi = (packed >> 4).to(torch.int8)
    lo = (packed & 0xF).to(torch.int8)
    hi = torch.where(hi >= 8, hi - 16, hi)
    lo = torch.where(lo >= 8, lo - 16, lo)
    out = torch.empty(*packed.shape[:-1], packed.shape[-1] * 2, dtype=torch.int8,
                      device=packed.device)
    out[..., 0::2] = hi
    out[..., 1::2] = lo
    return out
