"""Standalone GPTQ / OBS quantizer with per-column mixed bit-widths.

Implements the GPTQ error-compensation loop (Frantar et al.) without any
AutoGPTQ dependency, extended for SHMQ-Ultimate:
  * per-column bit-widths from the 3-cluster partition ({16, 8, 4});
  * externally calibrated per-group scales (SQC) per bit level;
  * AutoRound learnable rounding offsets V added before rounding.
FP16 columns (bits >= 16) are kept exact and contribute zero error.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from .quant_utils import group_scales, qmax


@torch.no_grad()
def gptq_quantize(
    weight: torch.Tensor,
    H: torch.Tensor,
    bits_per_col: torch.Tensor,
    group_size: int = 128,
    scales_map: Optional[Dict[int, torch.Tensor]] = None,
    blocksize: int = 128,
    percdamp: float = 0.01,
    V: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize `weight` (cout, cin) column-by-column with OBS compensation.

    bits_per_col: (cin,) int tensor with entries in {4, 8, 16}.
    scales_map:   {bits: (cout, cin//group_size)} calibrated scales.
    V:            (cout, cin) rounding offsets or None.

    Returns (W_deq float32, codes int16, scales_used (cout, ng) float32).
    codes for FP16 columns are 0 (unused); scales_used holds, per group, the
    scale of the group's dominant bit level (groups are bit-homogeneous by
    construction of the quanta-aligned partition).
    """
    w = weight.detach().float().clone()
    cout, cin = w.shape
    g = group_size
    ng = cin // g
    dev = w.device

    Hf = H.detach().float().clone().to(dev)
    # Handle dead columns
    dead = torch.diagonal(Hf) == 0
    Hf[dead, dead] = 1.0
    w[:, dead] = 0.0

    damp = percdamp * torch.mean(torch.diagonal(Hf)).clamp_min(1e-8)
    Hf += damp * torch.eye(cin, dtype=torch.float32, device=dev)

    L = torch.linalg.cholesky(Hf)
    Hinv = torch.cholesky_inverse(L)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)  # upper-triangular

    if scales_map is None:
        scales_map = {}
    for b in (8, 4):
        if b not in scales_map:
            scales_map[b] = group_scales(w, b, g)

    Q = torch.zeros_like(w)
    codes = torch.zeros(cout, cin, dtype=torch.int16, device=dev)
    scales_used = torch.zeros(cout, ng, dtype=torch.float32, device=dev)
    bits_per_col = bits_per_col.to(dev)

    for i1 in range(0, cin, blocksize):
        i2 = min(i1 + blocksize, cin)
        cnt = i2 - i1
        W1 = w[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        E1 = torch.zeros_like(W1)
        H1 = Hinv[i1:i2, i1:i2]

        for j in range(cnt):
            col = i1 + j
            wc = W1[:, j]
            d = H1[j, j]
            b = int(bits_per_col[col])
            gidx = col // g
            if b >= 16:
                q = wc
            else:
                s = scales_map[b][:, gidx].clamp_min(1e-12)
                off = V[:, col] if V is not None else 0.0
                qm = qmax(b)
                c = torch.clamp(torch.round(wc / s + off), -qm - 1, qm)
                codes[:, col] = c.to(torch.int16)
                scales_used[:, gidx] = s
                q = c * s
            Q1[:, j] = q
            err = (wc - q) / d
            if j + 1 < cnt:
                W1[:, j + 1 :] -= err.unsqueeze(1) * H1[j, j + 1 :].unsqueeze(0)
            E1[:, j] = err

        Q[:, i1:i2] = Q1
        if i2 < cin:
            w[:, i2:] -= E1 @ Hinv[i1:i2, i2:]

    return Q, codes, scales_used
