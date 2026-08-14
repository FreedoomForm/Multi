"""Per-layer quantization orchestration:
SQC scale calibration -> AutoRound offsets (INT4 sub-matrix) -> mixed-bit GPTQ.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from .autoround import autoround_layer
from .gptq import gptq_quantize
from .permutation import ChannelPartition
from .quant_utils import group_scales, qmax
from .sqc import sqc_calibrate_scales


@dataclass
class QuantizedLayerResult:
    w_deq: torch.Tensor          # (cout, cin) float — dequantized weight
    codes: torch.Tensor          # (cout, cin) int16 — integer codes (0 for FP16 cols)
    scales: torch.Tensor         # (cout, cin//g) float — per-group scales used
    bits_per_col: torch.Tensor   # (cin,) int — {16, 8, 4}
    partition: ChannelPartition


def bits_vector(cin: int, part: ChannelPartition) -> torch.Tensor:
    b = torch.empty(cin, dtype=torch.int64)
    b[: part.n16] = 16
    b[part.n16 : part.n16 + part.n8] = 8
    b[part.n16 + part.n8 :] = 4
    return b


def quantize_layer(
    weight: torch.Tensor,
    H: torch.Tensor,
    part: ChannelPartition,
    group_size: int = 128,
    act_samples: Optional[List[torch.Tensor]] = None,
    salience: Optional[torch.Tensor] = None,
    enable_sqc: bool = True,
    sqc_grid: int = 20,
    sqc_range: float = 0.1,
    enable_autoround: bool = True,
    autoround_iters: int = 200,
    autoround_lr: float = 5e-3,
    enable_gptq: bool = True,
    gptq_blocksize: int = 128,
    gptq_percdamp: float = 0.01,
) -> QuantizedLayerResult:
    """Quantize one (already permuted) linear weight.

    weight/H/act_samples/salience must already be in permuted channel order.
    """
    w = weight.detach().float()
    cout, cin = w.shape
    g = group_size
    bpc = bits_vector(cin, part)

    # 1) SQC-calibrated per-group scales for each integer bit level.
    scales_map: Dict[int, torch.Tensor] = {}
    for b in (8, 4):
        if enable_sqc:
            scales_map[b] = sqc_calibrate_scales(
                w, b, g, salience=salience, grid=sqc_grid, search_range=sqc_range
            )
        else:
            scales_map[b] = group_scales(w, b, g)

    # 2) AutoRound offsets for the INT4 sub-matrix (most error-prone part).
    V = torch.zeros_like(w)
    i0 = part.n16 + part.n8
    if enable_autoround and part.n4 > 0:
        acts4 = None
        if act_samples:
            acts4 = [a[:, i0:] for a in act_samples]
        V[:, i0:] = autoround_layer(
            w[:, i0:],
            bits=4,
            group_size=g,
            act_samples=acts4,
            iters=autoround_iters,
            lr=autoround_lr,
            scales=scales_map[4][:, i0 // g :],
        )

    # 3) Mixed-bit GPTQ with OBS error compensation.
    if enable_gptq:
        Q, codes, scales_used = gptq_quantize(
            w, H, bpc, group_size=g, scales_map=scales_map,
            blocksize=gptq_blocksize, percdamp=gptq_percdamp, V=V,
        )
    else:
        # RTN fallback
        Q = w.clone()
        codes = torch.zeros(cout, cin, dtype=torch.int16)
        scales_used = torch.zeros(cout, cin // g)
        for col in range(cin):
            b = int(bpc[col])
            gi = col // g
            if b >= 16:
                continue
            s = scales_map[b][:, gi].clamp_min(1e-12)
            qm = qmax(b)
            c = torch.clamp(torch.round(w[:, col] / s + V[:, col]), -qm - 1, qm)
            codes[:, col] = c.to(torch.int16)
            scales_used[:, gi] = s
            Q[:, col] = c * s

    return QuantizedLayerResult(
        w_deq=Q, codes=codes, scales=scales_used, bits_per_col=bpc, partition=part
    )
