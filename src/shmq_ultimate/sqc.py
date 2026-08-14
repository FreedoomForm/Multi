"""SQC — Salience-aware Quantization Calibration (SliM-LLM style).

Grid-searches a per-group scale multiplier m in [1-r, 1+r] minimising the
salience-weighted quantization error

    err(m) = sum_j sal_j * (w_j - Q(w_j; m*s))^2

where salience sal_j = diag(X X^T)_j is the calibration input energy per
input channel.
"""
from __future__ import annotations

from typing import Optional

import torch

from .quant_utils import group_scales, qmax


@torch.no_grad()
def sqc_calibrate_scales(
    weight: torch.Tensor,
    bits: int,
    group_size: int,
    salience: Optional[torch.Tensor] = None,
    grid: int = 20,
    search_range: float = 0.1,
    base_scales: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return calibrated scales (cout, cin//group_size)."""
    w = weight.detach().float()
    cout, cin = w.shape
    g = group_size
    ng = cin // g
    wg = w.view(cout, ng, g)

    if base_scales is None:
        base_scales = group_scales(w, bits, g)
    s0 = base_scales.float()  # (cout, ng)

    if salience is None:
        sal = torch.ones(cin, device=w.device)
    else:
        sal = salience.detach().float().to(w.device).clamp_min(1e-12)
    salg = sal.view(ng, g).unsqueeze(0)  # (1, ng, g)

    qm = qmax(bits)
    lo, hi = -qm - 1, qm

    best_err = torch.full((cout, ng), float("inf"), device=w.device)
    best_s = s0.clone()
    for i in range(grid + 1):
        m = 1.0 - search_range + (2.0 * search_range) * i / grid
        s = (s0 * m).clamp_min(1e-12).unsqueeze(-1)  # (cout, ng, 1)
        q = torch.clamp(torch.round(wg / s), lo, hi) * s
        err = (salg * (wg - q) ** 2).sum(dim=-1)  # (cout, ng)
        better = err < best_err
        best_err = torch.where(better, err, best_err)
        best_s = torch.where(better, s.squeeze(-1), best_s)
    return best_s
