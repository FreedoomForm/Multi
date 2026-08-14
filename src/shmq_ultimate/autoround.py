"""AutoRound-style learnable rounding via SignSGD (Intel auto-round).

Learns a per-element rounding offset V in [-0.5, 0.5] that minimises the
layer output reconstruction error (or plain weight MSE when no activation
samples are available), using 200 SignSGD steps by default.
"""
from __future__ import annotations

from typing import List, Optional

import torch

from .quant_utils import group_scales, qmax


class SignSGD(torch.optim.Optimizer):
    """p <- p - lr * sign(grad)."""

    def __init__(self, params, lr: float = 5e-3):
        super().__init__(params, dict(lr=lr))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is not None:
                    p.add_(torch.sign(p.grad), alpha=-lr)
        return loss


def _round_ste(t: torch.Tensor) -> torch.Tensor:
    return (t.round() - t).detach() + t


def autoround_layer(
    weight: torch.Tensor,
    bits: int,
    group_size: int,
    act_samples: Optional[List[torch.Tensor]] = None,
    iters: int = 200,
    lr: float = 5e-3,
    scales: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Learn rounding offsets V for `weight` at `bits`.

    weight: (cout, cin);  scales: (cout, cin//group_size) or None (computed).
    Returns V with shape (cout, cin), values in [-0.5, 0.5].
    """
    w = weight.detach().float()
    cout, cin = w.shape
    g = group_size
    ng = cin // g
    if scales is None:
        scales = group_scales(w, bits, g)
    s = scales.float().unsqueeze(-1)  # (cout, ng, 1)
    wg = w.view(cout, ng, g)
    qm = qmax(bits)
    lo, hi = -qm - 1, qm

    V = torch.zeros_like(wg, requires_grad=True)
    opt = SignSGD([V], lr=lr)

    X = None
    if act_samples:
        X = torch.cat([a.float() for a in act_samples], dim=0)  # (N, cin)

    for _ in range(iters):
        opt.zero_grad(set_to_none=True)
        q = torch.clamp(_round_ste(wg / s + V), lo, hi) * s
        wq = q.view(cout, cin)
        if X is not None:
            loss = ((X @ (wq - w).t()) ** 2).mean()
        else:
            loss = ((wq - w) ** 2).mean()
        loss.backward()
        opt.step()
        with torch.no_grad():
            V.clamp_(-0.5, 0.5)

    return V.detach().view(cout, cin)
