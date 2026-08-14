"""Sensitivity analysis for SHMQ-Ultimate.

Implements:
  * Layer-input statistics capture (Hessian proxy H = X X^T per linear layer)
    following GPTQ / SliM-LLM practice.
  * Inter-layer sensitivity via Fisher information (SHMQ paper Eq. 6-7,
    HAWQ-style trace proxy):
        S_l = (1 / 2|D|) * sum_D sum_i (g_i * dw_i)^2
    where dw = W - Q(W) at a probe bit-width.
  * Fallback inter-layer sensitivity from XX^T diagonal.
  * Intra-layer per-element OBS sensitivity (SHMQ paper Eq. 10/24):
        S_ij = 0.5 * (w_ij - Q(w_ij))^2 / [ (X X^T + lam*mean(diag)*I)^{-1} ]_jj
    and Manhattan channel sensitivity (Eq. 11):  S_j = sum_i |S_ij|.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .model_utils import BlockInfo, layer_key
from .quant_utils import fake_quantize_group_sym


class LayerStats:
    """Accumulated statistics for one linear layer."""

    def __init__(self, cin: int, device: torch.device):
        self.H = torch.zeros(cin, cin, dtype=torch.float32, device=device)
        self.count = 0
        self.inputs: List[torch.Tensor] = []  # a few kept input samples

    def update(self, x: torch.Tensor, keep: bool, max_rows: int = 256):
        # x: (..., cin) -> (N, cin)
        xf = x.detach().reshape(-1, x.shape[-1]).float()
        self.H += xf.t() @ xf
        self.count += xf.shape[0]
        if keep:
            self.inputs.append(xf[:max_rows].cpu())

    def hessian(self) -> torch.Tensor:
        if self.count == 0:
            return self.H
        return self.H / float(self.count)


@torch.no_grad()
def capture_layer_stats(
    model: nn.Module,
    batches: List[torch.Tensor],
    blocks: List[BlockInfo],
    keep_inputs: int = 4,
) -> Dict[str, LayerStats]:
    """Run calibration batches through the model, accumulating H = XX^T
    and keeping a few input samples per linear layer."""
    stats: Dict[str, LayerStats] = {}
    handles = []

    def make_hook(key: str, lin: nn.Linear):
        def hook(_mod, inputs, _out):
            x = inputs[0]
            if key not in stats:
                stats[key] = LayerStats(lin.in_features, torch.device("cpu"))
            st = stats[key]
            st.update(x.cpu(), keep=len(st.inputs) < keep_inputs)
        return hook

    for blk in blocks:
        for role, lin in blk.linears.items():
            key = layer_key(blk.index, role)
            handles.append(lin.register_forward_hook(make_hook(key, lin)))

    try:
        for b in batches:
            model(b)
    finally:
        for h in handles:
            h.remove()
    return stats


def fisher_sensitivity(
    model: nn.Module,
    batches: List[torch.Tensor],
    blocks: List[BlockInfo],
    bits_probe: int = 4,
    group_size: int = 128,
    n_batches: int = 8,
) -> Dict[str, float]:
    """Inter-layer sensitivity via Fisher information (SHMQ Eq. 6-7).

    S_l = (1/2B) * sum_batches sum_i (g_i * dw_i)^2
    with dw = W - fakequant(W, bits_probe).
    """
    model.train(False)
    # Disable grads everywhere first (saves memory: no embedding/lm_head grads)
    prev_rg = {n: p.requires_grad for n, p in model.named_parameters()}
    for p in model.parameters():
        p.requires_grad_(False)
    # Pre-compute dw for each layer.
    dw: Dict[str, torch.Tensor] = {}
    params: Dict[str, torch.Tensor] = {}
    for blk in blocks:
        for role, lin in blk.linears.items():
            key = layer_key(blk.index, role)
            w = lin.weight.data
            wq = fake_quantize_group_sym(w.float(), bits_probe, group_size)
            dw[key] = (w.float() - wq).cpu()
            params[key] = lin.weight
            lin.weight.requires_grad_(True)

    sens: Dict[str, float] = {k: 0.0 for k in dw}
    used = 0
    for b in batches[:n_batches]:
        model.zero_grad(set_to_none=True)
        out = model(b, labels=b)
        out.loss.backward()
        for key, w in params.items():
            g = w.grad
            if g is None:
                continue
            sens[key] += 0.5 * float(((g.detach().float().cpu() * dw[key]) ** 2).sum())
        used += 1
    model.zero_grad(set_to_none=True)
    for n, p in model.named_parameters():
        p.requires_grad_(prev_rg.get(n, False))
    if used > 0:
        for key in sens:
            sens[key] /= used
    return sens


def xxt_sensitivity(
    hessians: Dict[str, "LayerStats"],
    blocks: List[BlockInfo],
    bits_probe: int = 4,
    group_size: int = 128,
) -> Dict[str, float]:
    """Fallback inter-layer sensitivity: 0.5 * sum(dw^2 * H_jj)."""
    sens: Dict[str, float] = {}
    for blk in blocks:
        for role, lin in blk.linears.items():
            key = layer_key(blk.index, role)
            if key not in hessians:
                continue
            w = lin.weight.data.float()
            wq = fake_quantize_group_sym(w, bits_probe, group_size)
            d2 = (w - wq) ** 2  # (cout, cin)
            hdiag = torch.diagonal(hessians[key].hessian()).to(d2.device)
            sens[key] = 0.5 * float((d2 * hdiag.unsqueeze(0)).sum())
    return sens


def obs_inverse_diag(H: torch.Tensor, lam: float = 0.1) -> torch.Tensor:
    """diag of (H + lam*mean(diag(H))*I)^{-1} via Cholesky, pinv fallback."""
    Hf = H.float()
    n = Hf.shape[0]
    damp = lam * torch.mean(torch.diagonal(Hf)).clamp_min(1e-8)
    Hd = Hf + damp * torch.eye(n, dtype=torch.float32, device=Hf.device)
    try:
        L = torch.linalg.cholesky(Hd)
        Hinv = torch.cholesky_inverse(L)
    except Exception:
        Hinv = torch.linalg.pinv(Hd)
    return torch.diagonal(Hinv).clamp_min(1e-12)


def intra_channel_sensitivity(
    weight: torch.Tensor,
    H: torch.Tensor,
    bits_probe: int = 4,
    group_size: int = 128,
    lam: float = 0.1,
) -> torch.Tensor:
    """Per-input-channel Manhattan sensitivity (SHMQ Eq. 10-11).

    Returns tensor of shape (cin,).
    """
    w = weight.float()
    wq = fake_quantize_group_sym(w, bits_probe, group_size)
    inv_diag = obs_inverse_diag(H, lam).to(w.device)  # (cin,)
    S = 0.5 * (w - wq) ** 2 / inv_diag.unsqueeze(0)  # (cout, cin)
    return S.abs().sum(dim=0)
