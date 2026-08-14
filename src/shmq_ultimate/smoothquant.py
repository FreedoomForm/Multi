"""Step 1 — SmoothQuant activation-outlier migration (Xiao et al., 2023).

Adapted from mit-han-lab/smoothquant `smooth_ln_fcs_llama_like`:
    s_j = max(|X_j|)^alpha / max(|W_j|)^(1-alpha)
    ln.weight /= s ;  W[:, j] *= s_j
Zero inference overhead — scaling folds into existing parameters.
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from .model_utils import BlockInfo, ATTN_IN, MLP_IN


@torch.no_grad()
def collect_act_scales(model, batches, blocks: List[BlockInfo]) -> Dict[str, torch.Tensor]:
    """Per-input-channel abs-max of activations feeding q/k/v and up/gate."""
    scales: Dict[str, torch.Tensor] = {}
    hooks = []

    def make_hook(key):
        def hook(mod, inp, out):
            x = inp[0].detach()
            amax = x.abs().reshape(-1, x.shape[-1]).amax(dim=0).float()
            if key in scales:
                scales[key] = torch.maximum(scales[key], amax)
            else:
                scales[key] = amax
        return hook

    for blk in blocks:
        if "self_attn.q_proj" in blk.linears:
            hooks.append(blk.linears["self_attn.q_proj"].register_forward_hook(
                make_hook(f"{blk.index}.attn_in")))
        if "mlp.gate_proj" in blk.linears:
            hooks.append(blk.linears["mlp.gate_proj"].register_forward_hook(
                make_hook(f"{blk.index}.mlp_in")))

    for b in batches:
        model(b)
    for h in hooks:
        h.remove()
    return scales


@torch.no_grad()
def smooth_ln_fcs(ln: nn.Module, fcs: List[nn.Linear], act_scale: torch.Tensor,
                  alpha: float = 0.5) -> None:
    device = fcs[0].weight.device
    dtype = fcs[0].weight.dtype
    act_scale = act_scale.to(device=device, dtype=torch.float32)
    w_max = torch.stack([fc.weight.abs().amax(dim=0).float() for fc in fcs]).amax(dim=0)
    s = (act_scale.clamp_min(1e-5).pow(alpha)
         / w_max.clamp_min(1e-5).pow(1.0 - alpha)).clamp_min(1e-5)
    ln.weight.div_(s.to(dtype))
    if getattr(ln, "bias", None) is not None:
        ln.bias.div_(s.to(dtype))
    for fc in fcs:
        fc.weight.mul_(s.to(dtype).unsqueeze(0))


@torch.no_grad()
def apply_smoothquant(model, batches, blocks: List[BlockInfo],
                      alpha: float = 0.5) -> int:
    scales = collect_act_scales(model, batches, blocks)
    n = 0
    for blk in blocks:
        key = f"{blk.index}.attn_in"
        if key in scales and blk.input_norm is not None:
            fcs = [blk.linears[r] for r in ATTN_IN if r in blk.linears]
            if fcs:
                smooth_ln_fcs(blk.input_norm, fcs, scales[key], alpha)
                n += 1
        key = f"{blk.index}.mlp_in"
        if key in scales and blk.post_attn_norm is not None:
            fcs = [blk.linears[r] for r in MLP_IN if r in blk.linears]
            if fcs:
                smooth_ln_fcs(blk.post_attn_norm, fcs, scales[key], alpha)
                n += 1
    return n
