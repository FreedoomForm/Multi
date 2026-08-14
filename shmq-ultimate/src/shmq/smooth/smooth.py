"""SmoothQuant smooth functions for SHMQ-Ultimate.

Reference: smoothquant/smooth.py — adapted for Qwen2.5 (RMSNorm-based, no LN bias).

Smooth applies scaling to weights and folds the inverse scale into the preceding
norm layer:
    s_j = (max|X_j|)^alpha / (max|W_j|)^(1-alpha)
    ln.weight[j] /= s_j   (fold scale into RMSNorm)
    fc.weight[:, j] *= s_j  (smooth weights, broadcast over output rows)
"""
from __future__ import annotations
from typing import List, Optional, Dict
import torch
import torch.nn as nn
from ..utils import get_module_by_name


@torch.no_grad()
def smooth_ln_fcs_llama_like(ln_module: nn.Module,
                              fcs: List[nn.Linear],
                              alpha: float = 0.5,
                              act_scales: Optional[List[torch.Tensor]] = None,
                              weight_scales: Optional[List[torch.Tensor]] = None,
                              scale_min: float = 1e-5,
                              device: Optional[str] = None) -> List[torch.Tensor]:
    """Smooth a RMSNorm + list of Linear layers (Llama/Qwen2-style).

    For each input channel j:
        s_j = (max|X_j|)^alpha / (max|W_j|)^(1-alpha)
    Then:
        ln.weight[j] /= s_j     (fold scale into RMSNorm weight)
        fc.weight[:, j] *= s_j  (for each fc in fcs)
        fc.bias (if any) *= s_j

    Args:
        ln_module: the preceding RMSNorm (or LayerNorm) module
        fcs: list of Linear layers that share this norm's output as input
        alpha: SmoothQuant alpha (0.5 default)
        act_scales: list of (in_features,) tensors of max|X| per fc (one per fc)
        weight_scales: list of (in_features,) tensors of max|W| per fc (one per fc)
                       — if None, computed from fc.weight.abs().amax(dim=0)
        scale_min: floor for s_j (avoid div-by-zero)
        device: optional device override

    Returns:
        List of scale tensors (one per fc) — useful for inverse / debugging.
    """
    n_fcs = len(fcs)
    if act_scales is None:
        raise ValueError("act_scales must be provided (use get_act_scales to compute).")
    assert len(act_scales) == n_fcs, f"act_scales len ({len(act_scales)}) != fcs len ({n_fcs})"

    # Compute weight scales if not provided
    if weight_scales is None:
        weight_scales = []
        for fc in fcs:
            # max|W_j| over output rows, shape (in_features,)
            ws = fc.weight.abs().amax(dim=0).clamp(min=1e-8)
            weight_scales.append(ws)

    # Aggregate across the parallel fcs (take max, since they share the same input)
    # act_scales[j] should be the same across parallel fcs (they share input),
    # but weight_scales[j] differs per fc.
    # Per SmoothQuant paper, we use max|X| for the activation (shared) and
    # max|W_j| per fc, then average across fcs for the combined weight scale.
    device = device or str(next(ln_module.parameters()).device)
    act_scales_t = torch.stack(act_scales).to(device)  # (n_fcs, in_features)
    weight_scales_t = torch.stack(weight_scales).to(device)

    # Per-channel: take max activation across fcs, mean weight across fcs
    # (max|X| is the same across parallel fcs, so this is just a robustness measure)
    max_act = act_scales_t.amax(dim=0)  # (in_features,)
    mean_weight = weight_scales_t.mean(dim=0)  # (in_features,)

    # Compute scale: s = (max|X|^alpha) / (max|W|^(1-alpha))
    # The classic SmoothQuant formula, clamped to >= scale_min
    scale = (max_act.pow(alpha) / mean_weight.pow(1 - alpha)).clamp(min=scale_min)

    # Apply: ln.weight /= scale; fc.weight[:, j] *= scale (broadcast over output rows)
    if hasattr(ln_module, "weight") and ln_module.weight is not None:
        ln_module.weight.data.div_(scale)
    if hasattr(ln_module, "bias") and ln_module.bias is not None:
        ln_module.bias.data.div_(scale)

    for fc in fcs:
        # fc.weight shape: (out_features, in_features)
        # We multiply each column j by scale[j]
        fc.weight.data.mul_(scale.view(1, -1))
        if fc.bias is not None:
            # fc.bias shape: (out_features,) — bias is OUT-side, not affected by input scale
            # Per SmoothQuant paper, bias is left unchanged when scaling input.
            pass

    return [scale.clone() for _ in fcs]


@torch.no_grad()
def smooth_lm(model: nn.Module,
              layer_names_to_smooth: List[str],
              act_scales: Dict[str, torch.Tensor],
              alpha: float = 0.5,
              scale_min: float = 1e-5) -> Dict[str, torch.Tensor]:
    """Apply SmoothQuant to all (norm, [parallel_fcs]) pairs in the model.

    For Qwen2.5 (and Llama-style models):
        - For each transformer block:
            - input_layernorm <-> [q_proj, k_proj, v_proj]
            - post_attention_layernorm <-> [gate_proj, up_proj]
        - o_proj and down_proj are NOT smoothed (no preceding norm to fold into).

    Args:
        model: HuggingFace LLM
        layer_names_to_smooth: list of names whose INPUT will be smoothed
                               (must include all q/k/v/gate/up across all blocks)
        act_scales: dict {layer_name: (in_features,) max|X| tensor}
        alpha: SmoothQuant alpha (default 0.5)
        scale_min: floor for scale

    Returns:
        Dict {layer_name: scale tensor} for diagnostic purposes.
    """
    # Identify (norm, [fcs]) pairs from the layer names.
    # Pattern: model.layers.{i}.self_attn.{q,k,v}_proj  <-> model.layers.{i}.input_layernorm
    # Pattern: model.layers.{i}.mlp.{gate,up}_proj      <-> model.layers.{i}.post_attention_layernorm
    import re
    block_pattern = re.compile(r"model\.layers\.(\d+)\.(\w+)\.(\w+_proj)")

    pairs: Dict[str, Dict[str, object]] = {}  # key=norm_name, value={fcs: [...], act_scales: [...]}

    for name in layer_names_to_smooth:
        m = block_pattern.match(name)
        if not m:
            continue
        block_idx_str, sub_module, proj_name = m.groups()
        if proj_name in ("q_proj", "k_proj", "v_proj"):
            norm_name = f"model.layers.{block_idx_str}.input_layernorm"
        elif proj_name in ("gate_proj", "up_proj"):
            norm_name = f"model.layers.{block_idx_str}.post_attention_layernorm"
        else:
            continue  # o_proj, down_proj — skip
        pairs.setdefault(norm_name, {"fcs": [], "act_scales": []})
        pairs[norm_name]["fcs"].append(name)
        pairs[norm_name]["act_scales"].append(act_scales[name])

    # Apply smoothing per (norm, [fcs]) pair
    all_scales: Dict[str, torch.Tensor] = {}
    for norm_name, info in pairs.items():
        ln_module = get_module_by_name(model, norm_name)
        fc_modules = [get_module_by_name(model, n) for n in info["fcs"]]
        scales = smooth_ln_fcs_llama_like(
            ln_module=ln_module, fcs=fc_modules,
            alpha=alpha, act_scales=info["act_scales"],
            scale_min=scale_min,
        )
        for fc_name, sc in zip(info["fcs"], scales):
            all_scales[fc_name] = sc

    return all_scales
