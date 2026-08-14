"""SmoothQuant activation outlier migration (pre-processing).

SmoothQuant (Xiao et al., 2023) migrates the magnitude of activation outliers
from activations to weights, making both easier to quantize.

  Original:   y = X @ W^T
  Smoothed:   y = (X / s) @ (s * W^T)   = X_smooth @ W_smooth^T
              where s is a per-channel scaling factor

  s_j = max(|X[:, j]|)^α / max(|W[:, j]|)^(1-α)

After smoothing:
  - Activation outliers are "smoothed" (divided by large s)
  - Weights absorb the inverse scaling (multiplied by s)
  - Both are easier to quantize with INT8/INT4

This is a PRE-PROCESSING step: applied once during calibration, before
MixLLM's bit allocation. The smoothed weights are what MixLLM then quantizes.

For SHMQ-Ultimate v2:
  - Apply SmoothQuant to attention projections (q/k/v/o) and MLP up/gate
  - SKIP down_proj (its input is post-SiLU, which doesn't have outliers
    in the same way — SmoothQuant is designed for pre-attention/pre-MLP)
  - Default α = 0.5 (paper default, balanced W/A difficulty)

Reference: https://github.com/mit-han-lab/smoothquant
"""
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Dict, List, Optional


@torch.no_grad()
def compute_smooth_scales(
    weight: torch.Tensor,         # [out_features, in_features]
    activation: torch.Tensor,     # [n_tokens, in_features]
    alpha: float = 0.5,
    clamp_min: float = 1e-5,
) -> torch.Tensor:
    """Compute SmoothQuant per-input-channel scaling factor s.

    s_j = (max(|X[:, j]|)^α * max(|W[:, j]|)^(1-α))  → normalized
    But we use the standard SmoothQuant formulation:
    s_j = max(|X[:, j]|)^α / max(|W[:, j]|)^(1-α)

    Then:
      X_smooth = X / s   (divide activations)
      W_smooth = W * s   (multiply weights, broadcasting along out_features)

    Args:
        weight: [out_features, in_features]
        activation: [n_tokens, in_features]
        alpha: migration strength. 0 = all to weights, 1 = all to activations.
               0.5 = balanced (default).
        clamp_min: minimum scale value (avoid div-by-zero).

    Returns:
        scales: [in_features] tensor of per-channel scaling factors.
    """
    assert weight.shape[1] == activation.shape[1]
    in_features = weight.shape[1]

    # max(|X[:, j]|) per input channel
    act_max = activation.abs().amax(dim=0).clamp(min=clamp_min)  # [in_features]
    # max(|W[:, j]|) per input channel (along out_features axis)
    wt_max = weight.abs().amax(dim=0).clamp(min=clamp_min)       # [in_features]

    # s = act_max^α / wt_max^(1-α)
    scales = (act_max.pow(alpha) / wt_max.pow(1.0 - alpha)).clamp(min=clamp_min)
    return scales


@torch.no_grad()
def apply_smoothquant_to_linear(
    linear: nn.Linear,
    activation: torch.Tensor,
    alpha: float = 0.5,
) -> torch.Tensor:
    """Apply SmoothQuant to a single Linear layer in-place.

    Modifies linear.weight to be smoothed (W * s) and returns the activation
    scales so the caller can also scale activations if needed.

    Args:
        linear: nn.Linear with weight shape [out_features, in_features]
        activation: [n_tokens, in_features] calibration activations
        alpha: SmoothQuant strength

    Returns:
        scales: [in_features] tensor of per-channel scaling factors.
                The caller should divide activations by this before passing
                to the linear: x_smooth = x / scales
    """
    scales = compute_smooth_scales(linear.weight.data, activation, alpha=alpha)
    # W_smooth = W * s (broadcast along out_features)
    linear.weight.data *= scales.unsqueeze(0).to(linear.weight.dtype).to(linear.weight.device)
    # NOTE: We DON'T divide activations here — they're handled by the
    # "smoothquant scale injection" in the prior norm layer.
    # In a full SmoothQuant implementation, we'd also need to multiply the
    # prior RMSNorm/LayerNorm weight by scales, so the activation is
    # effectively divided by scales as a side-effect of the norm.
    return scales


@torch.no_grad()
def inject_scales_into_prior_norm(
    norm_module: nn.Module,
    scales: torch.Tensor,
) -> None:
    """Inject SmoothQuant scales into a prior norm layer's weight.

    For RMSNorm: γ_new = γ * scales
    Then x_new = (x / rms(x)) * γ_new = (x / rms(x)) * γ * scales
    The "x * scales" effectively divides the activation by (1/scales),
    which is the SmoothQuant activation smoothing.

    This is the SmoothQuant "fusion" trick: the activation scaling is absorbed
    into the prior norm's weight vector, so there's zero runtime overhead.

    Args:
        norm_module: nn.Module with `.weight` parameter of shape [in_features]
        scales: [in_features] SmoothQuant scales
    """
    assert hasattr(norm_module, "weight"), \
        f"Module {type(norm_module)} has no .weight attribute"
    assert norm_module.weight.shape == scales.shape, \
        f"Shape mismatch: {norm_module.weight.shape} vs {scales.shape}"
    norm_module.weight.data *= scales.to(norm_module.weight.dtype).to(norm_module.weight.device)


@torch.no_grad()
def apply_smoothquant_to_model(
    model: nn.Module,
    named_linears_with_activations: Dict[str, torch.Tensor],
    alpha: float = 0.5,
    skip_layers: Optional[List[str]] = None,
    norm_layer_map: Optional[Dict[str, str]] = None,
) -> Dict[str, torch.Tensor]:
    """Apply SmoothQuant to all eligible Linear layers in a model.

    Args:
        model: the model to smooth (modified in-place)
        named_linears_with_activations: dict of layer_name → activation tensor
            (activations captured during a calibration forward pass)
        alpha: SmoothQuant strength
        skip_layers: layer name suffixes to skip (default: ["down_proj"])
        norm_layer_map: maps linear suffix → prior norm suffix
            Default: q/k/v/o_proj → input_layernorm
                     gate/up_proj → post_attention_layernorm

    Returns:
        all_scales: dict of layer_name → [in_features] scales (for debugging)
    """
    if skip_layers is None:
        skip_layers = ["down_proj"]  # down_proj input is post-SiLU, no outliers
    if norm_layer_map is None:
        norm_layer_map = {
            "q_proj": "input_layernorm",
            "k_proj": "input_layernorm",
            "v_proj": "input_layernorm",
            "o_proj": "post_attention_layernorm",  # actually o_proj takes attention output, not norm
            "gate_proj": "post_attention_layernorm",
            "up_proj": "post_attention_layernorm",
        }

    all_scales: Dict[str, torch.Tensor] = {}

    for name, linear in model.named_modules():
        if not isinstance(linear, nn.Linear):
            continue
        if any(name.endswith(s) for s in skip_layers):
            continue
        if name not in named_linears_with_activations:
            continue

        activation = named_linears_with_activations[name]
        scales = apply_smoothquant_to_linear(linear, activation, alpha=alpha)
        all_scales[name] = scales

        # Inject scales into prior norm (fusion)
        # Determine which norm layer feeds this linear
        layer_suffix = name.split(".")[-1]
        norm_suffix = norm_layer_map.get(layer_suffix)
        if norm_suffix:
            # Find the prior norm in the same transformer block
            # E.g., for name = "model.layers.5.self_attn.q_proj",
            #       norm is "model.layers.5.input_layernorm"
            block_path = ".".join(name.split(".")[:-2])  # model.layers.5
            norm_name = f"{block_path}.{norm_suffix}"
            norm_module = model.get_submodule(norm_name) if hasattr(model, "get_submodule") else None
            if norm_module is None:
                # Fallback: walk to find it
                obj = model
                for p in norm_name.split("."):
                    obj = getattr(obj, p, None)
                    if obj is None:
                        break
                norm_module = obj
            if norm_module is not None and hasattr(norm_module, "weight"):
                # For o_proj: the prior "norm" is actually the attention output,
                # which doesn't have a simple weight to absorb scales.
                # Skip injection for o_proj — apply scales manually in a wrapper.
                if layer_suffix != "o_proj":
                    inject_scales_into_prior_norm(norm_module, scales)

    return all_scales
