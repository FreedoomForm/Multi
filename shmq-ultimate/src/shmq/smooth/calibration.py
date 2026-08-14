"""Activation scale calibration for SmoothQuant.

Hooks onto Linear layers' input, computes per-channel max(|X|) over the
calibration set, returns a dict mapping layer name -> scale tensor.

Reference: smoothquant/calibration.py (we vendor + adapt for Qwen2.5).
"""
from __future__ import annotations
from typing import Dict, List, Optional
import torch
import torch.nn as nn
from ..utils import get_module_by_name


class ActivationScaleCollector:
    """Collect max(|X|) per input channel for a list of Linear layers.

    Usage:
        collector = ActivationScaleCollector(layer_names)
        collector.attach(model)
        with torch.no_grad():
            for batch in calibration_data:
                model(batch)
        scales = collector.get_scales()  # dict {name: tensor}
        collector.detach()
    """

    def __init__(self, layer_names: List[str]):
        self.layer_names = layer_names
        self.handles: List = []
        self.scales: Dict[str, torch.Tensor] = {n: None for n in layer_names}
        self.n_samples: Dict[str, int] = {n: 0 for n in layer_names}

    def attach(self, model: nn.Module):
        for name in self.layer_names:
            mod = get_module_by_name(model, name)
            handle = mod.register_forward_pre_hook(self._make_hook(name))
            self.handles.append(handle)

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def _make_hook(self, name: str):
        def hook(module: nn.Module, inputs):
            x = inputs[0]  # (batch, seq_len, in_features) or (N, in_features)
            if x.dim() == 2:
                x = x.unsqueeze(0)
            # Per-channel max(|X|), reducing over batch and seq
            x = x.detach().abs()
            batch_max = x.amax(dim=tuple(range(x.dim() - 1)))  # (in_features,)
            if self.scales[name] is None:
                self.scales[name] = batch_max.clone()
            else:
                self.scales[name] = torch.maximum(self.scales[name], batch_max)
            self.n_samples[name] += 1
        return hook

    def get_scales(self) -> Dict[str, torch.Tensor]:
        """Return dict {layer_name: scale tensor of shape (in_features,)}."""
        return {n: s.clone() if s is not None else None
                for n, s in self.scales.items()}


def get_act_scales(model: nn.Module, layer_names: List[str],
                   calibration_data: torch.Tensor,
                   batch_size: int = 1) -> Dict[str, torch.Tensor]:
    """Compute per-channel max(|X|) activation scales for the given layers.

    Args:
        model: HuggingFace LLM (must accept input_ids of shape (B, S))
        layer_names: list of Linear layer names (e.g. ["model.layers.0.self_attn.q_proj", ...])
        calibration_data: (n_samples, seq_len) tensor of input_ids
        batch_size: forward batch size

    Returns:
        Dict mapping layer_name -> (in_features,) tensor of max(|X|).
    """
    collector = ActivationScaleCollector(layer_names)
    collector.attach(model)
    n = calibration_data.shape[0]
    print(f"[smooth] Collecting activation scales from {n} calibration samples...")
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = calibration_data[i : i + batch_size].to(next(model.parameters()).device)
            model(batch)
    scales = collector.get_scales()
    collector.detach()
    print(f"[smooth] Activation scales collected for {len(scales)} layers")
    return scales
