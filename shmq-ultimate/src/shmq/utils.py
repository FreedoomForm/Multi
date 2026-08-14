"""Common utilities for SHMQ-Ultimate."""
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Iterable, Tuple


def get_device_of(model: nn.Module) -> torch.device:
    """Get the device of the first parameter in the model."""
    for p in model.parameters():
        return p.device
    return torch.device("cpu")


def move_to_device(obj, device):
    """Recursively move tensors in a (possibly nested) structure to device."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(move_to_device(v, device) for v in obj)
    return obj


def get_module_by_name(model: nn.Module, name: str) -> nn.Module:
    """Get a submodule by its dotted name (e.g. 'model.layers.0.self_attn.q_proj')."""
    parts = name.split(".")
    mod = model
    for p in parts:
        mod = getattr(mod, p)
    return mod


def set_module_by_name(model: nn.Module, name: str, new_module: nn.Module):
    """Replace a submodule by its dotted name."""
    parts = name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


def get_parent_module_and_attr(model: nn.Module, name: str) -> Tuple[nn.Module, str]:
    """Return (parent_module, attr_name) such that getattr(parent, attr) == named module.

    Returns (None, name) if any intermediate module is missing.
    """
    parts = name.split(".")
    parent = model
    for p in parts[:-1]:
        if not hasattr(parent, p):
            return None, name
        parent = getattr(parent, p)
    if not hasattr(parent, parts[-1]):
        return None, name
    return parent, parts[-1]


def symmetric_quantize_weights(weight: torch.Tensor, n_bits: int = 4,
                                group_size: int = 128,
                                ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-group symmetric weight quantization.

    Args:
        weight: (out_features, in_features) tensor
        n_bits: 4 or 8
        group_size: 128 (typical)

    Returns:
        (qweight, scales) where:
          qweight: int8 tensor of same shape (dequantized values, for fake-quant)
          scales: (out_features, in_features // group_size) tensor
    """
    out_features, in_features = weight.shape
    assert in_features % group_size == 0, \
        f"in_features ({in_features}) must be divisible by group_size ({group_size})"
    n_groups = in_features // group_size

    max_q = 2 ** (n_bits - 1) - 1  # 7 for 4-bit, 127 for 8-bit

    # Reshape into (out_features, n_groups, group_size)
    w = weight.reshape(out_features, n_groups, group_size)
    # Per-group max abs
    max_abs = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = max_abs / max_q
    # Quantize
    q = (w / scale).round().clamp(-max_q, max_q)
    # Dequantize (fake quant)
    qweight = (q * scale).reshape(out_features, in_features)
    scales = scale.squeeze(-1)  # (out_features, n_groups)
    return qweight, scales


def symmetric_quantize_activations(activation: torch.Tensor, n_bits: int = 8,
                                    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-token symmetric activation quantization.

    Args:
        activation: (batch, seq_len, in_features) tensor
        n_bits: 8 (typical for activations)

    Returns:
        (qact, scales) where:
          qact: dequantized activation
          scales: (batch, seq_len, 1)
    """
    max_q = 2 ** (n_bits - 1) - 1
    max_abs = activation.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = max_abs / max_q
    q = (activation / scale).round().clamp(-max_q, max_q)
    qact = q * scale
    return qact, scale


def compute_quant_error(weight: torch.Tensor, n_bits: int = 4,
                        group_size: int = 128) -> float:
    """Compute ||W - Q(W)||^2 (sum squared quantization error).

    For n_bits >= 16, returns 0.0 (FP16/BF16 is considered lossless
    relative to its own precision).
    """
    if n_bits >= 16:
        return 0.0
    qweight, _ = symmetric_quantize_weights(weight, n_bits, group_size)
    return float(((weight - qweight) ** 2).sum().item())


def compute_quant_error_per_row(weight: torch.Tensor, n_bits: int = 4,
                                group_size: int = 128) -> torch.Tensor:
    """Compute per-output-channel ||w_i - Q(w_i)||^2.

    Returns: (out_features,) tensor
    """
    qweight, _ = symmetric_quantize_weights(weight, n_bits, group_size)
    return ((weight - qweight) ** 2).sum(dim=-1)


def topk_indices(values: torch.Tensor, k: int) -> torch.Tensor:
    """Return indices of the top-k largest values (descending)."""
    k = min(k, values.numel())
    return torch.topk(values, k, largest=True).indices


def bottomk_indices(values: torch.Tensor, k: int) -> torch.Tensor:
    """Return indices of the bottom-k smallest values (ascending)."""
    k = min(k, values.numel())
    return torch.topk(values, k, largest=False).indices
