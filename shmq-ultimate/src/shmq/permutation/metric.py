"""Permutation metric for SHMQ-Ultimate (SHMQ Appendix A.3.1).

Permutation metric = product of activations and weights l∞ norm:
    M_j = ||X[:, j]||_∞ × ||W[:, j]||_∞

This metric is used as an ALTERNATIVE to the sensitivity metric for sorting
channels during the "magnitude sort" step of the decoupled permutation.

Reference: SHMQ paper Appendix A.3.1: "we take the product of activations and
weights' l∞ as the permutation metric."
"""
from __future__ import annotations
from typing import Dict, List, Optional
import torch
from ..utils import get_module_by_name


def compute_permutation_metric(
    weight: torch.Tensor,
    input_activations: List[torch.Tensor],
) -> torch.Tensor:
    """Compute per-channel permutation metric M_j.

    Args:
        weight: (cout, cin) layer weight matrix
        input_activations: list of (N, cin) input activation tensors
                           (one per forward batch)

    Returns:
        (cin,) tensor of M_j = max|X_j| × max|W_j|
    """
    # max|W[:, j]| over output rows — per input channel
    w_max = weight.abs().amax(dim=0)  # (cin,)

    # max|X[:, j]| over all captured batches and samples
    if not input_activations:
        return w_max.clone()
    X = torch.cat([x.reshape(-1, x.shape[-1]) for x in input_activations], dim=0)  # (N_total, cin)
    x_max = X.abs().amax(dim=0)  # (cin,)

    # Product metric
    return (x_max * w_max)


def capture_input_activations(
    model, layer_names: List[str], calibration_data: torch.Tensor,
    batch_size: int = 1, max_samples: Optional[int] = None,
) -> Dict[str, List[torch.Tensor]]:
    """Capture input activations for the given layers via forward hooks.

    Returns: {layer_name: [list of (N, cin) tensors per batch]}
    """
    device = next(model.parameters()).device
    n_total = calibration_data.shape[0]
    if max_samples is not None:
        n_total = min(n_total, max_samples)

    captured: Dict[str, List[torch.Tensor]] = {n: [] for n in layer_names}
    handles = []

    def make_hook(name: str):
        def hook(module, inputs, outputs):
            x = inputs[0].detach()  # (B, S, cin)
            x = x.reshape(-1, x.shape[-1])  # (B*S, cin)
            captured[name].append(x.to(device))
        return hook

    for name in layer_names:
        mod = get_module_by_name(model, name)
        handles.append(mod.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        for i in range(0, n_total, batch_size):
            batch = calibration_data[i : i + batch_size].to(device)
            model(batch)

    for h in handles:
        h.remove()

    return captured
