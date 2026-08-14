"""Parallel layer constraint for SHMQ-Ultimate (SHMQ Appendix A.3.1).

The parallel layer constraint ensures that q/k/v proj (and up/gate proj) share:
1. The same inter-layer bit allocation (for permutation fusion compatibility).
2. The same intra-layer permutation indices (so the RMSNorm can be fused with
   one permutation for all parallel layers).

For inter-layer:
    average the sensitivities of parallel layers (q/k/v -> mean; up/gate -> mean)

For intra-layer:
    concatenate the per-element sensitivity matrices of parallel layers along
    the cout axis, then apply Manhattan norm — this gives a single per-channel
    sensitivity vector used for all parallel layers' permutation.

Reference: SHMQ paper Appendix A.3.1.
"""
from __future__ import annotations
from typing import Dict, List, Tuple
import torch


def average_inter_layer_parallel_sensitivity(
    sensitivities: Dict[str, float],
    parallel_groups: Dict[str, List[str]],
) -> Dict[str, float]:
    """Average sensitivities within each parallel group.

    For each parallel group (e.g. {"q_proj", "k_proj", "v_proj"}), replace each
    layer's sensitivity with the mean of the group. This ensures the ILP solver
    assigns the same bit-width to all layers in the group.

    Args:
        sensitivities: {layer_name: sensitivity_score}
        parallel_groups: {group_key: [layer_name1, layer_name2, ...]}

    Returns:
        Updated sensitivities dict with averaged values per parallel group.
    """
    out = dict(sensitivities)
    for group_key, layer_names in parallel_groups.items():
        values = [sensitivities[n] for n in layer_names if n in sensitivities]
        if not values:
            continue
        mean_val = sum(values) / len(values)
        for n in layer_names:
            if n in out:
                out[n] = mean_val
    return out


def concatenate_intra_layer_parallel_sensitivity(
    per_element_sensitivities: Dict[str, torch.Tensor],
    parallel_groups: Dict[str, List[str]],
) -> Dict[str, torch.Tensor]:
    """Concatenate per-element sensitivities of parallel layers, then Manhattan.

    For each parallel group, concatenate the (cout_l, cin) sensitivity matrices
    along the cout axis to get (Σcout_l, cin), then apply Manhattan norm over
    the new cout axis to get a single (cin,) channel sensitivity vector.

    This single vector is used for ALL parallel layers' permutation, ensuring
    they get the same reordering.

    Args:
        per_element_sensitivities: {layer_name: (cout, cin) tensor}
        parallel_groups: {group_key: [layer_name1, layer_name2, ...]}

    Returns:
        {layer_name: (cin,) channel sensitivity tensor} — same vector for all
        layers in a parallel group.
    """
    out: Dict[str, torch.Tensor] = {}
    for group_key, layer_names in parallel_groups.items():
        matrices = [per_element_sensitivities[n] for n in layer_names
                    if n in per_element_sensitivities]
        if not matrices:
            continue
        # Concatenate along cout axis
        concat = torch.cat(matrices, dim=0)  # (Σcout_l, cin)
        # Manhattan norm over the new cout axis
        channel_sens = concat.abs().sum(dim=0)  # (cin,)
        # Assign to all layers in the group
        for n in layer_names:
            if n in per_element_sensitivities:
                out[n] = channel_sens.clone()
    # Also handle non-parallel layers (no concat needed)
    for name, sens in per_element_sensitivities.items():
        if name not in out:
            out[name] = sens.abs().sum(dim=0)
    return out
