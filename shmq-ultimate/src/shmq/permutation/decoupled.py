"""Decoupled permutation for SHMQ-Ultimate (SHMQ Section 3.2.3, Eq. 12, Fig. 4).

Algorithm:
1. Identification (sort by sensitivity, partition):
   - Sort channels ASCENDING by sensitivity
   - Top K sensitive → Csen
   - Rest → Cinsen
   (K = ⌊cin * U_l⌉ where U_l is the high-precision ratio from ILP)

2. Permutation (sort by magnitude within each cluster):
   - Within Csen: sort by magnitude (descending) → minimize group-wise variance
   - Within Cinsen: sort by magnitude (descending) → minimize group-wise variance

3. Final order:
   final_indices = concat(Csen_sorted, Cinsen_sorted)

4. Apply the same permutation to parallel layers (q/k/v; up/gate)

5. Apply the permutation to both weights (along cin axis) AND input activations
   (which will be fused into the preceding RMSNorm — see rmsnorm_fusion.py)

Reference: SHMQ paper Section 3.2.3, Figure 4 (decoupled vs coupled).
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from ..utils import get_module_by_name, set_module_by_name


def decoupled_permutation(
    channel_sensitivity: torch.Tensor,
    permutation_metric: torch.Tensor,
    high_precision_ratio: float,
    group_size: int = 128,
    sort_descending_magnitude: bool = True,
) -> torch.Tensor:
    """Compute the decoupled permutation indices for a layer's input channels.

    Args:
        channel_sensitivity: (cin,) Manhattan-norm channel sensitivity (from OBS)
        permutation_metric: (cin,) M_j = max|X_j| × max|W_j| (for magnitude sort)
        high_precision_ratio: U_l — fraction of channels to designate as sensitive (INT8)
        group_size: must be multiple of this (128 for tensor cores). The final
                    permutation is rounded so each block of `group_size` channels
                    is internally reordered, but the overall partition into Csen /
                    Cinsen still respects K = ⌊cin * U_l⌉.
        sort_descending_magnitude: if True, sort by magnitude descending within
                                    each cluster (paper default).

    Returns:
        (cin,) LongTensor of permutation indices.
        Applying these to weight columns and input activations gives the permuted layer.
    """
    cin = channel_sensitivity.numel()
    K = int(round(cin * high_precision_ratio))
    K = max(0, min(K, cin))

    # ----- Step 1: Identification -----
    # Sort channels ASCENDING by sensitivity
    sens_sorted_indices = torch.argsort(channel_sensitivity, descending=False)
    # Top K sensitive = last K (highest sensitivity)
    # Csen = top K by sensitivity (descending)
    sen_indices = torch.topk(channel_sensitivity, K, largest=True).indices
    # Cinsen = the rest
    mask = torch.ones(cin, dtype=torch.bool, device=channel_sensitivity.device)
    mask[sen_indices] = False
    insen_indices = torch.where(mask)[0]

    # ----- Step 2: Permutation (sort by magnitude within each cluster) -----
    def _sort_by_magnitude(indices: torch.Tensor) -> torch.Tensor:
        if indices.numel() == 0:
            return indices
        m = permutation_metric[indices]
        order = torch.argsort(m, descending=sort_descending_magnitude)
        return indices[order]

    sen_sorted = _sort_by_magnitude(sen_indices)
    insen_sorted = _sort_by_magnitude(insen_indices)

    # ----- Step 3: Final permutation -----
    final_indices = torch.cat([sen_sorted, insen_sorted])

    # ----- Step 4: Group-size alignment (optional) -----
    # If group_size > 1, we ensure that the Csen / Cinsen boundary falls on a
    # group boundary. This is needed so each group of 128 channels is entirely
    # INT4 or INT8 (no mixing within a group).
    if group_size > 1 and K > 0 and K < cin:
        # Round K up to nearest group_size multiple
        K_aligned = ((K + group_size - 1) // group_size) * group_size
        K_aligned = min(K_aligned, cin)
        if K_aligned != K:
            # Re-partition
            sen_indices = torch.topk(channel_sensitivity, K_aligned, largest=True).indices
            mask = torch.ones(cin, dtype=torch.bool, device=channel_sensitivity.device)
            mask[sen_indices] = False
            insen_indices = torch.where(mask)[0]
            sen_sorted = _sort_by_magnitude(sen_indices)
            insen_sorted = _sort_by_magnitude(insen_indices)
            final_indices = torch.cat([sen_sorted, insen_sorted])

    return final_indices


def apply_permutation_to_layer(
    layer: nn.Linear,
    perm_indices: torch.Tensor,
    in_place: bool = True,
) -> nn.Linear:
    """Apply the permutation to a Linear layer's weight (along cin axis).

    Weight W has shape (cout, cin). We permute the cin axis:
        W'[:, j] = W[:, perm_indices[j]]

    Args:
        layer: nn.Linear module
        perm_indices: (cin,) LongTensor
        in_place: if True, modify layer.weight.data in place; else return a copy

    Returns:
        The (possibly modified) layer.
    """
    W = layer.weight.data
    assert W.shape[1] == perm_indices.numel(), \
        f"cin mismatch: weight has {W.shape[1]}, perm has {perm_indices.numel()}"
    perm_indices = perm_indices.to(W.device)
    if in_place:
        layer.weight.data = W[:, perm_indices].clone()
        return layer
    else:
        new_layer = nn.Linear(
            W.shape[1], W.shape[0], bias=layer.bias is not None,
            device=W.device, dtype=W.dtype,
        )
        new_layer.weight.data = W[:, perm_indices].clone()
        if layer.bias is not None:
            new_layer.bias.data = layer.bias.data.clone()
        return new_layer


def apply_permutation_to_parallel_layers(
    model: nn.Module,
    parallel_groups: Dict[str, List[str]],
    channel_sensitivities: Dict[str, torch.Tensor],
    permutation_metrics: Dict[str, torch.Tensor],
    bit_allocation: Dict[str, int],
    group_size: int = 128,
) -> Dict[str, torch.Tensor]:
    """Apply decoupled permutation to all layers, respecting parallel constraints.

    For each parallel group (q/k/v or up/gate):
    - All layers share the SAME channel_sensitivity (computed via Manhattan norm
      on concatenated per-element sensitivities — see sensitivity/parallel.py)
    - All layers share the SAME permutation_metric (max across the group)
    - All layers get the SAME high_precision_ratio (from ILP — q/k/v share bits)
    - The same permutation indices are applied to all layers in the group

    Args:
        model: HuggingFace LLM
        parallel_groups: {group_key: [layer_names]}
        channel_sensitivities: {layer_name: (cin,) tensor}
        permutation_metrics: {layer_name: (cin,) tensor}
        bit_allocation: {layer_name: 4 or 8}  (from ILP)
        group_size: 128 (for tensor core alignment)

    Returns:
        {layer_name: (cin,) permutation indices}
    """
    all_perm_indices: Dict[str, torch.Tensor] = {}

    # Process parallel groups
    for group_key, layer_names in parallel_groups.items():
        layer_names = [n for n in layer_names if n in channel_sensitivities]
        if not layer_names:
            continue

        # All layers in the group should share the same sensitivity (from concat+Manhattan)
        # but if they don't, take the max
        sens_stack = torch.stack([channel_sensitivities[n] for n in layer_names])
        shared_sens = sens_stack.max(dim=0).values  # (cin,)

        # For metric: also take max across the group
        metric_stack = torch.stack([permutation_metrics[n] for n in layer_names])
        shared_metric = metric_stack.max(dim=0).values  # (cin,)

        # high_precision_ratio: same for all layers in group (from ILP)
        # If layers got 8-bit, ratio = 1.0; if 4-bit, ratio = 0.0
        # If somehow different (shouldn't happen due to parallel constraint), take max
        bits_set = set(bit_allocation.get(n, 4) for n in layer_names)
        if bits_set == {8}:
            hp_ratio = 1.0  # all 8-bit → all sensitive
        elif bits_set == {4}:
            hp_ratio = 0.0  # all 4-bit → none sensitive
        else:
            # Mixed (shouldn't happen with parallel constraint) — use 0.2 default
            print(f"[perm] WARNING: parallel group {group_key} has mixed bits {bits_set}; using 0.2 ratio")
            hp_ratio = 0.2

        perm = decoupled_permutation(
            channel_sensitivity=shared_sens,
            permutation_metric=shared_metric,
            high_precision_ratio=hp_ratio,
            group_size=group_size,
        )

        # Apply the same permutation to all layers in the group
        for name in layer_names:
            mod = get_module_by_name(model, name)
            apply_permutation_to_layer(mod, perm, in_place=True)
            all_perm_indices[name] = perm.clone()

    # Process non-parallel layers (o_proj, down_proj) individually
    for name, sens in channel_sensitivities.items():
        if name in all_perm_indices:
            continue
        metric = permutation_metrics.get(name, torch.zeros_like(sens))
        bits = bit_allocation.get(name, 4)
        hp_ratio = 1.0 if bits == 8 else 0.0
        perm = decoupled_permutation(
            channel_sensitivity=sens,
            permutation_metric=metric,
            high_precision_ratio=hp_ratio,
            group_size=group_size,
        )
        mod = get_module_by_name(model, name)
        apply_permutation_to_layer(mod, perm, in_place=True)
        all_perm_indices[name] = perm.clone()

    return all_perm_indices
