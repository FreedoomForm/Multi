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


# ----------------------------------------------------------------------
# 3-level {16, 8, 4} decoupled permutation (SHMQ-Ultimate extension)
# ----------------------------------------------------------------------

def decoupled_permutation_3level(
    channel_sensitivity: torch.Tensor,
    permutation_metric: torch.Tensor,
    ratio_16: float,
    ratio_8: float,
    tile_16: int = 128,
    tile_8:  int = 128,
    tile_4:  int = 64,
    sort_descending_magnitude: bool = True,
) -> Tuple[torch.Tensor, Dict[int, int]]:
    """3-level decoupled permutation: partition into C16 / C8 / C4.

    Algorithm (extension of SHMQ Eq. 12 to 3 levels):
    1. Sort channels ASCENDING by sensitivity.
    2. Top K1 (HIGHEST sensitivity) channels → C16 (FP16, most protected).
    3. Next K2 channels → C8 (INT8).
    4. Remaining channels → C4 (INT4).
    5. Within each cluster, sort by magnitude (descending) to minimize
       group-wise variance — this is the "decoupled" part of SHMQ.
    6. Round cluster sizes to tensor-core tile boundaries (PolyQ ISA matching).

    Final permutation layout:
        [ C16 (sorted by mag) | C8 (sorted by mag) | C4 (sorted by mag) ]

    Args:
        channel_sensitivity: (cin,) Manhattan-norm sensitivity per channel.
        permutation_metric: (cin,) magnitude metric for within-cluster sort.
        ratio_16: fraction of channels in C16 (most sensitive).
        ratio_8:  fraction of channels in C8.
                   (ratio_4 = 1 - ratio_16 - ratio_8)
        tile_16, tile_8, tile_4: tensor-core tile sizes for ISA matching.
        sort_descending_magnitude: within-cluster sort direction.

    Returns:
        (perm_indices, cluster_sizes) where:
            perm_indices: (cin,) LongTensor — final permutation
            cluster_sizes: {16: k16, 8: k8, 4: k4}
    """
    cin = channel_sensitivity.numel()
    ratio_4 = 1.0 - ratio_16 - ratio_8
    assert ratio_4 >= 0.0, f"ratios must sum to <=1: r16={ratio_16} r8={ratio_8}"

    # Initial cluster sizes (floor; remainder goes to C4)
    k16_init = int(cin * ratio_16)
    k8_init  = int(cin * ratio_8)
    k4_init  = cin - k16_init - k8_init

    # ISA-aware rounding (PolyQ): round cluster sizes to tile boundaries.
    # We round DOWN first (safe), then distribute leftovers to C16 then C8.
    def _round_down(x: int, m: int) -> int:
        return (x // m) * m if m > 1 else x

    k16 = _round_down(k16_init, tile_16)
    k8  = _round_down(k8_init,  tile_8)
    k4  = _round_down(k4_init,  tile_4)

    leftover = cin - (k16 + k8 + k4)
    # Distribute leftovers: fill C16 first (if it already has a non-zero size),
    # then C8, then dump remaining into C4 (accept partial tile at C4 tail).
    if leftover > 0 and k16 > 0:
        needed = min(tile_16 - (k16 % tile_16) if k16 % tile_16 else 0, leftover)
        k16 += needed
        leftover -= needed
    if leftover > 0 and k8 > 0:
        needed = min(tile_8 - (k8 % tile_8) if k8 % tile_8 else 0, leftover)
        k8 += needed
        leftover -= needed
    if leftover > 0:
        # Dump into C4 — accept a partial tile
        k4 += leftover
        leftover = 0
    assert k16 + k8 + k4 == cin, \
        f"Cluster sizes {k16}+{k8}+{k4} != {cin} (leftover={leftover})"

    # ----- Step 1: Identification (sort ASCENDING by sensitivity) -----
    # Top K1 (highest sens) → C16
    # Next K2 → C8
    # Rest → C4
    # We use topk for the C16 and C8 selections.
    if k16 > 0:
        c16_indices = torch.topk(channel_sensitivity, k16, largest=True).indices
    else:
        c16_indices = torch.empty(0, dtype=torch.long, device=channel_sensitivity.device)

    # For C8, we need to exclude C16 channels
    if k8 > 0:
        # Build a mask of remaining channels
        mask = torch.ones(cin, dtype=torch.bool, device=channel_sensitivity.device)
        if k16 > 0:
            mask[c16_indices] = False
        remaining_idx = torch.where(mask)[0]
        remaining_sens = channel_sensitivity[remaining_idx]
        # Top k8 from remaining
        top_k8_local = torch.topk(remaining_sens, k8, largest=True).indices
        c8_indices = remaining_idx[top_k8_local]
    else:
        c8_indices = torch.empty(0, dtype=torch.long, device=channel_sensitivity.device)

    # C4 = everything else
    mask_c4 = torch.ones(cin, dtype=torch.bool, device=channel_sensitivity.device)
    if k16 > 0:
        mask_c4[c16_indices] = False
    if k8 > 0:
        mask_c4[c8_indices] = False
    c4_indices = torch.where(mask_c4)[0]

    # ----- Step 2: Permutation (sort by magnitude within each cluster) -----
    def _sort_by_magnitude(indices: torch.Tensor) -> torch.Tensor:
        if indices.numel() == 0:
            return indices
        m = permutation_metric[indices]
        order = torch.argsort(m, descending=sort_descending_magnitude)
        return indices[order]

    c16_sorted = _sort_by_magnitude(c16_indices)
    c8_sorted  = _sort_by_magnitude(c8_indices)
    c4_sorted  = _sort_by_magnitude(c4_indices)

    # ----- Step 3: Final permutation -----
    final_indices = torch.cat([c16_sorted, c8_sorted, c4_sorted])
    assert final_indices.numel() == cin, \
        f"Permutation has {final_indices.numel()} indices, expected {cin}"

    cluster_sizes = {16: k16, 8: k8, 4: k4}
    return final_indices, cluster_sizes


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


# ----------------------------------------------------------------------
# 3-level parallel-aware permutation
# ----------------------------------------------------------------------

def apply_permutation_to_parallel_layers_3level(
    model: nn.Module,
    parallel_groups: Dict[str, List[str]],
    channel_sensitivities: Dict[str, torch.Tensor],
    permutation_metrics: Dict[str, torch.Tensor],
    bit_allocation: Dict[str, int],
    intra_layer_hp_ratio_16: float = 0.05,
    intra_layer_hp_ratio_8: float = 0.20,
    tile_16: int = 128,
    tile_8:  int = 128,
    tile_4:  int = 64,
    all_layer_names: Optional[List[str]] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Dict[int, int]]]:
    """Apply 3-level decoupled permutation to all layers, respecting parallel constraints.

    For each layer:
        * bit_allocation == 16 → all channels in C16 (FP16), no permutation needed
          (but we still return identity permutation + cluster_sizes for uniformity)
        * bit_allocation == 8  → all channels in C8
        * bit_allocation == 4  → split into C16/C8/C4 using intra-layer ratios

    For parallel groups (q/k/v; up/gate), all layers share:
        * The same channel_sensitivity (max across the group)
        * The same permutation_metric (max across the group)
        * The same bit allocation (enforced by ILP)
        * The same final permutation

    Args:
        model: HuggingFace LLM.
        parallel_groups: {group_key: [layer_names]}.
        channel_sensitivities: {layer_name: (cin,) tensor}.
        permutation_metrics: {layer_name: (cin,) tensor}.
        bit_allocation: {layer_name: 4 | 8 | 16} from ILP.
        intra_layer_hp_ratio_16, _8: cluster ratios for 4-bit layers.
        tile_16, tile_8, tile_4: tensor-core tile sizes for ISA matching.
        all_layer_names: optional list of all layer names (including non-parallel).
            If None, inferred from channel_sensitivities keys.

    Returns:
        (perm_indices, cluster_sizes) where:
            perm_indices: {layer_name: (cin,) LongTensor}
            cluster_sizes: {layer_name: {16: k16, 8: k8, 4: k4}}
    """
    all_perm_indices: Dict[str, torch.Tensor] = {}
    all_cluster_sizes: Dict[str, Dict[int, int]] = {}

    # Determine which layers are in parallel groups
    parallel_layer_set = set()
    for layer_names in parallel_groups.values():
        parallel_layer_set.update(layer_names)

    # Helper: compute permutation for a single "logical layer" (sensitivity + metric + bits)
    def _compute_perm(sens: torch.Tensor, metric: torch.Tensor, bits: int):
        cin = sens.numel()
        if bits == 16:
            # All channels in C16 — identity permutation is fine (but we still
            # want the magnitude sort within C16 to optimize cuBLAS layout).
            perm, cs = decoupled_permutation_3level(
                channel_sensitivity=sens,
                permutation_metric=metric,
                ratio_16=1.0, ratio_8=0.0,
                tile_16=tile_16, tile_8=tile_8, tile_4=tile_4,
            )
        elif bits == 8:
            perm, cs = decoupled_permutation_3level(
                channel_sensitivity=sens,
                permutation_metric=metric,
                ratio_16=0.0, ratio_8=1.0,
                tile_16=tile_16, tile_8=tile_8, tile_4=tile_4,
            )
        else:
            # 4-bit layer: split into C16/C8/C4
            perm, cs = decoupled_permutation_3level(
                channel_sensitivity=sens,
                permutation_metric=metric,
                ratio_16=intra_layer_hp_ratio_16,
                ratio_8=intra_layer_hp_ratio_8,
                tile_16=tile_16, tile_8=tile_8, tile_4=tile_4,
            )
        return perm, cs

    # Process parallel groups
    for group_key, layer_names in parallel_groups.items():
        layer_names = [n for n in layer_names if n in channel_sensitivities]
        if not layer_names:
            continue

        # All layers in the group should share the same sensitivity (from concat+Manhattan)
        sens_stack = torch.stack([channel_sensitivities[n] for n in layer_names])
        shared_sens = sens_stack.max(dim=0).values
        metric_stack = torch.stack([permutation_metrics[n] for n in layer_names])
        shared_metric = metric_stack.max(dim=0).values

        # All layers in group share the same bit allocation (enforced by ILP)
        bits_set = set(bit_allocation.get(n, 4) for n in layer_names)
        if len(bits_set) != 1:
            print(f"[perm3L] WARNING: parallel group {group_key} has mixed bits {bits_set}; "
                  f"using max")
        bits = max(bits_set)  # conservative: use highest precision

        perm, cs = _compute_perm(shared_sens, shared_metric, bits)

        # Apply the same permutation to all layers in the group
        for name in layer_names:
            mod = get_module_by_name(model, name)
            apply_permutation_to_layer(mod, perm, in_place=True)
            all_perm_indices[name] = perm.clone()
            all_cluster_sizes[name] = cs

    # Process non-parallel layers
    if all_layer_names is None:
        all_layer_names = list(channel_sensitivities.keys())
    for name in all_layer_names:
        if name in all_perm_indices:
            continue
        if name not in channel_sensitivities:
            continue
        sens = channel_sensitivities[name]
        metric = permutation_metrics.get(name, torch.zeros_like(sens))
        bits = bit_allocation.get(name, 4)
        perm, cs = _compute_perm(sens, metric, bits)
        mod = get_module_by_name(model, name)
        apply_permutation_to_layer(mod, perm, in_place=True)
        all_perm_indices[name] = perm.clone()
        all_cluster_sizes[name] = cs

    return all_perm_indices, all_cluster_sizes
