"""SHMQ Parallel Layer Constraint (paper §3.2.4, Eq. 4).

Qwen2.5-7B-Instruct layer structure:
  - self_attn: q_proj, k_proj, v_proj (input from prior RMSNorm)
  - self_attn: o_proj (input from attention output, NOT shared)
  - mlp: gate_proj, up_proj (input from prior RMSNorm)
  - mlp: down_proj (input from SiLU(gate) * up, NOT shared)

Parallel groups (share the SAME K-axis permutation):
  Group A: {q_proj, k_proj, v_proj} — share RMSNorm input permutation
  Group B: {gate_proj, up_proj}     — share RMSNorm input permutation
  Standalone: {o_proj, down_proj}   — independent permutations

The constraint ensures that the K-axis permutation can be FUSED into the
prior RMSNorm: the RMSNorm only needs to permute once, and all three
attention projections (q/k/v) receive the same permuted input.

Implementation:
  1. Identify parallel groups via layer name suffix patterns
  2. Compute per-layer intra sensitivity (K-axis)
  3. Average sensitivities within each group → single per-group sensitivity
  4. Compute one permutation per group from the averaged sensitivity
  5. Assign the group's permutation to all member layers
"""
from __future__ import annotations
import torch
import re
from typing import Dict, List, Tuple, Optional


# === Parallel group identification ===

# Pattern: <layer_idx>.self_attn.{q_proj,k_proj,v_proj}  → Group A
# Pattern: <layer_idx>.mlp.{gate_proj,up_proj}            → Group B
# Pattern: <layer_idx>.self_attn.o_proj                   → Standalone
# Pattern: <layer_idx>.mlp.down_proj                      → Standalone

PARALLEL_PATTERNS = {
    # Group name → list of regex patterns matching the layer name suffix
    "attn_qkv": [r"\.self_attn\.q_proj$", r"\.self_attn\.k_proj$", r"\.self_attn\.v_proj$"],
    "mlp_gate_up": [r"\.mlp\.gate_proj$", r"\.mlp\.up_proj$"],
    # Standalone layers are auto-detected as everything else
}


def get_parallel_group(name: str) -> str:
    """Return the parallel group name for a layer name.

    Returns:
        - "attn_qkv" for q/k/v projections (share permutation)
        - "mlp_gate_up" for gate/up projections (share permutation)
        - "standalone_<name>" for o_proj, down_proj, etc. (independent permutation)

    The group name is unique per (layer_idx, group_type), so q/k/v in layer 5
    all return "attn_qkv" (matched by the same regex set, but the layer_idx
    distinguishes different transformer blocks).

    Actually — to be precise, the group key must include the layer index so
    that q/k/v in layer 5 share a permutation but q/k/v in layer 6 have a
    different one. We extract the layer index from the name.
    """
    # Extract layer index from name like "model.layers.5.self_attn.q_proj"
    m = re.search(r"layers\.(\d+)\.", name)
    layer_idx = m.group(1) if m else "0"

    for group_name, patterns in PARALLEL_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, name):
                return f"{layer_idx}_{group_name}"
    return f"{layer_idx}_standalone_{name.split('.')[-1]}"


def group_linears_by_parallel_constraint(
    linear_names: List[str],
) -> Dict[str, List[str]]:
    """Group linear layer names by parallel constraint.

    Returns a dict: group_key → list of layer names in that group.
    Groups with multiple members share a K-axis permutation.
    Groups with one member ("standalone_*") have their own permutation.
    """
    groups: Dict[str, List[str]] = {}
    for name in linear_names:
        group_key = get_parallel_group(name)
        groups.setdefault(group_key, []).append(name)
    return groups


def compute_group_sensitivity(
    group_name: str,
    layer_sensitivities: Dict[str, torch.Tensor],  # name → [cin] sensitivity
    group_members: List[str],
) -> torch.Tensor:
    """Compute the averaged sensitivity for a parallel group.

    For parallel groups (multiple members), average the per-channel sensitivities.
    For standalone groups (single member), just return that layer's sensitivity.

    Per SHMQ paper §3.2.4: "the sensitivities of parallel layers are averaged
    before computing the permutation, ensuring all layers agree on channel ordering."
    """
    member_sensitivities = [layer_sensitivities[name] for name in group_members]
    if len(member_sensitivities) == 1:
        return member_sensitivities[0]
    # All members must have the same cin (input dim) for averaging
    cin = member_sensitivities[0].shape[0]
    for s in member_sensitivities:
        assert s.shape[0] == cin, \
            f"Group {group_name} has mismatched cin: {s.shape[0]} vs {cin}"
    stacked = torch.stack(member_sensitivities, dim=0)
    return stacked.mean(dim=0)


def assign_group_permutation(
    group_members: List[str],
    group_perm: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Assign the same permutation to all members of a parallel group."""
    return {name: group_perm.clone() for name in group_members}
