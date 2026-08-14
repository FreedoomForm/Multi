"""SHMQ Decoupled Permutation (paper §3.2.3, Eq. 12, Fig. 3-4).

Decoupled permutation has THREE independent steps:
  1. Sort channels ASCENDING by sensitivity → partition into Csen (top K) | Cinsen
  2. Within Csen, sort by magnitude (reduce variance, lower quantization error)
  3. Within Cinsen, sort by magnitude

Final permutation = concat(Csen_sorted_by_mag, Cinsen_sorted_by_mag)

The K input channels are then reordered so that:
  - First K channels = Csen  (high-sensitivity, will be quantized at higher precision
                               in SHMQ native design — but in our v2, MixLLM still
                               decides N-axis bit allocation; this K-perm is purely
                               for RMSNorm fusion compatibility)
  - Last (cin - K) channels = Cinsen

Critical: This module permutes the K-AXIS (input channels), which is ORTHOGONAL
to MixLLM's N-axis (output channel) bit allocation. MixLLM kernel walks K in
groups of 128 and is agnostic to channel ordering, so applying a K-axis
permutation is purely a metadata/layout operation that doesn't affect the kernel
contract.

References:
  - SHMQ paper Eq. 12: Csen = I(S_IntraMQ, K), K = ⌊c_in · U_l⌉
  - SHMQ paper Fig. 3: decoupled permutation visualization
  - SHMQ paper Fig. 4: variance reduction via magnitude sort
"""
from __future__ import annotations
import torch
from typing import Tuple, Optional


def decoupled_permutation(
    sensitivity: torch.Tensor,   # shape [cin], per-channel sensitivity (Manhattan norm)
    hp_ratio: float,             # U_l: fraction of channels marked sensitive
    group_size: int = 128,       # round K to multiple of group_size for tensor cores
    magnitude: Optional[torch.Tensor] = None,  # shape [cin], per-channel weight magnitude
) -> torch.Tensor:
    """Compute the SHMQ decoupled permutation index.

    Args:
        sensitivity: 1D tensor of per-channel sensitivities (higher = more sensitive).
                     Computed from intra-layer Hessian via Manhattan norm
                     (S_IntraMQ_j = Σ_i |S^l_{i,j}|, SHMQ Eq. 11).
        hp_ratio: fraction of channels in the high-precision (sensitive) group.
                  U_l in SHMQ Eq. 8. Default 0.10 (matches MixLLM 10% INT8 ratio).
        group_size: round K (number of sensitive channels) to a multiple of this for
                    tensor-core friendly access. Default 128 (MixLLM group size).
        magnitude: per-channel weight magnitude used for within-cluster sort.
                   If None, computed as `sensitivity.abs()` (fallback).

    Returns:
        perm: 1D int64 tensor of shape [cin], the permutation index such that
              `weight[:, perm]` reorders input channels in SHMQ decoupled order.

    Algorithm (paper §3.2.3, Fig. 3-4):
      Step 1: Identify Csen = top K channels by sensitivity
              K = round(cin * hp_ratio / group_size) * group_size
      Step 2: Within Csen, sort by magnitude ASCENDING (variance reduction)
      Step 3: Within Cinsen, sort by magnitude ASCENDING
      Final:  perm = [Csen_sorted, Cinsen_sorted]
    """
    cin = sensitivity.shape[0]
    assert sensitivity.dim() == 1, f"Expected 1D sensitivity, got {sensitivity.dim()}D"
    assert 0.0 <= hp_ratio <= 1.0, f"hp_ratio out of range: {hp_ratio}"

    if magnitude is None:
        magnitude = sensitivity.abs()

    # Step 1: K = round(cin * hp_ratio / group_size) * group_size
    #   - Ensures K is a multiple of group_size for tensor cores
    #   - Falls back to 0 or cin if hp_ratio is too small/large
    k_raw = int(cin * hp_ratio)
    k = (k_raw // group_size) * group_size
    k = max(0, min(cin, k))

    if k == 0 or k == cin:
        # Edge case: uniform precision, just sort by magnitude
        return torch.argsort(magnitude, stable=True).to(torch.int64)

    # Step 1 (cont.): identify top-K channels by sensitivity
    #   Higher sensitivity → in Csen (high-precision group)
    #   Lower sensitivity → in Cinsen (low-precision group)
    sorted_by_sens = torch.argsort(sensitivity, descending=True, stable=True)
    csen_indices = sorted_by_sens[:k]      # sensitive (high-precision)
    cinsen_indices = sorted_by_sens[k:]    # insensitive (low-precision)

    # Step 2: within Csen, sort by magnitude ASCENDING
    #   Paper rationale (Fig. 4): grouping similar-magnitude channels reduces
    #   per-group variance → tighter quantization scale → lower quantization error.
    csen_mag = magnitude[csen_indices]
    csen_sort_idx = torch.argsort(csen_mag, stable=True)
    csen_sorted = csen_indices[csen_sort_idx]

    # Step 3: within Cinsen, sort by magnitude ASCENDING
    cinsen_mag = magnitude[cinsen_indices]
    cinsen_sort_idx = torch.argsort(cinsen_mag, stable=True)
    cinsen_sorted = cinsen_indices[cinsen_sort_idx]

    # Final: concat
    perm = torch.cat([csen_sorted, cinsen_sorted]).to(torch.int64)
    assert perm.shape[0] == cin
    assert len(torch.unique(perm)) == cin, "Permutation has duplicates!"

    return perm


def apply_k_permutation_to_weight(
    weight: torch.Tensor,   # shape [out_features, in_features]
    perm: torch.Tensor,     # shape [in_features], int64
) -> torch.Tensor:
    """Apply K-axis (input channel) permutation to a weight tensor.

    Returns weight[:, perm] — a NEW tensor with input channels reordered.
    The output channel axis (N) is untouched.

    This is the metadata-free permutation: we just gather columns.
    The MixLLM kernel is agnostic to K-axis ordering (it walks K in
    groups of 128 and accumulates), so this operation is safe.
    """
    assert weight.dim() == 2, f"Expected 2D weight, got {weight.dim()}D"
    assert perm.shape[0] == weight.shape[1], \
        f"perm length {perm.shape[0]} != in_features {weight.shape[1]}"
    return weight[:, perm.to(torch.int64).to(weight.device)].contiguous()


def apply_k_permutation_to_activation(
    activation: torch.Tensor,   # shape [..., in_features]
    perm: torch.Tensor,         # shape [in_features], int64
) -> torch.Tensor:
    """Apply K-axis permutation to an activation tensor.

    Returns activation[..., perm] — input channels reordered to match weight.
    """
    assert activation.shape[-1] == perm.shape[0], \
        f"activation last dim {activation.shape[-1]} != perm length {perm.shape[0]}"
    return activation[..., perm.to(torch.int64).to(activation.device)].contiguous()


def compute_permutation_metric(
    weight: torch.Tensor,   # shape [out_features, in_features]
    activation: torch.Tensor,  # shape [n_tokens, in_features]
) -> torch.Tensor:
    """Compute the SHMQ permutation metric (paper §3.2.3):

        metric_j = ||X[:, j]||_∞ × ||W[:, j]||_∞

    where j indexes input channels. This is the "permutation metric" used in
    SHMQ paper Fig. 4 to demonstrate variance reduction. Higher metric =
    channel that dominates quantization error → should be in Csen.

    Note: This is DIFFERENT from the per-element sensitivity S^l_{i,j}.
    The metric here is a simpler per-channel summary used for the final
    magnitude sort within each cluster.
    """
    # ||X[:, j]||_∞: max abs activation per input channel
    act_inf = activation.abs().amax(dim=0)  # [in_features]
    # ||W[:, j]||_∞: max abs weight per input channel
    wt_inf = weight.abs().amax(dim=0)       # [in_features]
    return act_inf * wt_inf                 # [in_features]
