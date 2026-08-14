"""SHMQ-Ultimate — Permutation subpackage.

Implements the CUSTOM SHMQ components (not available in any existing repo):
- Decoupled permutation (SHMQ Eq. 12, Section 3.2.3, Figure 4)
- Permutation metric: product of activations × weights l∞ norm
- PermutedRMSNorm: bake permutation into RMSNorm (zero overhead)

Decoupled permutation algorithm:
1. Sort channels ASCENDING by sensitivity
2. Partition into Csen (top K sensitive) and Cinsen (rest)
3. Within Csen: sort by magnitude → minimize group-wise variance
4. Within Cinsen: sort by magnitude → minimize group-wise variance
5. Final order = concat(Csen_sorted, Cinsen_sorted)
6. Apply same permutation to parallel layers (q/k/v; up/gate)
7. Block size multiple of group_size (128) for tensor cores
"""
from .metric import compute_permutation_metric, capture_input_activations
from .decoupled import (
    decoupled_permutation,
    decoupled_permutation_3level,
    apply_permutation_to_layer,
    apply_permutation_to_parallel_layers,
    apply_permutation_to_parallel_layers_3level,
)
from .rmsnorm_fusion import PermutedRMSNorm, fuse_permutation_into_rmsnorm

__all__ = [
    "compute_permutation_metric",
    "capture_input_activations",
    "decoupled_permutation",
    "decoupled_permutation_3level",
    "apply_permutation_to_layer",
    "apply_permutation_to_parallel_layers",
    "apply_permutation_to_parallel_layers_3level",
    "PermutedRMSNorm",
    "fuse_permutation_into_rmsnorm",
]
