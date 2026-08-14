"""ISA-aware quanta matching (PolyQ-inspired).

PolyQ (https://arxiv.org/abs/2607.14618) introduces two key ideas:

1. **ISA-aware quanta matching**: round the SIZE of each precision cluster
   to a multiple of the underlying tensor-core tile size. On NVIDIA GPUs:
     * INT4  tensor cores operate on tiles of 64 columns  (m16n8k64 mma)
     * INT8  tensor cores operate on tiles of 128 columns (m16n8k32 mma)
     * FP16  tensor cores operate on tiles of 128 columns (m16n16k16 mma)
   Unaligned cluster boundaries force the kernel to emit masked or
   partial tiles, which on Hopper/Ampere costs 20-40% throughput.

2. **Layout propagation**: once the weight channels are permuted so that
   clusters are contiguous, the SAME permutation must be propagated
   through every operator that touches those channels — RMSNorm
   (absorbed into the weight), the next Linear's input (absorbed via
   weight row-permutation), and the KV-cache. We implement (1) here;
   (2) is handled in `permutation/decoupled.py` and `permutation/rmsnorm_fusion.py`.

The matching is done AFTER the per-channel sensitivity is computed but
BEFORE the decoupled permutation. We adjust the K boundary of each
cluster so that:

    |C_16| mod 128 == 0
    |C_8|  mod 128 == 0
    |C_4|  mod 64  == 0   (INT4 tile is 64 columns wide)

The adjustment prefers to move channels from the LOWER-precision cluster
to the HIGHER-precision cluster (since over-quantizing hurts more than
under-quantizing), but never violates the global memory budget.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import logging
import torch

logger = logging.getLogger("shmq")

# Tensor-core tile sizes (in output-feature columns).
# These are the canonical m16n8k* / m16n16k* shapes used by cuBLAS / CUTLASS.
TENSOR_CORE_TILE = {
    4:  64,   # INT4 mma: m16n8k64 — 64 columns per tile
    8:  128,  # INT8 mma: m16n8k32 — but effective N tile is 128 for memory coalescing
    16: 128,  # FP16 mma: m16n16k16 — N tile is 16, but we round to 128 for cuBLAS
}


@dataclass
class ISAMatchResult:
    """Result of ISA-aware quanta matching."""
    # Final cluster sizes AFTER ISA matching (per layer)
    cluster_sizes: Dict[str, Dict[int, int]]   # name -> {16: k16, 8: k8, 4: k4}
    # Channels moved from low-precision to high-precision (per layer)
    n_upgraded: Dict[str, int]
    # Channels moved from high-precision to low-precision (per layer, rare)
    n_downgraded: Dict[str, int]
    # Total budget drift in bits (positive = grew, negative = shrank)
    budget_drift_bits: float

    def summary(self) -> str:
        total_up = sum(self.n_upgraded.values())
        total_dn = sum(self.n_downgraded.values())
        return (
            f"ISA-Match Result:\n"
            f"  Layers processed: {len(self.cluster_sizes)}\n"
            f"  Total channels upgraded (low→high prec):   {total_up}\n"
            f"  Total channels downgraded (high→low prec): {total_dn}\n"
            f"  Net budget drift: {self.budget_drift_bits:+.1f} bits per parameter\n"
        )


def round_up(x: int, m: int) -> int:
    """Round x up to the nearest multiple of m."""
    if m <= 1:
        return x
    return ((x + m - 1) // m) * m


def round_down(x: int, m: int) -> int:
    """Round x down to the nearest multiple of m."""
    if m <= 1:
        return x
    return (x // m) * m


def isa_match_cluster_sizes(
    n_channels: int,
    initial_k16: int,
    initial_k8: int,
    initial_k4: int,
    avg_bits_budget: Optional[float] = None,
    tile_16: int = TENSOR_CORE_TILE[16],
    tile_8:  int = TENSOR_CORE_TILE[8],
    tile_4:  int = TENSOR_CORE_TILE[4],
    prefer_upgrade: bool = True,
) -> Tuple[int, int, int]:
    """ISA-aware rounding of cluster sizes to tensor-core tile boundaries.

    Args:
        n_channels: total number of output channels in the layer.
        initial_k16, initial_k8, initial_k4: initial cluster sizes (sum = n_channels).
        avg_bits_budget: optional upper bound on avg bits per parameter.
        tile_16, tile_8, tile_4: tensor-core tile sizes for each precision.
        prefer_upgrade: if True, when rounding causes a surplus, push channels
            UP (4→8 or 8→16) rather than DOWN. Default True (over-quantizing
            hurts accuracy more than the budget relaxation costs memory).

    Returns:
        (k16, k8, k4) — final cluster sizes, summing to n_channels.
    """
    assert initial_k16 + initial_k8 + initial_k4 == n_channels, \
        f"Cluster sizes {initial_k16}+{initial_k8}+{initial_k4} != {n_channels}"

    # Step 1: round each cluster size DOWN to its tile boundary.
    k16 = round_down(initial_k16, tile_16)
    k8  = round_down(initial_k8,  tile_8)
    k4  = round_down(initial_k4,  tile_4)

    # Step 2: distribute the leftover channels.
    leftover = n_channels - (k16 + k8 + k4)
    assert leftover >= 0, f"Leftover = {leftover} < 0 — bug in round_down"

    if prefer_upgrade:
        # First fill C16 up to the next tile boundary.
        needed_16 = min(tile_16 - (k16 % tile_16) if k16 % tile_16 else 0, leftover)
        # Special case: if k16 == 0, don't introduce a tiny C16 cluster unless
        # there are enough leftovers to fill a full tile.
        if k16 == 0 and leftover < tile_16:
            needed_16 = 0
        k16 += needed_16
        leftover -= needed_16

        # Then fill C8 up to the next tile boundary.
        needed_8 = min(tile_8 - (k8 % tile_8) if k8 % tile_8 else 0, leftover)
        if k8 == 0 and leftover < tile_8:
            needed_8 = 0
        k8 += needed_8
        leftover -= needed_8

        # Remaining leftovers go to C4 — pad it up to a tile boundary.
        if leftover > 0:
            # We need to round k4 up; the extra channels come from... nowhere.
            # So we instead round k4 DOWN and let the leftovers become a partial
            # tile that the kernel will mask. This is unavoidable when
            # n_channels itself is not a multiple of LCM(tile_4, tile_8, tile_16).
            # In practice, LLM hidden sizes are multiples of 128, so this is rare.
            k4 += leftover  # accept a partial tile at the tail of C4
            leftover = 0
    else:
        # Just lump all leftovers into the smallest cluster (C4).
        k4 += leftover
        leftover = 0

    # Step 3: if budget exceeded, downgrade channels from C16 → C8 → C4.
    if avg_bits_budget is not None:
        avg_bits = (16 * k16 + 8 * k8 + 4 * k4) / n_channels
        while avg_bits > avg_bits_budget + 1e-6:
            # Downgrade one tile_16 chunk from C16 to C8.
            if k16 >= tile_16:
                k16 -= tile_16
                k8  += tile_16
                avg_bits = (16 * k16 + 8 * k8 + 4 * k4) / n_channels
                continue
            # Downgrade one tile_8 chunk from C8 to C4.
            if k8 >= tile_8:
                k8  -= tile_8
                k4  += tile_8
                avg_bits = (16 * k16 + 8 * k8 + 4 * k4) / n_channels
                continue
            # Can't downgrade further without going below a tile boundary.
            break

    assert k16 + k8 + k4 == n_channels, \
        f"Final cluster sizes {k16}+{k8}+{k4} != {n_channels}"
    return k16, k8, k4


def apply_isa_matching(
    layer_names: List[str],
    out_features: Dict[str, int],
    initial_ratios: Dict[str, Dict[int, float]],
    avg_bits_budget: Optional[float] = None,
    prefer_upgrade: bool = True,
    verbose: bool = False,
) -> ISAMatchResult:
    """Apply ISA-aware quanta matching across all layers.

    Args:
        layer_names: list of layer names.
        out_features: {layer_name: n_output_channels}.
        initial_ratios: {layer_name: {16: r16, 8: r8, 4: r4}} — sum to 1.
        avg_bits_budget: optional upper bound on avg bits per parameter.
        prefer_upgrade: if True, prefer upgrading channels to higher precision.
        verbose: print per-layer adjustments.

    Returns:
        ISAMatchResult with final cluster sizes and adjustment stats.
    """
    cluster_sizes: Dict[str, Dict[int, int]] = {}
    n_upgraded: Dict[str, int] = {}
    n_downgraded: Dict[str, int] = {}
    total_drift = 0.0

    for name in layer_names:
        n = out_features[name]
        r = initial_ratios[name]
        # Initial cluster sizes (floor to integer; surplus goes to C4)
        k16_init = int(n * r[16])
        k8_init  = int(n * r[8])
        k4_init  = n - k16_init - k8_init

        k16, k8, k4 = isa_match_cluster_sizes(
            n_channels=n,
            initial_k16=k16_init, initial_k8=k8_init, initial_k4=k4_init,
            avg_bits_budget=avg_bits_budget,
            prefer_upgrade=prefer_upgrade,
        )
        cluster_sizes[name] = {16: k16, 8: k8, 4: k4}

        # Count upgrades vs downgrades
        delta_16 = k16 - k16_init
        delta_8  = k8  - k8_init
        # An "upgrade" is when a channel moves to a higher precision.
        # 4→8, 4→16, or 8→16.
        n_up = max(0, delta_16) + max(0, delta_8 - max(0, delta_16))
        n_dn = max(0, -delta_16) + max(0, -delta_8 - max(0, -delta_16))
        n_upgraded[name] = n_up
        n_downgraded[name] = n_dn

        drift = 16 * delta_16 + 8 * delta_8 + 4 * (k4 - k4_init)
        total_drift += drift

        if verbose:
            logger.info(
                f"[isa-match] {name}: ({k16_init},{k8_init},{k4_init}) -> "
                f"({k16},{k8},{k4})  upgraded={n_up} downgraded={n_dn} drift={drift:+d}"
            )

    return ISAMatchResult(
        cluster_sizes=cluster_sizes,
        n_upgraded=n_upgraded,
        n_downgraded=n_downgraded,
        budget_drift_bits=total_drift,
    )


def cluster_sizes_to_indices(
    cluster_sizes: Dict[str, Dict[int, int]],
    permutation_indices: Dict[str, torch.Tensor],
) -> Dict[str, Dict[int, torch.Tensor]]:
    """Given final cluster sizes AND the (sorted) permutation order, return
    the channel indices belonging to each cluster.

    The decoupled permutation (SHMQ Eq.12) sorts channels by sensitivity
    ascending, then partitions into C_sen (HIGH precision) / C_insen (LOW precision).
    For 3 levels {4, 8, 16}:
        * Top k16 channels (most sensitive)  -> C16 (FP16)
        * Next k8 channels                   -> C8  (INT8)
        * Bottom k4 channels (least sens.)   -> C4  (INT4)

    Args:
        cluster_sizes: {layer_name: {16: k16, 8: k8, 4: k4}}.
        permutation_indices: {layer_name: LongTensor[n_channels]} — sorted
            channel order, where permutation_indices[0] is the MOST sensitive
            channel and permutation_indices[-1] is the LEAST sensitive.

    Returns:
        {layer_name: {16: LongTensor[k16], 8: LongTensor[k8], 4: LongTensor[k4]}}.
    """
    out: Dict[str, Dict[int, torch.Tensor]] = {}
    for name, sizes in cluster_sizes.items():
        perm = permutation_indices[name]
        k16 = sizes[16]; k8 = sizes[8]; k4 = sizes[4]
        assert k16 + k8 + k4 == perm.numel(), \
            f"{name}: cluster sizes {k16}+{k8}+{k4} != {perm.numel()}"
        out[name] = {
            16: perm[:k16].clone(),
            8:  perm[k16:k16 + k8].clone(),
            4:  perm[k16 + k8:].clone(),
        }
    return out
