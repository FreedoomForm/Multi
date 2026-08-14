"""Decoupled channel permutation (SHMQ Eq. 12) extended to 3 precision
clusters {C16, C8, C4} with PolyQ-style quanta matching.

Channels are sorted by OBS channel sensitivity (descending), partitioned
into high/mid/low precision clusters whose sizes are rounded to hardware
quanta, and within each cluster re-sorted by a magnitude metric to improve
group-wise scale quality.

Only "fusable" parallel groups (those directly preceded by an RMSNorm:
attention q/k/v and MLP gate/up) receive a non-identity permutation, so the
permutation can be fused into the norm with zero runtime overhead.
o_proj / down_proj keep identity permutations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from .model_utils import ATTN_IN, MLP_IN, BlockInfo, layer_key, parallel_groups


@dataclass
class ChannelPartition:
    perm: torch.Tensor  # (cin,) long — new order of input channels
    n16: int
    n8: int
    n4: int

    @property
    def cin(self) -> int:
        return int(self.perm.numel())

    def is_identity(self) -> bool:
        return bool(torch.equal(self.perm, torch.arange(self.cin)))


def round_to_quanta(n: int, quanta: int, total: int) -> int:
    """Round n to the nearest multiple of quanta, clamped to [0, total]."""
    r = int(round(n / quanta)) * quanta
    return max(0, min(r, total))


def partition_sizes(
    cin: int,
    layer_bits: int,
    hp_base_ratio: float = 0.125,
    quanta_int8: int = 128,
    quanta_int4: int = 64,
) -> tuple:
    """(n16, n8, n4) for a layer given its ILP bit level.

    bits=16 : everything FP16.
    bits=8  : UB (12.5%) of channels kept FP16, rest INT8.
    bits=4  : UB kept INT8 (mid-precision buffer), rest INT4.
    """
    if layer_bits >= 16:
        return cin, 0, 0
    if layer_bits == 8:
        n16 = round_to_quanta(int(hp_base_ratio * cin), quanta_int8, cin)
        return n16, cin - n16, 0
    # layer_bits == 4
    n8 = round_to_quanta(int(hp_base_ratio * cin), quanta_int8, cin)
    n4 = cin - n8
    # make n4 a multiple of quanta_int4 by growing n8
    rem = n4 % quanta_int4
    if rem != 0:
        n8 += rem
        n4 -= rem
    return 0, n8, n4


def magnitude_metric(weight: torch.Tensor, act_sample: Optional[torch.Tensor]) -> torch.Tensor:
    """Per-input-channel magnitude metric: |w|_inf (* |x|_inf if acts given)."""
    wmax = weight.float().abs().amax(dim=0)  # (cin,)
    if act_sample is not None and act_sample.numel() > 0:
        amax = act_sample.float().abs().amax(dim=0).to(wmax.device)
        return wmax * amax
    return wmax


def decoupled_permutation(
    channel_sens: torch.Tensor,
    magnitude: torch.Tensor,
    n16: int,
    n8: int,
    n4: int,
) -> torch.Tensor:
    """SHMQ Eq. 12: sort by sensitivity desc -> partition into clusters ->
    within each cluster sort by magnitude desc.  Returns perm (cin,)."""
    order = torch.argsort(channel_sens, descending=True)
    perm_parts: List[torch.Tensor] = []
    off = 0
    for n in (n16, n8, n4):
        if n <= 0:
            continue
        cluster = order[off : off + n]
        mag = magnitude[cluster]
        sub = torch.argsort(mag, descending=True)
        perm_parts.append(cluster[sub])
        off += n
    return torch.cat(perm_parts) if perm_parts else torch.arange(channel_sens.numel())


def build_partitions(
    blocks: List[BlockInfo],
    channel_sens: Dict[str, torch.Tensor],
    magnitudes: Dict[str, torch.Tensor],
    bit_alloc: Dict[str, int],
    hp_base_ratio: float = 0.125,
    quanta_int8: int = 128,
    quanta_int4: int = 64,
    enable_permutation: bool = True,
) -> Dict[str, ChannelPartition]:
    """Build a ChannelPartition per layer.

    Parallel groups (q/k/v, gate/up) share one permutation computed from the
    SUM of their per-element channel sensitivities (SHMQ Eq. 4 CONCAT rule).
    Only fusable groups (preceded by RMSNorm) get non-identity permutations.
    """
    parts: Dict[str, ChannelPartition] = {}
    for blk in blocks:
        for grp in parallel_groups(blk):
            keys = [layer_key(blk.index, r) for r in grp]
            lin0 = blk.linears[grp[0]]
            cin = lin0.in_features
            bits = min(bit_alloc.get(k, 16) for k in keys)
            n16, n8, n4 = partition_sizes(cin, bits, hp_base_ratio, quanta_int8, quanta_int4)

            fusable = grp[0] in ATTN_IN or grp[0] in MLP_IN
            if enable_permutation and fusable and (n16 or n8) and n4 + n8 > 0 and bits < 16:
                sens = torch.zeros(cin)
                mag = torch.zeros(cin)
                for k, r in zip(keys, grp):
                    if k in channel_sens:
                        sens += channel_sens[k].float().cpu()
                    if k in magnitudes:
                        mag += magnitudes[k].float().cpu()
                perm = decoupled_permutation(sens, mag, n16, n8, n4)
            else:
                perm = torch.arange(cin)

            for k in keys:
                parts[k] = ChannelPartition(perm=perm.clone(), n16=n16, n8=n8, n4=n4)
    return parts


@torch.no_grad()
def apply_permutation_to_weights(
    blocks: List[BlockInfo], parts: Dict[str, ChannelPartition]
) -> None:
    """Permute input channels of each linear weight in place."""
    for blk in blocks:
        for role, lin in blk.linears.items():
            key = layer_key(blk.index, role)
            p = parts.get(key)
            if p is None or p.is_identity():
                continue
            lin.weight.data = lin.weight.data[:, p.perm].contiguous()


def permute_hessian(H: torch.Tensor, part: ChannelPartition) -> torch.Tensor:
    """Apply the same permutation to a layer Hessian H = XX^T."""
    if part.is_identity():
        return H
    p = part.perm
    return H[p][:, p].contiguous()
