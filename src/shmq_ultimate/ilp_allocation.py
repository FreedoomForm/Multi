"""ILP bit allocation (HAWQ-V3 style) with SHMQ parallel-layer constraint.

Chooses one bit-width per layer from bit_levels (default {16, 8, 4}) to
minimise total expected quantization perturbation Omega subject to a
model-size budget (average bits) and the SHMQ Eq. 4 parallel constraint
(q/k/v share bits; up/gate share bits).
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch

from .model_utils import BlockInfo, layer_key, parallel_groups
from .quant_utils import fake_quantize_group_sym


def quant_error_omega(
    blocks: List[BlockInfo],
    inter_sens: Dict[str, float],
    bit_levels: Sequence[int] = (16, 8, 4),
    group_size: int = 128,
    probe_bits: int = 4,
) -> Dict[Tuple[str, int], float]:
    """Omega_{l,b} = S_l * mse_b / mse_probe (0 for b >= 16)."""
    omega: Dict[Tuple[str, int], float] = {}
    for blk in blocks:
        for role, lin in blk.linears.items():
            key = layer_key(blk.index, role)
            s_l = inter_sens.get(key, 1.0)
            w = lin.weight.data.float()
            mse_probe = float(((w - fake_quantize_group_sym(w, probe_bits, group_size)) ** 2).mean())
            mse_probe = max(mse_probe, 1e-20)
            for b in bit_levels:
                if b >= 16:
                    omega[(key, b)] = 0.0
                else:
                    mse_b = float(((w - fake_quantize_group_sym(w, b, group_size)) ** 2).mean())
                    omega[(key, b)] = s_l * (mse_b / mse_probe)
    return omega


def ilp_bit_allocation(
    blocks: List[BlockInfo],
    inter_sens: Dict[str, float],
    bit_levels: Sequence[int] = (16, 8, 4),
    target_avg_bits: float = 4.8,
    group_size: int = 128,
    parallel_constraint: bool = True,
) -> Dict[str, int]:
    """Solve the HAWQ-V3 style ILP.  Returns {layer_key: bits}."""
    omega = quant_error_omega(blocks, inter_sens, bit_levels, group_size)

    keys: List[str] = []
    numel: Dict[str, int] = {}
    for blk in blocks:
        for role, lin in blk.linears.items():
            key = layer_key(blk.index, role)
            keys.append(key)
            numel[key] = lin.weight.numel()
    total = sum(numel.values())

    try:
        import pulp
    except ImportError:
        return {k: min(bit_levels) for k in keys}

    prob = pulp.LpProblem("shmq_bit_allocation", pulp.LpMinimize)
    x = {
        (k, b): pulp.LpVariable(f"x_{i}_{b}", cat="Binary")
        for i, k in enumerate(keys)
        for b in bit_levels
    }
    # Objective: total perturbation
    prob += pulp.lpSum(x[(k, b)] * omega[(k, b)] for k in keys for b in bit_levels)
    # One bit-width per layer
    for k in keys:
        prob += pulp.lpSum(x[(k, b)] for b in bit_levels) == 1
    # Size budget
    prob += (
        pulp.lpSum(x[(k, b)] * b * numel[k] for k in keys for b in bit_levels)
        <= target_avg_bits * total
    )
    # SHMQ Eq. 4: parallel layers share bit-width
    if parallel_constraint:
        for blk in blocks:
            for grp in parallel_groups(blk):
                if len(grp) < 2:
                    continue
                k0 = layer_key(blk.index, grp[0])
                for r in grp[1:]:
                    kr = layer_key(blk.index, r)
                    for b in bit_levels:
                        prob += x[(k0, b)] == x[(kr, b)]

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        return {k: min(bit_levels) for k in keys}

    alloc: Dict[str, int] = {}
    for k in keys:
        for b in bit_levels:
            if x[(k, b)].value() is not None and x[(k, b)].value() > 0.5:
                alloc[k] = b
                break
        else:
            alloc[k] = min(bit_levels)
    return alloc


def average_bits(alloc: Dict[str, int], blocks: List[BlockInfo]) -> float:
    tot_bits = 0.0
    tot = 0
    for blk in blocks:
        for role, lin in blk.linears.items():
            key = layer_key(blk.index, role)
            n = lin.weight.numel()
            tot_bits += alloc.get(key, 16) * n
            tot += n
    return tot_bits / max(tot, 1)
