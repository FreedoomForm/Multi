"""ILP bit allocation for SHMQ-Ultimate.

Adapted from HAWQ-V3 (https://github.com/Zhen-Dong/HAWQ/blob/main/ILP.ipynb).

Key changes for LLM:
- 2 bit levels {4, 8} (instead of CNN's per-layer continuous bits)
- Memory budget = W4.8A8 (target average bits = 4.8 = 0.8*4 + 0.2*8)
- Parallel-layer equality constraint (q/k/v same bits; up/gate same bits)
  — needed for permutation fusion compatibility (SHMQ Appendix A.3.1)
- Drop CNN-specific BOPS / latency constraints (LLM cost is KV-cache memory)
- Objective: minimize Σ (Hessian_trace_i × (||W-Q8||² - ||W-Q4||²) × x_i)
  where x_i ∈ {1, 2}, 1→4-bit, 2→8-bit

Reference: HAWQ-V3 ILP.ipynb (extracted in worklog Task 0.4).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Sequence
import numpy as np
import pulp


@dataclass
class ILPResult:
    """Result of ILP bit allocation."""
    bit_allocation: Dict[str, int]  # layer_name -> 4 or 8
    n_layers_4bit: int
    n_layers_8bit: int
    total_bits: float  # average bits per layer
    total_memory_mb: float  # estimated memory in MB
    solver_status: str
    objective_value: float

    def summary(self) -> str:
        return (
            f"ILP Result:\n"
            f"  Status: {self.solver_status}\n"
            f"  Layers @ 4-bit: {self.n_layers_4bit}\n"
            f"  Layers @ 8-bit: {self.n_layers_8bit}\n"
            f"  Average bits per layer: {self.total_bits:.3f}\n"
            f"  Estimated memory: {self.total_memory_mb:.2f} MB\n"
            f"  Objective value: {self.objective_value:.6f}\n"
        )


def solve_ilp_bit_allocation(
    layer_names: List[str],
    sensitivities: Dict[str, float],
    n_params: Dict[str, int],
    quant_error_4bit: Dict[str, float],
    quant_error_8bit: Dict[str, float],
    target_hp_ratio: float = 0.20,
    base_hp_ratio: float = 0.125,
    parallel_groups: Optional[Dict[str, List[str]]] = None,
    solver: str = "CBC",
    time_limit: int = 30,
    verbose: bool = False,
) -> ILPResult:
    """Solve the ILP bit allocation problem.

    Variables: x_i ∈ {1, 2}  (1 → 4-bit, 2 → 8-bit)
    Objective: minimize Σ_i (x_i - 1) * sensitivity_diff_i
        where sensitivity_diff_i = sensitivity_i * (||W - Q8||² - ||W - Q4||²)
              This is NEGATIVE (Q8 error < Q4 error), so the solver prefers 8-bit
              (drives objective down), but budget constraint caps how many can be 8-bit.
    Constraint 1 (memory budget):
        Σ_i (bytes_i if x_i=4 else bytes_i*2) / Σ_i bytes_i ≤ (1 - target_hp_ratio) + 2 * target_hp_ratio
        Equivalent: Σ_i params_i * (4 if x_i=1 else 8) / Σ_i params_i ≤ 4 + 4 * target_hp_ratio
        For W4.8A8 with target_hp_ratio=0.20: average bits = 4.8
    Constraint 2 (parallel layers — equality):
        For each parallel group {q, k, v}: x_q == x_k == x_v
        For each parallel group {up, gate}: x_up == x_gate

    Args:
        layer_names: list of layer names to allocate bits for
        sensitivities: {layer_name: Fisher sensitivity}
        n_params: {layer_name: number of parameters (cout * cin)}
        quant_error_4bit: {layer_name: ||W - Q4(W)||²}
        quant_error_8bit: {layer_name: ||W - Q8(W)||²}
        target_hp_ratio: target fraction of layers at 8-bit (default 0.20 for W4.8A8)
        base_hp_ratio: base fraction (unused in ILP directly; kept for API compat)
        parallel_groups: {group_key: [layer_name1, ...]} — equality constraint
        solver: "CBC" (default, bundled with PULP) or "GLPK" (requires system install)
        time_limit: solver time limit in seconds
        verbose: print solver output

    Returns:
        ILPResult with bit allocation per layer
    """
    n_layers = len(layer_names)
    if n_layers == 0:
        return ILPResult({}, 0, 0, 0.0, 0.0, "Empty", 0.0)

    # Build variables: x_i ∈ {1, 2}
    variables = {}
    for i, name in enumerate(layer_names):
        variables[name] = pulp.LpVariable(f"x_{i}", lowBound=1, upBound=2, cat=pulp.LpInteger)

    prob = pulp.LpProblem("SHMQ_BitAllocation", pulp.LpMinimize)

    # ----- Objective -----
    # minimize Σ_i (x_i - 1) * sensitivity_diff_i
    # where sensitivity_diff_i = sensitivity_i * (||W-Q8||² - ||W-Q4||²)
    # This is negative (Q8 error < Q4 error), so the solver prefers x_i = 2 (8-bit).
    sensitivity_diffs = []
    for name in layer_names:
        s = sensitivities.get(name, 0.0)
        d8 = quant_error_8bit.get(name, 0.0)
        d4 = quant_error_4bit.get(name, 0.0)
        diff = s * (d8 - d4)  # negative
        sensitivity_diffs.append(diff)

    prob += pulp.lpSum(
        (variables[name] - 1) * diff
        for name, diff in zip(layer_names, sensitivity_diffs)
    )

    # ----- Constraint 1: memory budget -----
    # Average bits per parameter = 4 + 4 * (fraction at 8-bit)
    # For W4.8A8: average = 4.8, so fraction at 8-bit = 0.20
    # Σ_i params_i * bits_i / Σ_i params_i <= 4 + 4 * target_hp_ratio
    total_params = sum(n_params[name] for name in layer_names)
    target_avg_bits = 4 + 4 * target_hp_ratio  # 4.8 for target_hp_ratio=0.20
    # bits_i = 4 if x_i=1, else 8 (if x_i=2)
    # bits_i = 4 + 4*(x_i - 1) for x_i ∈ {1, 2}
    prob += pulp.lpSum(
        n_params[name] * (4 + 4 * (variables[name] - 1)) for name in layer_names
    ) <= target_avg_bits * total_params, "memory_budget"

    # ----- Constraint 2: parallel layer equality -----
    if parallel_groups:
        for group_key, group_layers in parallel_groups.items():
            group_layers = [n for n in group_layers if n in variables]
            if len(group_layers) <= 1:
                continue
            ref = group_layers[0]
            for other in group_layers[1:]:
                prob += variables[ref] == variables[other], f"parallel_{group_key}_{ref}_{other}"

    # ----- Solve -----
    if solver.upper() == "GLPK":
        solver_obj = pulp.GLPK_CMD(msg=int(verbose), timeLimit=time_limit)
    else:
        solver_obj = pulp.PULP_CBC_CMD(msg=int(verbose), timeLimit=time_limit)
    status = prob.solve(solver_obj)
    status_str = pulp.LpStatus[status]

    # ----- Extract result -----
    bit_alloc: Dict[str, int] = {}
    n_4, n_8 = 0, 0
    total_bits_weighted = 0.0
    total_memory = 0.0
    for name in layer_names:
        x_val = int(round(pulp.value(variables[name])))
        bits = 4 if x_val == 1 else 8
        bit_alloc[name] = bits
        if bits == 4:
            n_4 += 1
        else:
            n_8 += 1
        total_bits_weighted += bits * n_params[name]
        # Memory in MB: params * (bits / 8 bytes) / (1024 * 1024)
        total_memory += n_params[name] * (bits / 8) / (1024 * 1024)

    avg_bits = total_bits_weighted / max(total_params, 1)
    obj_val = pulp.value(prob.objective) if prob.objective is not None else 0.0

    return ILPResult(
        bit_allocation=bit_alloc,
        n_layers_4bit=n_4,
        n_layers_8bit=n_8,
        total_bits=avg_bits,
        total_memory_mb=total_memory,
        solver_status=status_str,
        objective_value=float(obj_val) if obj_val is not None else 0.0,
    )


def solve_ilp_with_base_constraint(
    layer_names: List[str],
    sensitivities: Dict[str, float],
    n_params: Dict[str, int],
    quant_error_4bit: Dict[str, float],
    quant_error_8bit: Dict[str, float],
    target_hp_ratio: float = 0.20,
    base_hp_ratio: float = 0.125,
    parallel_groups: Optional[Dict[str, List[str]]] = None,
    solver: str = "CBC",
    time_limit: int = 30,
) -> ILPResult:
    """Solve ILP with an additional constraint: each layer gets at least base_hp_ratio
    fraction of its channels at 8-bit (the SHMQ paper's UB constraint).

    Note: this is an upper-bound on the ILP — it forces ALL layers to be at 8-bit
    for at least base_hp_ratio of their parameters. This is hard to express at the
    layer-level ILP (since each layer is either 4 or 8 bits), so we implement it
    as a soft constraint: at least base_hp_ratio fraction of LAYERS (by param count)
    must be 8-bit.

    Args: same as solve_ilp_bit_allocation, plus base_hp_ratio.

    Returns: ILPResult.
    """
    n_layers = len(layer_names)
    total_params = sum(n_params[name] for name in layer_names)
    target_avg_bits = 4 + 4 * target_hp_ratio  # 4.8 for W4.8A8
    base_avg_bits = 4 + 4 * base_hp_ratio       # 4.5 for UB=0.125

    variables = {}
    for i, name in enumerate(layer_names):
        variables[name] = pulp.LpVariable(f"x_{i}", lowBound=1, upBound=2, cat=pulp.LpInteger)

    prob = pulp.LpProblem("SHMQ_BitAllocation_WithBase", pulp.LpMinimize)

    # Objective
    sensitivity_diffs = []
    for name in layer_names:
        s = sensitivities.get(name, 0.0)
        d8 = quant_error_8bit.get(name, 0.0)
        d4 = quant_error_4bit.get(name, 0.0)
        diff = s * (d8 - d4)
        sensitivity_diffs.append(diff)
    prob += pulp.lpSum(
        (variables[name] - 1) * diff
        for name, diff in zip(layer_names, sensitivity_diffs)
    )

    # Memory budget constraint (upper bound on total bits)
    prob += pulp.lpSum(
        n_params[name] * (4 + 4 * (variables[name] - 1)) for name in layer_names
    ) <= target_avg_bits * total_params, "memory_budget"

    # Base ratio constraint (lower bound on total bits — guarantees UB floor)
    prob += pulp.lpSum(
        n_params[name] * (4 + 4 * (variables[name] - 1)) for name in layer_names
    ) >= base_avg_bits * total_params, "base_ratio_floor"

    # Parallel constraint
    if parallel_groups:
        for group_key, group_layers in parallel_groups.items():
            group_layers = [n for n in group_layers if n in variables]
            if len(group_layers) <= 1:
                continue
            ref = group_layers[0]
            for other in group_layers[1:]:
                prob += variables[ref] == variables[other], f"parallel_{group_key}_{ref}_{other}"

    if solver.upper() == "GLPK":
        solver_obj = pulp.GLPK_CMD(msg=0, timeLimit=time_limit)
    else:
        solver_obj = pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit)
    status = prob.solve(solver_obj)
    status_str = pulp.LpStatus[status]

    bit_alloc: Dict[str, int] = {}
    n_4, n_8 = 0, 0
    total_bits_weighted = 0.0
    total_memory = 0.0
    for name in layer_names:
        x_val = int(round(pulp.value(variables[name])))
        bits = 4 if x_val == 1 else 8
        bit_alloc[name] = bits
        if bits == 4:
            n_4 += 1
        else:
            n_8 += 1
        total_bits_weighted += bits * n_params[name]
        total_memory += n_params[name] * (bits / 8)
    avg_bits = total_bits_weighted / max(total_params, 1)
    obj_val = pulp.value(prob.objective) if prob.objective is not None else 0.0

    return ILPResult(
        bit_allocation=bit_alloc,
        n_layers_4bit=n_4,
        n_layers_8bit=n_8,
        total_bits=avg_bits,
        total_memory_mb=total_memory,
        solver_status=status_str,
        objective_value=float(obj_val) if obj_val is not None else 0.0,
    )
