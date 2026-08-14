"""3-level ILP bit allocation for SHMQ-Ultimate.

Extends HAWQ-V3 ILP (https://github.com/Zhen-Dong/HAWQ/blob/main/ILP.ipynb)
to support THREE bit levels {4, 8, 16} instead of just two.

Key design choices
------------------
* 16-bit means "keep original FP16/BF16" — quantization error ≈ 0.
  This is the natural upper bound for any FP16/BF16-trained LLM.
* Indicator variables y_4, y_8, y_16 per layer (binary, sum to 1).
* Objective: minimize Σ_i sensitivity_i × (y_4·qerr_4 + y_8·qerr_8 + y_16·qerr_16).
  Since qerr_16 ≈ 0 and qerr_8 < qerr_4, the solver is incentivized
  to push sensitive layers to higher precision, subject to the memory budget.
* Memory budget: average bits per parameter ≤ target_avg_bits.
  For W5.0A8 (target_hp_ratio_16=0.05, target_hp_ratio_8=0.20):
      avg_bits = 4*0.75 + 8*0.20 + 16*0.05 = 3.0 + 1.6 + 0.8 = 5.4
* Parallel-layer equality constraint (q/k/v same bits; up/gate same bits)
  — needed for permutation fusion compatibility (SHMQ Appendix A.3.1).
* ISA-aware quanta matching (PolyQ) is enforced as a POST-PROCESS step,
  not in the ILP itself — see polyq.isa_matching.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
import logging
import numpy as np
import pulp

logger = logging.getLogger("shmq")


@dataclass
class ILPResult3L:
    """Result of 3-level ILP bit allocation {4, 8, 16}."""
    bit_allocation: Dict[str, int]                  # layer_name -> 4 | 8 | 16
    n_layers_4bit: int
    n_layers_8bit: int
    n_layers_16bit: int
    total_bits: float                              # average bits per parameter
    total_memory_mb: float                         # estimated memory in MB
    solver_status: str
    objective_value: float
    constraint_slack: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"ILP-3L Result:\n"
            f"  Status: {self.solver_status}\n"
            f"  Layers @ 4-bit:  {self.n_layers_4bit}\n"
            f"  Layers @ 8-bit:  {self.n_layers_8bit}\n"
            f"  Layers @ 16-bit: {self.n_layers_16bit}\n"
            f"  Average bits per parameter: {self.total_bits:.4f}\n"
            f"  Estimated memory: {self.total_memory_mb:.2f} MB\n"
            f"  Objective value:  {self.objective_value:.6e}\n"
        )

    def per_layer_summary(self, layer_names: Optional[List[str]] = None) -> str:
        """Return a human-readable per-layer breakdown."""
        names = layer_names or sorted(self.bit_allocation.keys())
        lines = ["Per-layer bit allocation:"]
        for n in names:
            lines.append(f"  {n:60s} -> {self.bit_allocation[n]:2d}-bit")
        return "\n".join(lines)


def solve_ilp_3level(
    layer_names: List[str],
    sensitivities: Dict[str, float],
    n_params: Dict[str, int],
    quant_error_4bit: Dict[str, float],
    quant_error_8bit: Dict[str, float],
    quant_error_16bit: Optional[Dict[str, float]] = None,
    target_avg_bits: float = 5.4,
    min_avg_bits: Optional[float] = None,
    parallel_groups: Optional[Dict[str, List[str]]] = None,
    solver: str = "CBC",
    time_limit: int = 60,
    verbose: bool = False,
) -> ILPResult3L:
    """Solve the 3-level {4, 8, 16} ILP bit allocation problem.

    Variables: y_i_4, y_i_8, y_i_16 ∈ {0, 1} with y_4 + y_8 + y_16 = 1.
    Objective:
        minimize Σ_i s_i · (y_4·q4 + y_8·q8 + y_16·q16)
    Constraints:
        (1) Memory budget: Σ params_i · bits_i / Σ params_i ≤ target_avg_bits
            where bits_i = 4·y_4 + 8·y_8 + 16·y_16
        (2) Optional floor: same as (1) but ≥ min_avg_bits (UB floor)
        (3) Parallel-layer equality: x_q = x_k = x_v (encoded via indicator
            equality constraints on the y-vector)

    Args:
        layer_names: list of layer names to allocate bits for.
        sensitivities: {layer_name: Fisher sensitivity (inter-layer)}.
        n_params: {layer_name: number of parameters (cout * cin)}.
        quant_error_4bit: {layer_name: ||W - Q4(W)||^2}.
        quant_error_8bit: {layer_name: ||W - Q8(W)||^2}.
        quant_error_16bit: optional {layer_name: ||W - Q16(W)||^2}.
            If None, assumed 0 (FP16 quantization is lossless).
        target_avg_bits: target average bits per parameter (e.g. 5.4 for
            75% 4-bit + 20% 8-bit + 5% 16-bit).
        min_avg_bits: optional lower bound on average bits (UB floor).
        parallel_groups: {group_key: [layer_name1, ...]} — equality constraint.
        solver: "CBC" (default, bundled with PULP) or "GLPK".
        time_limit: solver time limit in seconds.
        verbose: print solver output.

    Returns:
        ILPResult3L with bit allocation per layer.
    """
    n_layers = len(layer_names)
    if n_layers == 0:
        return ILPResult3L({}, 0, 0, 0, 0.0, 0.0, "Empty", 0.0)

    # Default qerr_16 = 0 (FP16 ≈ lossless for FP16/BF16 weights)
    if quant_error_16bit is None:
        quant_error_16bit = {n: 0.0 for n in layer_names}

    # ----- Variables -----
    # y[name][bit] ∈ {0, 1}; bits ∈ {4, 8, 16}
    y: Dict[str, Dict[int, pulp.LpVariable]] = {}
    for i, name in enumerate(layer_names):
        y[name] = {
            4:  pulp.LpVariable(f"y4_{i}",  cat=pulp.LpBinary),
            8:  pulp.LpVariable(f"y8_{i}",  cat=pulp.LpBinary),
            16: pulp.LpVariable(f"y16_{i}", cat=pulp.LpBinary),
        }

    prob = pulp.LpProblem("SHMQ_3Level_BitAllocation", pulp.LpMinimize)

    # ----- Constraint: y_4 + y_8 + y_16 = 1 for each layer -----
    for name in layer_names:
        prob += (y[name][4] + y[name][8] + y[name][16] == 1,
                 f"onehot_{name.replace('.', '_')}")

    # ----- Objective -----
    # minimize Σ_i s_i · (y_4·q4 + y_8·q8 + y_16·q16)
    # Equivalent: minimize Σ_i s_i · q4  (constant)
    #           + Σ_i s_i · (q8 - q4) · y_8
    #           + Σ_i s_i · (q16 - q4) · y_16
    # Since q8 < q4 and q16 ≤ q8, the coefficients on y_8 and y_16 are negative,
    # so the solver wants to set them to 1 — subject to the memory budget.
    objective_terms = []
    for name in layer_names:
        s  = float(sensitivities.get(name, 0.0))
        q4 = float(quant_error_4bit.get(name, 0.0))
        q8 = float(quant_error_8bit.get(name, 0.0))
        q16 = float(quant_error_16bit.get(name, 0.0))
        # Full form: s * (y4*q4 + y8*q8 + y16*q16)
        # Substitute y4 = 1 - y8 - y16:
        #   = s * (q4 + (q8 - q4) * y8 + (q16 - q4) * y16)
        objective_terms.append(s * (q8 - q4)  * y[name][8])
        objective_terms.append(s * (q16 - q4) * y[name][16])
    prob += pulp.lpSum(objective_terms), "objective"

    # ----- Constraint (1): memory budget -----
    total_params = sum(int(n_params[n]) for n in layer_names)
    bits_expr = pulp.lpSum(
        n_params[name] * (4 * y[name][4] + 8 * y[name][8] + 16 * y[name][16])
        for name in layer_names
    )
    prob += (bits_expr <= target_avg_bits * total_params, "memory_budget")

    # ----- Constraint (2): optional floor -----
    if min_avg_bits is not None:
        prob += (bits_expr >= min_avg_bits * total_params, "base_ratio_floor")

    # ----- Constraint (3): parallel-layer equality -----
    # For each parallel group, force all members to share the same y-vector.
    # We enforce: y[name_a][b] == y[name_b][b] for b in {4, 8, 16}.
    if parallel_groups:
        for group_key, group_layers in parallel_groups.items():
            present = [n for n in group_layers if n in y]
            if len(present) <= 1:
                continue
            ref = present[0]
            for other in present[1:]:
                for b in (4, 8, 16):
                    prob += (y[ref][b] == y[other][b],
                             f"parallel_{group_key}_{ref}_{other}_b{b}")

    # ----- Solve -----
    if solver.upper() == "GLPK":
        solver_obj = pulp.GLPK_CMD(msg=int(verbose), timeLimit=time_limit)
    else:
        solver_obj = pulp.PULP_CBC_CMD(msg=int(verbose), timeLimit=time_limit)
    status = prob.solve(solver_obj)
    status_str = pulp.LpStatus[status]

    # ----- Extract result -----
    bit_alloc: Dict[str, int] = {}
    n_4 = n_8 = n_16 = 0
    total_bits_weighted = 0.0
    total_memory = 0.0
    for name in layer_names:
        y4_val  = pulp.value(y[name][4])  or 0.0
        y8_val  = pulp.value(y[name][8])  or 0.0
        y16_val = pulp.value(y[name][16]) or 0.0
        # Numerical safety: pick argmax
        vals = {4: y4_val, 8: y8_val, 16: y16_val}
        bits = max(vals, key=vals.get)
        bit_alloc[name] = bits
        if bits == 4:
            n_4 += 1
        elif bits == 8:
            n_8 += 1
        else:
            n_16 += 1
        total_bits_weighted += bits * n_params[name]
        total_memory += n_params[name] * (bits / 8)  # bytes

    avg_bits = total_bits_weighted / max(total_params, 1)
    obj_val = pulp.value(prob.objective) if prob.objective is not None else 0.0

    # Constraint slack (memory budget)
    slack: Dict[str, float] = {}
    for c in prob.constraints.values():
        slack[c.name] = float(pulp.value(c) - c.constant) if c.constant else 0.0

    return ILPResult3L(
        bit_allocation=bit_alloc,
        n_layers_4bit=n_4,
        n_layers_8bit=n_8,
        n_layers_16bit=n_16,
        total_bits=avg_bits,
        total_memory_mb=total_memory / (1024 * 1024),
        solver_status=status_str,
        objective_value=float(obj_val) if obj_val is not None else 0.0,
        constraint_slack=slack,
    )


def compute_target_avg_bits(
    ratio_4: float, ratio_8: float, ratio_16: float,
) -> float:
    """Compute target average bits per parameter from ratios.

    Example: ratio_4=0.75, ratio_8=0.20, ratio_16=0.05 → 5.4 bits/param.
    """
    assert abs(ratio_4 + ratio_8 + ratio_16 - 1.0) < 1e-6, \
        f"Ratios must sum to 1.0, got {ratio_4 + ratio_8 + ratio_16}"
    return 4.0 * ratio_4 + 8.0 * ratio_8 + 16.0 * ratio_16
