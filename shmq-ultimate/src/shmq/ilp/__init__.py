"""SHMQ-Ultimate — ILP subpackage.

Integer Linear Programming bit allocation, adapted from HAWQ-V3 (PULP).

Two solvers:
    * `solve_ilp_bit_allocation`   — 2-level {4, 8} (legacy, original SHMQ).
    * `solve_ilp_3level`           — 3-level {4, 8, 16} (SHMQ-Ultimate extension).

Both support memory budget constraints and parallel-layer equality
(q/k/v share bits; up/gate share bits).
"""
from .solver import solve_ilp_bit_allocation, solve_ilp_with_base_constraint, ILPResult
from .solver_3level import solve_ilp_3level, ILPResult3L, compute_target_avg_bits

__all__ = [
    "solve_ilp_bit_allocation",
    "solve_ilp_with_base_constraint",
    "ILPResult",
    "solve_ilp_3level",
    "ILPResult3L",
    "compute_target_avg_bits",
]
