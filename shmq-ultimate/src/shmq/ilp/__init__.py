"""SHMQ-Ultimate — ILP subpackage.

Integer Linear Programming bit allocation, adapted from HAWQ-V3 (PULP).

2 bit levels {4, 8}, memory budget = W4.8A8, parallel-layer equality constraint
(q/k/v share bits; up/gate share bits).
"""
from .solver import solve_ilp_bit_allocation, ILPResult

__all__ = ["solve_ilp_bit_allocation", "ILPResult"]
