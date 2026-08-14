"""SHMQ-Ultimate — Sensitivity subpackage.

Implements:
- Inter-layer Fisher sensitivity (SHMQ Eq. 7)
- Inter-layer PyHessian trace (HAWQ-V3 alternative)
- Intra-layer per-element OBS sensitivity (SHMQ Eq. 10/24)
- Manhattan norm aggregation for channel sensitivity (SHMQ Eq. 11)
- Parallel layer constraint (SHMQ Appendix A.3.1)
"""
from .fisher import compute_inter_layer_fisher_sensitivity
from .pyhessian_trace import compute_inter_layer_pyhessian_trace
from .obs import compute_intra_layer_obs_sensitivity, OBSHessian
from .manhattan import aggregate_manhattan_channel_sensitivity
from .parallel import (
    average_inter_layer_parallel_sensitivity,
    concatenate_intra_layer_parallel_sensitivity,
)

__all__ = [
    "compute_inter_layer_fisher_sensitivity",
    "compute_inter_layer_pyhessian_trace",
    "compute_intra_layer_obs_sensitivity",
    "OBSHessian",
    "aggregate_manhattan_channel_sensitivity",
    "average_inter_layer_parallel_sensitivity",
    "concatenate_intra_layer_parallel_sensitivity",
]
