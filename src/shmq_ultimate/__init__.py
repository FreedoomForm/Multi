"""SHMQ-Ultimate: hierarchical 3-level {FP16, INT8, INT4} mixed-precision
quantization for LLMs, vLLM-compatible.

Combines:
  - SHMQ (EMNLP Industry 2025): hierarchical inter/intra-layer mixed precision,
    decoupled permutation, permutation fusion into RMSNorm, parallel constraint
  - HAWQ-V3: ILP bit allocation (PULP)
  - SliM-LLM: GPTQ/OBS per-element Hessian + SQC calibration
  - AutoRound (Intel): SignSGD learnable rounding
  - SmoothQuant (MIT): activation outlier migration
  - PolyQ (ICCAD 2026): ISA-aware quanta matching + layout propagation
  - MixLLM (Microsoft): mixed-precision GEMM kernel design + vLLM integration
"""

__version__ = "3.0.0"

from .config import SHMQUltimateConfig  # noqa: F401
from .pipeline import SHMQUltimatePipeline  # noqa: F401
