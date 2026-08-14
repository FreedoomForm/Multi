"""
SHMQ-Ultimate: Static Hierarchical Mix-precision Quantization for LLMs.

Combines the best of:
- HAWQ-V3   : ILP bit allocation (PULP)  [https://github.com/Zhen-Dong/HAWQ]
- SliM-LLM  : AutoGPTQ + per-element OBS Hessian + SQC  [https://github.com/Aaronhuang-778/SliM-LLM]
- AutoRound : SignSGD learnable rounding (200 steps)  [https://github.com/intel/auto-round]
- SmoothQuant: Activation outlier migration  [https://github.com/mit-han-lab/smoothquant]
- SHMQ paper: Decoupled permutation + PermutedRMSNorm + parallel constraint (custom)
              [https://aclanthology.org/2025.emnlp-industry.175/]

Pipeline (9 steps):
1. SmoothQuant pre-processing (migration of activation outliers)
2. Inter-layer Fisher sensitivity (Eq. 7) + intra-layer OBS (Eq. 10) + Manhattan (Eq. 11)
3. ILP bit allocation {4, 8} (PULP) with parallel-layer constraint
4. Decoupled permutation (Eq. 12) — sort by sensitivity, partition, sort by magnitude
5. PermutedRMSNorm fusion — bake permutation into RMSNorm (zero overhead)
6. AutoRound SignSGD — 200 steps to optimize V (rounding offset), then bake into W
7. SQC calibration — optimize scale factors with salience weighting
8. AutoGPTQ quantization — apply final W4.8A8 mixed-precision
9. Inference (Marlin on GPU; fake-quant on CPU)

Reference targets:
- Format: W4.8A8 = W4A8 + 20% W8A8 (so Ut = 0.20)
- Base high-precision ratio (Ub = UB): 12.5%
- Dampening factor (lambda): 0.1
- Group size: 128
- Calibration: 128 samples x 2048 tokens from WikiText-2
- Test model: Qwen2.5-7B-Instruct (target gap <= 0.13% from FP16, speedup >= 2.86x)
"""
from .config import SHMQConfig
from .pipeline import SHMQPipeline

__version__ = "0.1.0"
__all__ = ["SHMQConfig", "SHMQPipeline"]
