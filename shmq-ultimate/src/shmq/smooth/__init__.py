"""SHMQ-Ultimate — SmoothQuant subpackage.

Implements activation outlier migration as Step 1 of the SHMQ pipeline.
Reference: https://github.com/mit-han-lab/smoothquant

SmoothQuant migrates the "quantization difficulty" from activations to weights:
    s_j = (max|X_j|)^alpha / (max|W_j|)^(1-alpha)
    W'_j = W_j * s_j  (smoother weights, harder to quantize)
    X'_j = X_j / s_j  (smoother activations, easier to quantize)

For Qwen2.5 (RMSNorm-based, no LN bias):
    - input_layernorm  <-> [q_proj, k_proj, v_proj]   (3 siblings share input)
    - post_attention_layernorm <-> [gate_proj, up_proj] (2 siblings)
    - o_proj and down_proj are NOT smoothed (no preceding Norm to fold into)
"""
from .smooth import smooth_lm, smooth_ln_fcs_llama_like
from .calibration import get_act_scales, ActivationScaleCollector

__all__ = [
    "smooth_lm", "smooth_ln_fcs_llama_like",
    "get_act_scales", "ActivationScaleCollector",
]
