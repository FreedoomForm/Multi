"""SHMQ-Ultimate — AutoRound subpackage.

Implements SignSGD learnable rounding (Intel AutoRound, paper:
"Optimize Weight Rounding via Signed Gradient Descent for the Quantization of LLMs").

Algorithm:
1. Initialize V (rounding offset) = zeros, shape = grouped weight shape
2. For 200 steps:
     - Forward: compute Q(W) = scale * clamp(round(W/scale + V), -max_q, max_q-1)
     - Compute reconstruction loss = ||W_quantized - W_original||^2 (block-wise)
     - Backward: get gradient ∂L/∂V
     - Update: V ← V - lr * sign(∂L/∂V)  (SignSGD)
3. Bake V into W: W_baked = Q(W, V*) — zero inference overhead (V is folded in)

Reference: AutoRound repo (sign_round/quantizer.py, sign_sgd.py, wrapper.py).
"""
from .sign_sgd import SignSGD
from .wrapper import WrapperLinear, wrap_model_linears
from .baking import bake_v_into_weights
from .autoround_block import autoround_block

__all__ = [
    "SignSGD",
    "WrapperLinear",
    "wrap_model_linears",
    "bake_v_into_weights",
    "autoround_block",
]
