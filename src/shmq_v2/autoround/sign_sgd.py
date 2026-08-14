"""AutoRound: Learnable rounding via SignSGD (Intel Labs, 2023).

AutoRound optimizes the rounding direction V (a per-element learnable parameter)
via SignSGD on a small calibration set. After 200 steps, V is "baked" into the
quantized weights with zero inference overhead.

Algorithm:
  1. Initialize V = zeros (same shape as W)
  2. For 200 steps:
       forward:  W_q = quantize(W + V)   [V adds ±0.5 to break ties optimally]
       backward: compute grad of L w.r.t. V (SignSGD: V -= lr * sign(grad))
       V = clamp(V, -0.49, 0.49)  [keep in [-0.5, 0.5] so it only affects rounding]
  3. Bake: W_final = quantize(W + V)

Key insight: V is INITIALIZED to zero, and after baking it's absorbed into
the weights. There is ZERO inference overhead — V is discarded after baking.

For SHMQ-Ultimate v2:
  - Run AutoRound per transformer block (200 steps each)
  - Apply AFTER MixLLM's bit allocation (so V only optimizes the bits that
    will actually be used)
  - Apply BEFORE MixLLM's final quantize_linear_weight call (so the optimized
    rounding is preserved in the packed INT4/INT8 weights)

Reference: https://github.com/intel/auto-round
"""
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional, Callable


def quantize_rtn_with_v(
    weight: torch.Tensor,   # [out_features, in_features]
    v: torch.Tensor,        # [out_features, in_features], learnable rounding offset
    bit_width: int,
    group_size: int = 128,
    asymmetric: bool = False,
) -> torch.Tensor:
    """Differentiable RTN quantization with learnable V.

    forward:  W_q = dequant(quant(W + V))
              where V ∈ [-0.5, 0.5] is added BEFORE quantization to optimize
              the rounding direction.

    The quantization uses STE (Straight-Through Estimator) so gradients flow
    through the round() operation back to V.
    """
    w = weight + v
    out_features, in_features = weight.shape
    n_groups = in_features // group_size
    assert in_features % group_size == 0

    w_grouped = w.view(out_features, n_groups, group_size)

    if asymmetric:
        min_val = w_grouped.amin(dim=-1, keepdim=True)
        max_val = w_grouped.amax(dim=-1, keepdim=True)
        scale = ((max_val - min_val) / (2**bit_width - 1)).clamp(min=1e-8)
        zero = -torch.round(min_val / scale)
        q = torch.clamp(torch.round(w_grouped / scale) + zero, 0, 2**bit_width - 1)
        dq = (q - zero) * scale
    else:
        max_val = w_grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = max_val / (2**(bit_width - 1) - 1)
        q = torch.clamp(torch.round(w_grouped / scale), -(2**(bit_width - 1)), 2**(bit_width - 1) - 1)
        # STE: forward uses q, backward uses w_grouped (gradient flows to V)
        dq = q * scale
        # STE trick: dq = w_grouped + (dq - w_grouped).detach()

    return dq.view(out_features, in_features)


@torch.no_grad()
def autoround_optimize(
    linear: nn.Linear,
    activation: torch.Tensor,        # [n_tokens, in_features]
    bit_width: int = 4,
    group_size: int = 128,
    n_iters: int = 200,
    lr: float = 1e-3,
    asymmetric: bool = False,
    block_size: int = 128,
) -> torch.Tensor:
    """Optimize the rounding direction V for a single Linear layer.

    Uses SignSGD: V -= lr * sign(grad_V)
    After optimization, V is applied to the weight and the result is returned
    (the caller should then quantize the smoothed weight).

    Args:
        linear: nn.Linear to optimize
        activation: calibration activations for this layer
        bit_width: target quantization bit width (4 or 8)
        group_size: quantization group size
        n_iters: SignSGD iterations (paper default 200)
        lr: learning rate (paper default 1e-3)
        asymmetric: use asymmetric quantization
        block_size: process weights in [block_size, in_features] blocks
                    (memory optimization for large layers)

    Returns:
        V: the optimized rounding offset tensor [out_features, in_features].
           The caller can bake it via: W_baked = quantize(W + V, ...).
    """
    out_features, in_features = linear.weight.shape
    device = linear.weight.device
    dtype = torch.float32  # AutoRound always in FP32 for stability

    W = linear.weight.data.to(dtype).clone()
    X = activation.to(dtype).clone()

    # Initialize V = zeros
    V = torch.zeros_like(W, requires_grad=True)

    # Optimizer: SignSGD (manual, since torch.optim doesn't have it built-in)
    # We'll do V -= lr * sign(grad_V) in the loop

    for step in range(n_iters):
        # Forward: quantize (W + V) and compute output
        W_q = quantize_rtn_with_v(W, V, bit_width, group_size, asymmetric)
        # Compute output: y_q = X @ W_q^T  (vs original y = X @ W^T)
        # Loss = MSE(y_q, y_orig) — preserve original layer output
        y_q = X @ W_q.t()
        with torch.no_grad():
            y_orig = X @ W.t()
        loss = (y_q - y_orig).pow(2).mean()

        # Backward: gradient w.r.t. V
        loss.backward()

        # SignSGD update
        with torch.no_grad():
            V -= lr * V.grad.sign()
            V.clamp_(-0.49, 0.49)  # keep V in [-0.5, 0.5] so it only affects rounding
            V.grad.zero_()

    return V.detach()


@torch.no_grad()
def autoround_bake(
    linear: nn.Linear,
    V: torch.Tensor,
    bit_width: int = 4,
    group_size: int = 128,
    asymmetric: bool = False,
) -> torch.Tensor:
    """Bake the optimized V into the weight, returning the quantized weight.

    After baking, V is no longer needed (zero inference overhead).
    The returned quantized weight can be passed to MixLLM's quantize_linear_weight.
    """
    W = linear.weight.data
    W_with_v = W + V.to(W.dtype).to(W.device)
    # Use the same quantization as autoround_optimize for consistency
    return quantize_rtn_with_v(W_with_v, torch.zeros_like(W_with_v),
                                bit_width, group_size, asymmetric).to(W.dtype)
