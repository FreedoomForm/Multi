"""Bake V (rounding offset) into weights for zero inference overhead.

After SignSGD optimization, V is folded into the weights:
    W_baked = Q(W, V*) = scale * clamp(round(W/scale + V*), -max_q, max_q-1) * scale

The wrapper is then replaced by the original Linear (with quantized weights).
No V is needed at inference time → zero overhead.

Reference: AutoRound wrapper.py::WrapperLinear.unwrapper.
"""
from __future__ import annotations
from typing import Dict
import torch
from .wrapper import WrapperLinear


def bake_v_into_weights(wrappers: Dict[str, WrapperLinear]) -> None:
    """Bake the optimized V into each wrapper's original weight tensor.

    After this, the wrapper's orig_layer.weight contains the quantized weights
    (with V applied), and the wrapper can be safely removed.
    """
    for name, wrapper in wrappers.items():
        wrapper.bake()
        print(f"[baking] Baked V into {name}")
