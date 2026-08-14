"""Wrapper for nn.Linear that adds a learnable rounding value V (AutoRound).

For each group of `group_size` weights, V has shape (cout, cin // group_size, group_size)
(or equivalently reshaped to weight shape). V is initialized to zero and optimized
via SignSGD to minimize the block reconstruction error.

Forward:
    Q(w) = scale * clamp(round_ste(w / scale + V), -max_q, max_q-1)
    where round_ste(x) = (x.round() - x).detach() + x  (STE: forward=round, backward=identity)

Reference: AutoRound wrapper.py (WrapperLinear), data_type/int.py (quant_tensor_sym).
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from ..utils import get_module_by_name, set_module_by_name


def round_ste(x: torch.Tensor) -> torch.Tensor:
    """Round with Straight-Through Estimator (STE).

    Forward: round(x)
    Backward: identity (gradient passes through as if no rounding happened)
    """
    return (x.round() - x).detach() + x


def compute_scale_and_maxq(w: torch.Tensor, n_bits: int, group_size: int,
                           symmetric: bool = True) -> Tuple[torch.Tensor, int]:
    """Compute per-group scale and max_q for symmetric quantization.

    Returns:
        scale: (cout, n_groups) tensor
        max_q: int (2^(n_bits-1) - 1 for symmetric)
    """
    cout, cin = w.shape
    n_groups = cin // group_size
    w_grouped = w.reshape(cout, n_groups, group_size).float()
    max_abs = w_grouped.abs().amax(dim=-1)  # (cout, n_groups)
    max_q = 2 ** (n_bits - 1) - 1  # 7 for 4-bit, 127 for 8-bit
    scale = max_abs.clamp(min=1e-8) / max_q
    return scale, max_q


class WrapperLinear(nn.Module):
    """Wraps an nn.Linear with a learnable V (rounding offset).

    Attributes:
        orig_layer: the original nn.Linear (kept frozen)
        value (V): the learnable rounding offset, shape = grouped weight shape
        n_bits, group_size, symmetric: quantization config
    """

    def __init__(self, orig_layer: nn.Linear, n_bits: int = 4, group_size: int = 128,
                 symmetric: bool = True):
        super().__init__()
        self.orig_layer = orig_layer
        self.n_bits = n_bits
        self.group_size = group_size
        self.symmetric = symmetric

        # Compute scale (frozen, not optimized)
        with torch.no_grad():
            scale, max_q = compute_scale_and_maxq(
                orig_layer.weight.data, n_bits, group_size, symmetric
            )
        self.register_buffer("scale", scale)  # (cout, n_groups)
        self.max_q = max_q

        # Initialize V = zeros (shape = weight shape)
        # V will be optimized via SignSGD
        w_shape = orig_layer.weight.shape  # (cout, cin)
        self.value = nn.Parameter(torch.zeros(w_shape, dtype=torch.float32),
                                  requires_grad=True)

    def _qdq_weight(self, v: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Quantize-dequantize the weight with the given V (rounding offset).

        Q(w) = scale * clamp(round_ste(w / scale + V), -max_q, max_q-1)

        Args:
            v: optional V tensor (defaults to self.value)

        Returns:
            qdq_weight: (cout, cin) fake-quantized weight (dequantized)
        """
        if v is None:
            v = self.value
        w = self.orig_layer.weight.data.float()  # (cout, cin)
        cout, cin = w.shape
        n_groups = cin // self.group_size
        # Expand scale to match weight shape
        scale_expanded = self.scale.unsqueeze(-1).expand(cout, n_groups, self.group_size)
        scale_expanded = scale_expanded.reshape(cout, cin)
        # Quantize with V
        w_normed = w / scale_expanded + v
        q = round_ste(w_normed).clamp(-self.max_q, self.max_q - 1)
        # Dequantize
        qdq = q * scale_expanded
        return qdq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using the quantized weight."""
        qdq_w = self._qdq_weight()
        # Use functional linear with quantized weight
        return nn.functional.linear(x, qdq_w, self.orig_layer.bias)

    def bake(self) -> None:
        """Bake the optimized V into the original layer's weight (zero inference overhead).

        After baking:
            orig_layer.weight.data = Q(W, V*)
            V is discarded
        The wrapper can then be replaced by the original layer for inference.
        """
        with torch.no_grad():
            qdq_w = self._qdq_weight(self.value.detach())
            self.orig_layer.weight.data = qdq_w.to(self.orig_layer.weight.dtype)

    def unwrapper(self) -> nn.Linear:
        """Bake V and return the original nn.Linear (with quantized weights)."""
        self.bake()
        return self.orig_layer


def wrap_model_linears(model: nn.Module, layer_names: List[str],
                       n_bits: int = 4, group_size: int = 128,
                       symmetric: bool = True) -> Dict[str, WrapperLinear]:
    """Wrap the given Linear layers in the model with WrapperLinear.

    Returns: dict {layer_name: WrapperLinear}
    """
    wrappers = {}
    for name in layer_names:
        mod = get_module_by_name(model, name)
        if not isinstance(mod, nn.Linear):
            print(f"[autoround] WARNING: {name} is not nn.Linear, skipping")
            continue
        wrapper = WrapperLinear(mod, n_bits=n_bits, group_size=group_size,
                                symmetric=symmetric)
        set_module_by_name(model, name, wrapper)
        wrappers[name] = wrapper
    return wrappers


def unwrap_model_linears(model: nn.Module,
                          wrappers: Dict[str, WrapperLinear]) -> None:
    """Bake V into weights and restore original Linear modules.

    After this, the model is ready for inference (V is folded in).
    """
    for name, wrapper in wrappers.items():
        wrapper.bake()
        set_module_by_name(model, name, wrapper.orig_layer)
