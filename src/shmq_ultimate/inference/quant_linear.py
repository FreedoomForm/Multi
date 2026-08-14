"""Mixed-precision linear layer (MixLLM-style, extended to 3 levels).

Stores the input-channel-clustered weight as three contiguous segments:
    [0, n16)          FP16 columns (kept exact)
    [n16, n16+n8)     INT8 columns, per-group (g=128) symmetric scales
    [n16+n8, cin)     INT4 columns, packed 2 per byte, per-group scales

Forward (PyTorch reference; the CUDA kernel in benchmarks/gpu implements
the same math fused):
    y  = x16 @ W16^T
       + (qx8 @ W8^T)  * (s8  outer sx8)
       + (qx4 @ W4u^T) * (s4  outer sx4)
where activations are quantized per token to INT8 (SmoothQuant-style
static-friendly symmetric quantization).
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from ..layer_quantizer import QuantizedLayerResult
from ..quant_utils import pack_int4, quantize_activation_per_token, unpack_int4


class SHMQUltimateLinear(nn.Module):
    def __init__(
        self,
        cout: int,
        n16: int,
        n8: int,
        n4: int,
        group_size: int = 128,
        bias: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.cout, self.n16, self.n8, self.n4 = cout, n16, n8, n4
        self.group_size = group_size
        g = group_size
        self.register_buffer("w16", torch.zeros(cout, n16, dtype=dtype))
        self.register_buffer("w8", torch.zeros(cout, n8, dtype=torch.int8))
        self.register_buffer("s8", torch.zeros(cout, max(n8 // g, 0), dtype=torch.float32))
        self.register_buffer("w4", torch.zeros(cout, n4 // 2, dtype=torch.uint8))
        self.register_buffer("s4", torch.zeros(cout, max(n4 // g, 0), dtype=torch.float32))
        if bias:
            self.register_buffer("bias", torch.zeros(cout, dtype=dtype))
        else:
            self.bias = None

    @classmethod
    def from_quantized(
        cls,
        result: QuantizedLayerResult,
        group_size: int = 128,
        bias: Optional[torch.Tensor] = None,
        dtype: torch.dtype = torch.float32,
    ) -> "SHMQUltimateLinear":
        part = result.partition
        cout, cin = result.w_deq.shape
        n16, n8, n4 = part.n16, part.n8, part.n4
        g = group_size
        mod = cls(cout, n16, n8, n4, g, bias is not None, dtype)
        # FP16 segment: keep dequantized (== original) weight
        mod.w16.copy_(result.w_deq[:, :n16].to(dtype))
        # INT8 segment
        if n8 > 0:
            mod.w8.copy_(result.codes[:, n16 : n16 + n8].to(torch.int8))
            mod.s8.copy_(result.scales[:, n16 // g : (n16 + n8) // g])
        # INT4 segment
        if n4 > 0:
            c4 = result.codes[:, n16 + n8 :].to(torch.int8)
            mod.w4.copy_(pack_int4(c4))
            mod.s4.copy_(result.scales[:, (n16 + n8) // g :])
        if bias is not None:
            mod.bias.copy_(bias.to(dtype))
        return mod

    def dequantize_weight(self) -> torch.Tensor:
        g = self.group_size
        parts = []
        if self.n16 > 0:
            parts.append(self.w16.float())
        if self.n8 > 0:
            w8 = self.w8.float().view(self.cout, self.n8 // g, g)
            parts.append((w8 * self.s8.unsqueeze(-1)).view(self.cout, self.n8))
        if self.n4 > 0:
            w4 = unpack_int4(self.w4).float().view(self.cout, self.n4 // g, g)
            parts.append((w4 * self.s4.unsqueeze(-1)).view(self.cout, self.n4))
        return torch.cat(parts, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_shape = x.shape
        xf = x.reshape(-1, in_shape[-1]).float()
        g = self.group_size
        y = torch.zeros(xf.shape[0], self.cout, dtype=torch.float32, device=x.device)
        # FP16 segment — dense matmul (cuBLAS path on GPU)
        if self.n16 > 0:
            y += xf[:, : self.n16] @ self.w16.float().t()
        # INT8 segment — per-token activation quant, group-wise integer matmul
        if self.n8 > 0:
            xs = xf[:, self.n16 : self.n16 + self.n8]
            qx, sx = quantize_activation_per_token(xs, 8)  # (M,n8) int8, (M,) scale
            qxf = qx.float().view(-1, self.n8 // g, g)
            w8 = self.w8.float().view(self.cout, self.n8 // g, g)
            # per-group partial products: (M, cout)
            acc = torch.einsum("mkg,ckg->mck", qxf, w8)  # (M, cout, ngroups)
            y += (acc * self.s8.float().unsqueeze(0)).sum(-1) * sx
        # INT4 segment
        if self.n4 > 0:
            xs = xf[:, self.n16 + self.n8 :]
            qx, sx = quantize_activation_per_token(xs, 8)
            w4 = unpack_int4(self.w4).float().view(self.cout, self.n4 // g, g)
            qxf = qx.float().view(-1, self.n4 // g, g)
            acc = torch.einsum("mkg,ckg->mck", qxf, w4)
            y += (acc * self.s4.float().unsqueeze(0)).sum(-1) * sx
        if self.bias is not None:
            y += self.bias.float().unsqueeze(0)
        return y.to(x.dtype).reshape(*in_shape[:-1], self.cout)

    def memory_bytes(self) -> int:
        n = self.w16.numel() * self.w16.element_size()
        n += self.w8.numel() + self.s8.numel() * 4
        n += self.w4.numel() + self.s4.numel() * 4
        if self.bias is not None:
            n += self.bias.numel() * self.bias.element_size()
        return n

    def extra_repr(self) -> str:
        return f"cout={self.cout}, n16={self.n16}, n8={self.n8}, n4={self.n4}, g={self.group_size}"


def convert_linears(
    model: nn.Module,
    results: Dict[str, QuantizedLayerResult],
    group_size: int = 128,
) -> int:
    """Replace nn.Linear modules named in `results` with SHMQUltimateLinear."""
    replaced = 0
    named = dict(model.named_modules())
    for key, res in results.items():
        parent_name, _, child = key.rpartition(".")
        parent = named.get(parent_name)
        if parent is None:
            continue
        old = getattr(parent, child, None)
        if not isinstance(old, nn.Linear):
            continue
        bias = old.bias.data if old.bias is not None else None
        new = SHMQUltimateLinear.from_quantized(res, group_size, bias, dtype=old.weight.dtype)
        setattr(parent, child, new)
        replaced += 1
    return replaced
