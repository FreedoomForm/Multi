"""SHMQQuantLinear — drop-in replacement for nn.Linear at inference time.

Stores REAL packed INT4 + INT8 weights (no fake-quant FP16). Each forward
pass:
  1. Quantizes the input activation to INT8 per-token (symmetric).
  2. Calls the SHMQ parallel two-bit matmul kernel, which partitions the
     reduction along cin into a sensitive INT8 block and an insensitive
     INT4 block, doing both matmuls in one kernel pass and summing.
  3. Adds the (optional) bias and returns the FP16 output.

This module is what gives SHMQ its 2.86x speedup: because both INT4 and
INT8 are native GPU integer formats, there is no dequantization overhead.
"""
from __future__ import annotations
from typing import Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from .weight_packing import pack_shmq_linear, quantize_activation_int8
from .kernel_loader import shmq_matmul, is_cuda_kernel_available


class SHMQQuantLinear(nn.Module):
    """Inference-only Linear with real INT4+INT8 packed weights.

    Construction:
        SHMQQuantLinear.from_packed(packed_dict, bias=...)  — recommended
        SHMQQuantLinear.from_weight(permuted_weight, K, group_size, bias=...)

    Forward:
        y = shmq_matmul(quantize_int8(x), W_int8, W_int4, ...) + bias
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 n_sensitive: int,
                 group_size: int = 128,
                 bias: Optional[torch.Tensor] = None,
                 device: Optional[torch.device] = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_sensitive = n_sensitive
        self.n_insensitive = in_features - n_sensitive
        self.group_size = group_size
        self.n_groups_8 = n_sensitive // group_size if n_sensitive > 0 else 0
        self.n_groups_4 = self.n_insensitive // group_size if self.n_insensitive > 0 else 0
        self._cuda_available = is_cuda_kernel_available()

        dev = device or torch.device("cpu")
        # Persistent buffers (move to device on .to())
        self.register_buffer("qweight_int8",
            torch.zeros(out_features, n_sensitive, dtype=torch.int8, device=dev))
        self.register_buffer("scales_int8",
            torch.zeros(out_features, self.n_groups_8, dtype=torch.float16, device=dev))
        self.register_buffer("qweight_int4",
            torch.zeros(out_features, self.n_insensitive // 2, dtype=torch.uint8, device=dev))
        self.register_buffer("scales_int4",
            torch.zeros(out_features, self.n_groups_4, dtype=torch.float16, device=dev))
        if bias is not None:
            self.register_buffer("bias", bias.to(dev).to(torch.float16).contiguous())
        else:
            self.register_buffer("bias", None)

    # ----------------------------------------------------------------------
    # Constructors
    # ----------------------------------------------------------------------

    @classmethod
    def from_weight(cls,
                    weight: torch.Tensor,            # (cout, cin) — ALREADY permuted
                    n_sensitive: int,
                    group_size: int = 128,
                    bias: Optional[torch.Tensor] = None,
                    perm: Optional[torch.Tensor] = None,
                    precomputed_codes: Optional[torch.Tensor] = None,
                    precomputed_scales: Optional[torch.Tensor] = None,
                    precomputed_n_bits: Optional[int] = None,
                    device: Optional[torch.device] = None) -> "SHMQQuantLinear":
        """Build a SHMQQuantLinear from a single permuted weight tensor.

        Args:
            weight: (cout, cin) float16/float32 weight, ALREADY permuted along
                    cin by the SHMQ permutation (so the first `n_sensitive`
                    channels are the sensitive cluster, the rest insensitive).
            n_sensitive: K — number of leading input channels that get INT8.
            group_size: per-group quantization group size (default 128).
            bias: (cout,) optional bias tensor.
            perm: (cin,) optional permutation buffer (stored for reference).
            precomputed_codes: optional (cout, cin) int8 GPTQ codes from Step 8.
            precomputed_scales: optional (cout, cin // g) float16 scales matching codes.
            precomputed_n_bits: 4 or 8, the bit-width of precomputed_codes.
            device: target device.
        """
        out_features, in_features = weight.shape
        packed = pack_shmq_linear(
            weight, n_sensitive, group_size, perm=perm,
            precomputed_codes=precomputed_codes,
            precomputed_scales=precomputed_scales,
            precomputed_n_bits=precomputed_n_bits,
        )
        obj = cls(in_features, out_features, n_sensitive, group_size, bias, device)
        obj.qweight_int8 = packed["qweight_int8"]
        obj.scales_int8  = packed["scales_int8"]
        obj.qweight_int4 = packed["qweight_int4"]
        obj.scales_int4  = packed["scales_int4"]
        if device is not None:
            obj = obj.to(device)
        return obj

    @classmethod
    def from_packed(cls, packed: Dict[str, torch.Tensor],
                    bias: Optional[torch.Tensor] = None,
                    device: Optional[torch.device] = None) -> "SHMQQuantLinear":
        """Build a SHMQQuantLinear from a pre-packed dict (see weight_packing)."""
        in_features  = packed["in_features"]
        out_features = packed["out_features"]
        n_sensitive  = packed["n_sensitive"]
        group_size   = packed["group_size"]
        obj = cls(in_features, out_features, n_sensitive, group_size, bias, device)
        obj.qweight_int8 = packed["qweight_int8"]
        obj.scales_int8  = packed["scales_int8"]
        obj.qweight_int4 = packed["qweight_int4"]
        obj.scales_int4  = packed["scales_int4"]
        if device is not None:
            obj = obj.to(device)
        return obj

    # ----------------------------------------------------------------------
    # Forward
    # ----------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """y = shmq_matmul(quant_int8(x), W_int8, W_int4, ...) + bias.

        Args:
            x: (..., in_features) float16/float32 — activations ALREADY
               permuted (by fused RMSNorm) so the first n_sensitive channels
               align with the INT8 weight block.

        Returns:
            y: (..., out_features) float16
        """
        # Make sure x is on the same device as the weights.
        x = x.to(self.qweight_int8.device)
        # Quantize activation to INT8 per-token (returns integer codes + scale)
        x_q, x_scale = quantize_activation_int8(x)
        # Parallel two-bit matmul
        y = shmq_matmul(
            x_q, x_scale,
            self.qweight_int8, self.qweight_int4,
            self.scales_int8, self.scales_int4,
            self.group_size,
        )
        if self.bias is not None:
            y = y + self.bias.to(y.dtype)
        return y

    # ----------------------------------------------------------------------
    # Utilities
    # ----------------------------------------------------------------------

    def extra_repr(self) -> str:
        backend = "CUDA" if self._cuda_available else "CPU-fallback"
        bits = (4 * self.n_insensitive + 8 * self.n_sensitive) / max(self.in_features, 1)
        return (f"in={self.in_features}, out={self.out_features}, "
                f"K_s={self.n_sensitive} (INT8), K_i={self.n_insensitive} (INT4), "
                f"avg_bits={bits:.2f}, group={self.group_size}, backend={backend}")

    @torch.no_grad()
    def dequantize_weight(self) -> torch.Tensor:
        """Recover the (cout, cin) float weight by dequantizing the packed buffers.

        Useful for debug / accuracy comparison against fake-quant reference.
        Returns FP32 (cout, cin) tensor with the SHMQ permutation applied.
        """
        from .weight_packing import unpack_int4
        out = torch.zeros(self.out_features, self.in_features, dtype=torch.float32,
                          device=self.qweight_int8.device)
        if self.n_sensitive > 0:
            ws8 = self.scales_int8.to(torch.float32).repeat_interleave(
                self.group_size, dim=1)  # (cout, K_s)
            out[:, :self.n_sensitive] = self.qweight_int8.to(torch.float32) * ws8
        if self.n_insensitive > 0:
            W4_codes = unpack_int4(self.qweight_int4).to(torch.float32)  # (cout, cin-K_s)
            ws4 = self.scales_int4.to(torch.float32).repeat_interleave(
                self.group_size, dim=1)
            out[:, self.n_sensitive:] = W4_codes * ws4
        return out
