"""MixLLM adapter for SHMQ-Ultimate 3-level {4, 8, 16} quantization.

This module wraps Microsoft's MixLLM CUDA kernel
(https://github.com/microsoft/MixLLM) and extends it with an FP16 path
for the new {16, 8, 4} 3-level scheme.

Key design
-----------
MixLLM's `LinearMixLLM` natively supports mixed INT4/INT8 in a single
linear layer via two parallel channel-index arrays:
    * `indices_int8` — output channels quantized to INT8
    * `indices_int4` — output channels quantized to INT4
The kernel itself uses these indices to scatter outputs back to their
original positions, which means SHMQ's decoupled permutation is
**natively compatible** with MixLLM — we just feed the permuted channel
indices directly.

For 3-level {4, 8, 16}:
    * C16 (FP16) channels are routed through a standard cuBLAS GEMM
      (torch.matmul), since MixLLM has no FP16 weight path.
    * C8 + C4 channels go through MixLLM's mixed-precision kernel.
    * The two partial outputs are summed.

This avoids modifying MixLLM's CUDA kernel source — we treat it as a
black box and add a Python-side FP16 path around it.

For SHMQ, since SHMQ's decoupled permutation already ensures that
sensitive channels (C16, C8) and insensitive channels (C4) are
CONTIGUOUS in the permuted order, the FP16/INT8/INT4 channel sets are
also contiguous, which is what MixLLM's kernel expects.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import logging
import os
import sys
import torch
import torch.nn as nn

logger = logging.getLogger("shmq")

# Lazy import of MixLLM (the package may not be installed in all environments)
_MIXLLM_AVAILABLE: Optional[bool] = None


def is_mixllm_available() -> bool:
    """Check whether MixLLM (with compiled CUDA kernel) is importable."""
    global _MIXLLM_AVAILABLE
    if _MIXLLM_AVAILABLE is not None:
        return _MIXLLM_AVAILABLE
    try:
        # Make sure external/MixLLM is on PYTHONPATH if it was not pip-installed
        external_mixllm = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))))),
            "external", "MixLLM",
        )
        if os.path.isdir(external_mixllm) and external_mixllm not in sys.path:
            sys.path.insert(0, external_mixllm)
        import mixllm  # noqa: F401
        from mixllm.nn.modules.ops import mixllm_gemm  # noqa: F401
        _MIXLLM_AVAILABLE = True
        logger.info("[mixllm] MixLLM package is available.")
    except Exception as e:
        _MIXLLM_AVAILABLE = False
        logger.warning(f"[mixllm] MixLLM not available: {e}")
    return _MIXLLM_AVAILABLE


def _import_mixllm_ops():
    """Import the mixllm ops module (assumes is_mixllm_available() == True)."""
    from mixllm.nn.modules import ops as mixllm_ops
    return mixllm_ops


# ----------------------------------------------------------------------
# Weight packing helpers
# ----------------------------------------------------------------------

def pack_int4_weights(weight_fp16: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack a (k, n) FP16 weight tensor into MixLLM's INT4 format.

    MixLLM expects:
        weight_int4   : (k, n//2) uint8 (two INT4 values per byte, row-major)
                        where k = n_out_channels, n = in_features
        weight_scale  : (n // 128, k) fp16 (per-group-of-128 input channels,
                                             one scale per output channel per group)
        weight_zero   : (n // 128, k) uint8 (asymmetric zero points)

    For SHMQ we use SYMMETRIC INT4 quantization (zero=8, range [-7, 7]),
    so weight_zero is constant 8. We keep the asymmetric interface for
    MixLLM compatibility.

    Args:
        weight_fp16: (out_features, in_features) FP16/BF16 weight tensor.

    Returns:
        (weight_int4, weight_scale, weight_zero) — all on the same device
        as `weight_fp16`.
    """
    assert weight_fp16.dim() == 2, f"Expected 2D weight, got {weight_fp16.dim()}D"
    n_out, n_in = weight_fp16.shape
    assert n_in % 128 == 0, f"in_features={n_in} must be divisible by 128 (group_size)"
    device = weight_fp16.device

    # Reshape to (n_in // 128, 128, n_out) for per-group scaling
    w = weight_fp16.float().t().reshape(n_in // 128, 128, n_out)
    # Per-group scale = max(|w|) / 7  (symmetric INT4 range = [-7, 7])
    group_max = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)  # (n_in//128, 1, n_out)
    scale = (group_max / 7.0).squeeze(1)  # (n_in//128, n_out) = (n_groups, n_out)
    # Quantize: round(w / scale), clip to [-8, 7] (INT4 uses [-8, 7] to be safe)
    q = torch.round(w / scale.unsqueeze(1)).clamp(-8, 7)
    # Convert to uint8 representation: add 8 to shift to [0, 15]
    q_uint = (q + 8).to(torch.uint8)  # (n_in//128, 128, n_out)
    # Reshape back to (n_in, n_out) then transpose to (n_out, n_in)
    q_uint = q_uint.reshape(n_in, n_out).t().contiguous()  # (n_out, n_in)
    # Pack two INT4 values into one uint8 byte (little-endian: low nibble first)
    assert q_uint.shape[1] % 2 == 0, "in_features must be even for INT4 packing"
    low  = q_uint[:, 0::2]
    high = q_uint[:, 1::2]
    packed = (high << 4) | low  # (n_out, n_in // 2)

    # MixLLM convention: scale shape = (n_groups, n_out_channels)
    weight_scale = scale.contiguous().to(torch.float16)  # (n_groups, n_out)
    # Zero points: constant 8 (symmetric INT4) — shape (n_groups, n_out)
    weight_zero = torch.full_like(weight_scale, 8, dtype=torch.uint8)

    return packed, weight_scale, weight_zero


def pack_int8_weights(weight_fp16: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pack a (k, n) FP16 weight tensor into MixLLM's INT8 format.

    MixLLM expects:
        weight_int8  : (k, n) int8  (k = n_out_channels, n = in_features)
        weight_scale : (n // 128, k) fp16  (per-group-of-128 input channels,
                                            one scale per output channel per group)

    Returns:
        (weight_int8, weight_scale) — both on the same device as `weight_fp16`.
    """
    n_out, n_in = weight_fp16.shape
    assert n_in % 128 == 0, f"in_features={n_in} must be divisible by 128 (group_size)"
    device = weight_fp16.device

    # Reshape to (n_in // 128, 128, n_out) for per-group scaling
    w = weight_fp16.float().t().reshape(n_in // 128, 128, n_out)
    group_max = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)  # (n_in//128, 1, n_out)
    scale = (group_max / 127.0).squeeze(1)  # (n_in//128, n_out) = (n_groups, n_out)
    q = torch.round(w / scale.unsqueeze(1)).clamp(-128, 127).to(torch.int8)
    q = q.reshape(n_in, n_out).t().contiguous()  # (n_out, n_in)
    # MixLLM convention: scale shape = (n_groups, n_out_channels)
    weight_scale = scale.contiguous().to(torch.float16)  # (n_groups, n_out)
    return q, weight_scale


# ----------------------------------------------------------------------
# Linear module wrapping MixLLM + FP16 path
# ----------------------------------------------------------------------

@dataclass
class SHMQMixLLMConfig:
    """Configuration for the SHMQMixLLMLinear module."""
    in_features: int
    out_features: int
    n_fp16_channels: int = 0
    n_int8_channels: int = 0
    n_int4_channels: int = 0
    group_size: int = 128
    bias: bool = False
    # Original channel order (after permutation, indices into the ORIGINAL
    # weight rows). If None, identity permutation is used.
    permutation: Optional[torch.Tensor] = None


class SHMQMixLLMLinear(nn.Module):
    """Linear layer that combines FP16 + INT8 + INT4 weight paths.

    Weight layout (after permutation):
        [ FP16 channels | INT8 channels | INT4 channels ]

    The first `n_fp16_channels` rows are kept in FP16 and multiplied via
    torch.matmul (cuBLAS). The next `n_int8_channels` rows are quantized
    to INT8, the last `n_int4_channels` rows to INT4, and both go through
    MixLLM's mixed-precision GEMM kernel.

    The forward pass:
        y = x @ W_fp16.T  +  MixLLM_gemm(x_int8, W_int8, W_int4, indices)

    where `x_int8` is the per-group INT8 quantized activation (MixLLM
    handles this internally).

    If MixLLM is not available (e.g. on CPU), falls back to a pure-PyTorch
    dequantized GEMM. This is correct but slow — useful for unit tests.
    """

    def __init__(self, config: SHMQMixLLMConfig, fp16_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.cfg = config
        self.in_features  = config.in_features
        self.out_features = config.out_features
        self.n_fp16 = config.n_fp16_channels
        self.n_int8 = config.n_int8_channels
        self.n_int4 = config.n_int4_channels
        assert self.n_fp16 + self.n_int8 + self.n_int4 == self.out_features, \
            f"Channel counts {self.n_fp16}+{self.n_int8}+{self.n_int4} " \
            f"!= out_features {self.out_features}"

        # Store the permutation (indices into original weight rows)
        if config.permutation is not None:
            self.register_buffer(
                "permutation",
                config.permutation.to(torch.long),
                persistent=True,
            )
        else:
            self.register_buffer(
                "permutation",
                torch.arange(self.out_features, dtype=torch.long),
                persistent=True,
            )

        # FP16 weight path (kept as fp16 for cuBLAS)
        if self.n_fp16 > 0:
            assert fp16_weight is not None, "fp16_weight required when n_fp16 > 0"
            # Slice the first n_fp16 rows of the (already permuted) weight
            w16 = fp16_weight[: self.n_fp16].to(torch.float16).contiguous()
            self.register_buffer("weight_fp16", w16, persistent=True)
        else:
            self.register_buffer("weight_fp16", torch.empty(0), persistent=True)

        # INT8 + INT4 paths go through MixLLM (or PyTorch fallback)
        self._mixllm_linear = None  # will be set in _build_mixllm()

        # SHMQ 3-level fused kernel (preferred on T4 sm_75). Set in _build_shmq_3level().
        self._shmq_3level_kernel = None
        self._fallback_int8_weight = None
        self._fallback_int8_scale = None
        self._fallback_int4_weight = None
        self._fallback_int4_scale = None

        if self.n_int8 + self.n_int4 > 0:
            self._build_mixllm(fp16_weight)

        # Try to build the SHMQ 3-level fused kernel (preferred on T4 sm_75).
        # On failure, falls back to the MixLLM/PyTorch path above.
        try:
            self._build_shmq_3level(fp16_weight)
        except Exception as e:
            logger.info(f"[adapter] SHMQ 3-level kernel unavailable ({e}); "
                        "using MixLLM/PyTorch fallback")

        # Bias
        if config.bias:
            self.register_buffer("bias", torch.zeros(self.out_features, dtype=torch.float16),
                                 persistent=True)
        else:
            self.bias = None

    def _build_mixllm(self, fp16_weight: Optional[torch.Tensor]):
        """Build the MixLLM LinearMixLLM module for the INT8+INT4 slice."""
        assert fp16_weight is not None
        device = fp16_weight.device
        n_total = self.n_int8 + self.n_int4

        # Slice the INT8 and INT4 portions of the permuted weight
        w8  = fp16_weight[self.n_fp16 : self.n_fp16 + self.n_int8].contiguous() \
              if self.n_int8 > 0 else None
        w4  = fp16_weight[self.n_fp16 + self.n_int8 :].contiguous() \
              if self.n_int4 > 0 else None

        # Pack into MixLLM format
        if w8 is not None:
            weight_int8, weight_scale_int8 = pack_int8_weights(w8)
        else:
            weight_int8  = torch.empty(0, dtype=torch.int8,  device=device)
            weight_scale_int8 = torch.empty(0, dtype=torch.float16, device=device)
        if w4 is not None:
            weight_int4, weight_scale_int4, weight_zero_int4 = pack_int4_weights(w4)
        else:
            weight_int4  = torch.empty(0, dtype=torch.uint8, device=device)
            weight_scale_int4 = torch.empty(0, dtype=torch.float16, device=device)
            weight_zero_int4  = torch.empty(0, dtype=torch.uint8, device=device)

        # Store the INT8/INT4 weights as buffers (for fallback path + saving)
        self.register_buffer("weight_int8",  weight_int8,  persistent=True)
        self.register_buffer("weight_int4",  weight_int4,  persistent=True)
        self.register_buffer("weight_scale_int8",  weight_scale_int8,  persistent=True)
        self.register_buffer("weight_scale_int4",  weight_scale_int4,  persistent=True)
        self.register_buffer("weight_zero_int4",   weight_zero_int4,   persistent=True)

        # Indices: since SHMQ permutation already places the INT8 channels
        # first and INT4 channels last, the indices are simple ranges.
        # NOTE: these indices are into the OUTPUT of the MixLLM kernel
        # (which only computes the INT8+INT4 part), NOT into the full layer
        # output. The full layer output is assembled in forward().
        if self.n_int8 > 0:
            indices_int8 = torch.arange(self.n_int8, dtype=torch.int32, device=device)
        else:
            indices_int8 = torch.empty(0, dtype=torch.int32, device=device)
        if self.n_int4 > 0:
            indices_int4 = torch.arange(self.n_int4, dtype=torch.int32, device=device)
        else:
            indices_int4 = torch.empty(0, dtype=torch.int32, device=device)
        self.register_buffer("indices_int8", indices_int8, persistent=True)
        self.register_buffer("indices_int4", indices_int4, persistent=True)

        # Build the MixLLM module if available
        if is_mixllm_available() and device.type == "cuda":
            try:
                from mixllm.nn.modules.linear import LinearMixLLM
                self._mixllm_linear = LinearMixLLM(
                    weight_int8=weight_int8 if self.n_int8 > 0 else None,
                    weight_int4=weight_int4 if self.n_int4 > 0 else None,
                    weight_scale_int8=weight_scale_int8,
                    weight_scale_int4=weight_scale_int4 if self.n_int4 > 0 else None,
                    weight_zero_int4=weight_zero_int4 if self.n_int4 > 0 else None,
                    indices_int8=indices_int8 if self.n_int8 > 0 else None,
                    indices_int4=indices_int4 if self.n_int4 > 0 else None,
                    bias=None,
                )
                logger.info(f"[mixllm] Built LinearMixLLM with "
                            f"{self.n_int8} INT8 + {self.n_int4} INT4 channels.")
            except Exception as e:
                logger.warning(f"[mixllm] Failed to build LinearMixLLM: {e}")
                self._mixllm_linear = None
        else:
            logger.info(f"[mixllm] MixLLM unavailable; using PyTorch fallback.")

        # Always populate the fallback dequantized weights
        if self.n_int8 > 0:
            self._fallback_int8_weight = w8.to(torch.float16)
            self._fallback_int8_scale = weight_scale_int8
        if self.n_int4 > 0:
            self._fallback_int4_weight = w4.to(torch.float16)
            self._fallback_int4_scale = weight_scale_int4

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: combine FP16 + INT8 + INT4 outputs.

        Three execution paths are supported, tried in order:
          1. SHMQ 3-level fused kernel (cupy.RawKernel, sm_75+) — single launch
          2. MixLLM CUDA kernel (sm_80+) — for INT8+INT4 only, FP16 via cuBLAS
          3. PyTorch fallback — dequant + matmul, correctness-only

        Args:
            x: (..., in_features) input tensor.

        Returns:
            y: (..., out_features) output tensor in FP16.
        """
        # Flatten leading dims
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features).to(torch.float16)
        M = x_flat.shape[0]

        # ---- Path 1: SHMQ 3-level fused kernel (preferred on T4) ----
        if self._shmq_3level_kernel is not None:
            y = self._shmq_3level_kernel.forward(x_flat)
            # y is already in permuted order [C16 | C8 | C4]
            # Scatter back to original channel order
            y_original = torch.empty(M, self.out_features, dtype=y.dtype, device=y.device)
            y_original[:, self.permutation] = y
            if self.bias is not None:
                y_original = y_original + self.bias
            return y_original.reshape(*orig_shape[:-1], self.out_features)

        # ---- Path 2/3: split FP16 (cuBLAS) + INT8/INT4 (MixLLM or PT fallback) ----
        outputs: List[torch.Tensor] = []

        # --- FP16 path ---
        if self.n_fp16 > 0:
            y16 = x_flat @ self.weight_fp16.t()  # (M, n_fp16)
            outputs.append(y16)

        # --- INT8 + INT4 path ---
        if self.n_int8 + self.n_int4 > 0:
            y_mix = self._forward_mixllm(x_flat)
            outputs.append(y_mix)

        if len(outputs) == 1:
            y = outputs[0]
        else:
            y = torch.cat(outputs, dim=-1)

        # Scatter back to original channel order
        y_original = torch.empty(M, self.out_features, dtype=y.dtype, device=y.device)
        y_original[:, self.permutation] = y

        if self.bias is not None:
            y_original = y_original + self.bias

        return y_original.reshape(*orig_shape[:-1], self.out_features)

    @torch.no_grad()
    def _build_shmq_3level(self, fp16_weight: Optional[torch.Tensor]):
        """Build the SHMQ 3-level fused kernel wrapper.

        Pulls the quantized weights from the MixLLM wrapper (or fallback buffers)
        and feeds them into SHMQ3LevelKernel. The new kernel (cupy.RawKernel
        + NVRTC for T4 sm_75) takes:
            W16: (N16, K) FP16
            W8:  (N8,  K) INT8
            W4:  (N4,  K/2) UINT8  (packed INT4, lower nibble = even idx)
            S8:  (N8,  K/128) FP16  (per-output-channel, per-group-of-128)
            S4:  (N4,  K/128) FP16

        MixLLM stores scales transposed (n_groups, n_out), so we transpose
        to match our kernel's (n_out, n_groups) layout. MixLLM stores INT4
        already packed as uint8 — we pass it through directly without
        unpacking (the kernel does unpacking in registers).

        If cupy is unavailable or kernel compilation fails, this is a no-op
        and forward() falls back to the MixLLM/PyTorch path.
        """
        from ..inference.shmq_3level_kernel import SHMQ3LevelKernel

        # ---- FP16 path ----
        W16 = self.weight_fp16 if (self.n_fp16 > 0 and self.weight_fp16.numel() > 0) else None

        # ---- INT8 path ----
        W8 = None
        S8 = None
        if self.n_int8 > 0:
            if self.weight_int8 is not None and self.weight_int8.numel() > 0:
                # MixLLM format: weight_int8 (n_int8, n_in) int8
                W8 = self.weight_int8.to(torch.int8).contiguous()
                # MixLLM scale shape (n_groups, n_int8) -> kernel expects (n_int8, n_groups)
                S8 = self.weight_scale_int8.t().contiguous().to(torch.float16)
            elif self._fallback_int8_weight is not None:
                # Fallback stores unpacked FP16 — re-quantize to INT8 here.
                from ..inference.weight_packing import _symmetric_quantize_int
                W8, S8 = _symmetric_quantize_int(
                    self._fallback_int8_weight.to(torch.float16),
                    n_bits=8, group_size=self.cfg.group_size,
                )
                W8 = W8.contiguous()
                S8 = S8.contiguous().to(torch.float16)

        # ---- INT4 path ----
        # Pass the packed uint8 directly to the kernel — it unpacks in registers.
        W4_packed = None
        S4 = None
        if self.n_int4 > 0:
            if self.weight_int4 is not None and self.weight_int4.numel() > 0:
                # MixLLM format: weight_int4 (n_int4, n_in//2) uint8 — already packed.
                # Verify packing convention: MixLLM uses (high << 4) | low with
                # low at even index, high at odd index — matches our kernel.
                W4_packed = self.weight_int4.to(torch.uint8).contiguous()
                # MixLLM scale shape (n_groups, n_int4) -> (n_int4, n_groups)
                if self.weight_scale_int4 is not None and self.weight_scale_int4.numel() > 0:
                    S4 = self.weight_scale_int4.t().contiguous().to(torch.float16)
            elif self._fallback_int4_weight is not None:
                # Fallback stores unpacked FP16 — pack to INT4 here.
                from ..inference.weight_packing import _symmetric_quantize_int, pack_int4
                codes4, S4 = _symmetric_quantize_int(
                    self._fallback_int4_weight.to(torch.float16),
                    n_bits=4, group_size=self.cfg.group_size,
                )
                W4_packed = pack_int4(codes4).contiguous()
                S4 = S4.contiguous().to(torch.float16)

        # ---- Build the kernel wrapper ----
        # If everything is on CPU, the kernel will use the PyTorch fallback path
        # (correctness-only). On CUDA, it will use the cupy.RawKernel.
        K = self.in_features
        N = self.out_features
        try:
            self._shmq_3level_kernel = SHMQ3LevelKernel(
                W16=W16,
                W8=W8,
                W4_packed=W4_packed,
                S8=S8,
                S4=S4,
                K=K, N=N,
                N16=self.n_fp16, N8=self.n_int8, N4=self.n_int4,
                group_size=self.cfg.group_size,
            )
            if self._shmq_3level_kernel.is_cuda_native:
                logger.info(
                    f"[adapter] SHMQ 3-level fused kernel active: "
                    f"N16={self.n_fp16}, N8={self.n_int8}, N4={self.n_int4}, "
                    f"K={K} (single-launch cupy.RawKernel on sm_75+)"
                )
        except Exception as e:
            logger.warning(f"[adapter] Failed to build SHMQ 3-level kernel: {e}")
            self._shmq_3level_kernel = None

    def _forward_mixllm(self, x_flat: torch.Tensor) -> torch.Tensor:
        """Forward through the MixLLM INT8+INT4 path (or PyTorch fallback)."""
        if self._mixllm_linear is not None:
            # MixLLM expects input on CUDA
            x_cuda = x_flat.cuda() if not x_flat.is_cuda else x_flat
            try:
                y_mix = self._mixllm_linear(x_cuda)
                # MixLLM returns column-major output if both INT8 and INT4 are present
                # (we handle the transpose in forward() via cat).
                return y_mix if y_mix.dim() == 2 else y_mix.reshape(-1, self.n_int8 + self.n_int4)
            except Exception as e:
                logger.warning(f"[mixllm] Forward failed ({e}); falling back to PyTorch.")
                return self._forward_mixllm_fallback(x_flat)
        else:
            return self._forward_mixllm_fallback(x_flat)

    @torch.no_grad()
    def _forward_mixllm_fallback(self, x_flat: torch.Tensor) -> torch.Tensor:
        """Pure-PyTorch fallback for MixLLM (correct but slow)."""
        # Dequantize INT8 weights
        n_in = self.in_features
        gs = self.cfg.group_size
        n_groups = n_in // gs

        outputs_mix: List[torch.Tensor] = []

        if self.n_int8 > 0:
            # weight_int8: (n_int8, n_in) int8
            # weight_scale_int8: (n_groups, n_int8) fp16  [MixLLM convention]
            # We need to dequantize: w_deq[i, g*gs:(g+1)*gs] = w_int8[i, g*gs:(g+1)*gs] * scale[g, i]
            w8 = self.weight_int8.to(torch.float16).view(self.n_int8, n_groups, gs)
            # Transpose scale so it broadcasts: (n_groups, n_int8) -> (n_int8, n_groups, 1)
            s8 = self.weight_scale_int8.t().unsqueeze(-1)  # (n_int8, n_groups, 1)
            w8_deq = (w8 * s8).reshape(self.n_int8, n_in)
            y8 = x_flat @ w8_deq.t()  # (M, n_int8)
            outputs_mix.append(y8)

        if self.n_int4 > 0:
            # weight_int4: (n_int4, n_in // 2) uint8 (packed)
            # Unpack to (n_int4, n_in) int8
            low  = (self.weight_int4 & 0x0F)
            high = (self.weight_int4 >> 4) & 0x0F
            w4_unpacked = torch.stack([low, high], dim=-1).reshape(self.n_int4, n_in)
            # Subtract zero point (8) to get signed INT4 in [-8, 7]
            w4_signed = (w4_unpacked.to(torch.float16) - 8.0)
            w4 = w4_signed.view(self.n_int4, n_groups, gs)
            # weight_scale_int4: (n_groups, n_int4) -> (n_int4, n_groups, 1)
            s4 = self.weight_scale_int4.t().unsqueeze(-1)
            w4_deq = (w4 * s4).reshape(self.n_int4, n_in)
            y4 = x_flat @ w4_deq.t()  # (M, n_int4)
            outputs_mix.append(y4)

        if len(outputs_mix) == 1:
            return outputs_mix[0]
        return torch.cat(outputs_mix, dim=-1)

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"n_fp16={self.n_fp16}, n_int8={self.n_int8}, n_int4={self.n_int4}, "
                f"bias={self.bias is not None}")


# ----------------------------------------------------------------------
# Conversion utility: replace nn.Linear with SHMQMixLLMLinear
# ----------------------------------------------------------------------

@dataclass
class ConversionSummary:
    n_layers_converted: int
    n_fp16_channels_total: int
    n_int8_channels_total: int
    n_int4_channels_total: int
    per_layer_summary: Dict[str, Dict[str, int]] = field(default_factory=dict)
    mixllm_available: bool = False

    def __str__(self) -> str:
        return (
            f"Conversion summary:\n"
            f"  Layers converted: {self.n_layers_converted}\n"
            f"  Total FP16 channels: {self.n_fp16_channels_total}\n"
            f"  Total INT8 channels: {self.n_int8_channels_total}\n"
            f"  Total INT4 channels: {self.n_int4_channels_total}\n"
            f"  MixLLM kernel available: {self.mixllm_available}\n"
        )


def convert_linear_to_mixllm(
    layer: nn.Linear,
    bit_allocation_for_layer: int,
    permutation_indices: Optional[torch.Tensor] = None,
    cluster_sizes: Optional[Dict[int, int]] = None,
    intra_layer_hp_ratio_8: float = 0.20,
    intra_layer_hp_ratio_16: float = 0.05,
    group_size: int = 128,
) -> SHMQMixLLMLinear:
    """Convert a single nn.Linear to a SHMQMixLLMLinear.

    Args:
        layer: the source nn.Linear.
        bit_allocation_for_layer: 4, 8, or 16 (inter-layer bit allocation).
            If 16: all channels are FP16.
            If 8: all INT8 channels, no FP16/INT4.
            If 4: split into C8/C4 based on intra_layer_hp_ratio_8.
            (Default SHMQ behavior: 4-bit layers have a small C8 cluster
             for the most-sensitive channels.)
        permutation_indices: (out_features,) LongTensor mapping
            permuted_position -> original_channel. If None, identity.
        cluster_sizes: optional {16: k16, 8: k8, 4: k4} dict (overrides
            intra_layer_hp_ratio_*). If None, computed from ratios.
        intra_layer_hp_ratio_8: fraction of channels to keep at INT8
            within a 4-bit layer (only used if cluster_sizes is None
            and bit_allocation_for_layer == 4).
        intra_layer_hp_ratio_16: fraction of channels to keep at FP16
            within a 4-bit layer (only used if cluster_sizes is None
            and bit_allocation_for_layer == 4).
        group_size: quantization group size (default 128).

    Returns:
        A SHMQMixLLMLinear module with weights packed and ready for inference.
    """
    weight = layer.weight.data  # (out_features, in_features)
    n_out, n_in = weight.shape
    bias = layer.bias is not None

    # Apply permutation to the weight rows
    if permutation_indices is not None:
        # permutation_indices[i] = original_channel_at_position_i
        # So we want: permuted_weight[i] = weight[permutation_indices[i]]
        weight_permuted = weight[permutation_indices.to(weight.device)].contiguous()
    else:
        weight_permuted = weight

    # Determine cluster sizes
    if cluster_sizes is not None:
        k16 = cluster_sizes.get(16, 0)
        k8  = cluster_sizes.get(8, 0)
        k4  = cluster_sizes.get(4, 0)
        assert k16 + k8 + k4 == n_out, \
            f"Cluster sizes {k16}+{k8}+{k4} != out_features {n_out}"
    elif bit_allocation_for_layer == 16:
        k16, k8, k4 = n_out, 0, 0
    elif bit_allocation_for_layer == 8:
        k16, k8, k4 = 0, n_out, 0
    elif bit_allocation_for_layer == 4:
        k16 = int(n_out * intra_layer_hp_ratio_16)
        k8  = int(n_out * intra_layer_hp_ratio_8)
        k4  = n_out - k16 - k8
    else:
        raise ValueError(f"Unsupported bit_allocation: {bit_allocation_for_layer}")

    cfg = SHMQMixLLMConfig(
        in_features=n_in,
        out_features=n_out,
        n_fp16_channels=k16,
        n_int8_channels=k8,
        n_int4_channels=k4,
        group_size=group_size,
        bias=bias,
        permutation=permutation_indices,
    )

    new_module = SHMQMixLLMLinear(cfg, fp16_weight=weight_permuted)
    if bias:
        new_module.bias = layer.bias.data.to(torch.float16).clone()

    return new_module


def convert_model_to_mixllm(
    model: nn.Module,
    layer_names: List[str],
    bit_allocation: Dict[str, int],
    permutation_indices: Dict[str, torch.Tensor],
    cluster_sizes: Optional[Dict[str, Dict[int, int]]] = None,
    intra_layer_hp_ratio_8: float = 0.20,
    intra_layer_hp_ratio_16: float = 0.05,
    group_size: int = 128,
    verbose: bool = False,
) -> ConversionSummary:
    """Replace every nn.Linear in `layer_names` with a SHMQMixLLMLinear.

    Args:
        model: the model to convert in-place.
        layer_names: list of layer names (dotted module paths) to convert.
        bit_allocation: {layer_name: 4 | 8 | 16}.
        permutation_indices: {layer_name: LongTensor[out_features]}.
        cluster_sizes: optional {layer_name: {16: k16, 8: k8, 4: k4}}.
        intra_layer_hp_ratio_8, _16: fallback ratios when cluster_sizes is None.
        group_size: quantization group size.
        verbose: print per-layer info.

    Returns:
        ConversionSummary with statistics.
    """
    from ..utils import get_module_by_name, get_parent_module_and_attr

    total_fp16 = total_int8 = total_int4 = 0
    per_layer: Dict[str, Dict[str, int]] = {}
    n_converted = 0

    for name in layer_names:
        parent, attr = get_parent_module_and_attr(model, name)
        if parent is None or not hasattr(parent, attr):
            logger.warning(f"[mixllm] Layer {name} not found in model; skipping.")
            continue
        layer = getattr(parent, attr)
        if not isinstance(layer, nn.Linear):
            logger.warning(f"[mixllm] Layer {name} is not nn.Linear ({type(layer)}); skipping.")
            continue

        bits = bit_allocation.get(name, 4)
        perm = permutation_indices.get(name)
        cs = cluster_sizes.get(name) if cluster_sizes else None

        new_module = convert_linear_to_mixllm(
            layer,
            bit_allocation_for_layer=bits,
            permutation_indices=perm,
            cluster_sizes=cs,
            intra_layer_hp_ratio_8=intra_layer_hp_ratio_8,
            intra_layer_hp_ratio_16=intra_layer_hp_ratio_16,
            group_size=group_size,
        )
        setattr(parent, attr, new_module)
        n_converted += 1
        per_layer[name] = {
            "bits": bits,
            "n_fp16": new_module.n_fp16,
            "n_int8": new_module.n_int8,
            "n_int4": new_module.n_int4,
        }
        total_fp16 += new_module.n_fp16
        total_int8 += new_module.n_int8
        total_int4 += new_module.n_int4

        if verbose:
            logger.info(
                f"[mixllm] {name}: bits={bits}, "
                f"fp16={new_module.n_fp16}, int8={new_module.n_int8}, int4={new_module.n_int4}"
            )

    return ConversionSummary(
        n_layers_converted=n_converted,
        n_fp16_channels_total=total_fp16,
        n_int8_channels_total=total_int8,
        n_int4_channels_total=total_int4,
        per_layer_summary=per_layer,
        mixllm_available=is_mixllm_available(),
    )
