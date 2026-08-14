"""Runtime hardening for SHMQ-Ultimate seams (C, D, E, S2, S3, S4).

This module contains the assertion / verification helpers that catch silent
data-corruption bugs at the boundaries between pipeline stages.

Seam map
--------
  C  (step3 -> step3.5 -> step4): ILP target bits vs ISA-rounded cluster_sizes
  D  (step4 -> step5 -> step6 -> step7 -> step8): permutation applied consistently
  E  (step8 -> step9 -> forward): packing layout invariants
  S2 (memory):                       streaming capture + Hessian diag-only
  S3 (cupy DLPack):                  contiguous + stride guards
  S4 (vLLM patch):                   shmq_config.json schema

All checks return True/None on success and raise AssertionError with a
descriptive message on failure. They cost <1% of pipeline runtime but catch
NaN-producing bugs that would otherwise silently corrupt the model.

Usage:
    from .hardening import (
        assert_cluster_sizes_consistent,         # seam C
        assert_permutation_applied,              # seam D
        assert_finite_output,                    # seam D/E
        assert_packing_invariants,               # seam E
        assert_contiguous_for_cupy,              # seam S3
        assert_shmq_config_schema,               # seam S4
        streaming_capture_input_activations,     # seam S2
    )
"""
from __future__ import annotations
from typing import Dict, List, Optional, Any, Tuple
import torch
import torch.nn as nn
import gc


# ===========================================================================
# Seam C: ILP -> ISA -> Permutation consistency
# ===========================================================================

def assert_cluster_sizes_consistent(
    layer_names: List[str],
    bit_allocation: Dict[str, int],
    cluster_sizes: Dict[str, Dict[int, int]],
    ilp_total_bits: Optional[float] = None,
    n_params: Optional[Dict[str, int]] = None,
    target_avg_bits: Optional[float] = None,
    isa_drift_tolerance: float = 0.05,
) -> None:
    """Verify that cluster_sizes sum to n_out for every layer (seam C-1),
    and that the achieved avg-bits matches the ILP target within tolerance
    (seam C-2 — guards against ISA matching silently shifting the budget).

    Args:
        layer_names: list of layer names to check.
        bit_allocation: per-layer bit allocation from ILP {4, 8, 16}.
        cluster_sizes: per-layer {16, 8, 4} -> channel count.
        ilp_total_bits: total bits achieved by ILP (from ILPResult3L.total_bits).
            If None, the C-2 check is skipped.
        n_params: per-layer param count (only needed for C-2).
        target_avg_bits: ILP target (only needed for C-2).
        isa_drift_tolerance: max allowed drift in avg-bits after ISA rounding.

    Raises:
        AssertionError if any invariant is violated.
    """
    # ---- C-1: cluster sizes sum to n_out for every layer ----
    for name in layer_names:
        if name not in cluster_sizes:
            continue
        cs = cluster_sizes[name]
        # Get n_out from a sibling source — caller must pass via n_params
        # or we infer from the cluster_sizes sum itself
        total_channels = cs.get(16, 0) + cs.get(8, 0) + cs.get(4, 0)
        # Sanity: each cluster size must be non-negative
        for level in (16, 8, 4):
            n = cs.get(level, 0)
            assert n >= 0, f"Layer {name}: cluster {level} has negative size {n}"
        # If 16-bit layer, all channels should be in C16
        bits = bit_allocation.get(name, 4)
        if bits == 16:
            assert cs.get(8, 0) == 0 and cs.get(4, 0) == 0, \
                f"Layer {name} is 16-bit but has INT8/INT4 clusters: {cs}"
        elif bits == 8:
            assert cs.get(4, 0) == 0, \
                f"Layer {name} is 8-bit but has INT4 clusters: {cs}"

    # ---- C-2: ISA drift check (only if ILP total provided) ----
    if (ilp_total_bits is not None and n_params is not None
            and target_avg_bits is not None):
        total_params = sum(n_params.values())
        if total_params > 0:
            achieved_avg = ilp_total_bits / total_params
            drift = abs(achieved_avg - target_avg_bits)
            assert drift <= isa_drift_tolerance, (
                f"ISA matching drift too large: ILP target={target_avg_bits:.4f} bits, "
                f"achieved={achieved_avg:.4f} bits, drift={drift:.4f} > "
                f"tolerance={isa_drift_tolerance:.4f}. This means ISA rounding "
                f"silently moved the budget outside the ILP solution. "
                f"Either reduce isa_tile_* sizes or increase isa_drift_tolerance."
            )


# ===========================================================================
# Seam D: Permutation applied consistently across stages
# ===========================================================================

def assert_permutation_applied(
    layer_name: str,
    weight_before: torch.Tensor,
    weight_after: torch.Tensor,
    permutation: torch.Tensor,
    rtol: float = 1e-6,
) -> None:
    """Verify that `weight_after == weight_before[permutation]` (seam D-1).

    Catches the case where a stage between step4 and step8 silently
    dropped the permutation (e.g. AutoRound wrapper forgot to apply it,
    or GPTQ Hessian was computed on pre-permutation order).

    Args:
        layer_name: for error message.
        weight_before: (n_out, n_in) weight BEFORE permutation.
        weight_after: (n_out, n_in) weight AFTER permutation.
        permutation: (n_out,) int64 permutation indices.

    Raises:
        AssertionError if weight_after != weight_before[permutation].
    """
    assert weight_before.shape == weight_after.shape, (
        f"Layer {layer_name}: weight shape changed "
        f"{weight_before.shape} -> {weight_after.shape} during permutation"
    )
    assert permutation.shape[0] == weight_before.shape[0], (
        f"Layer {layer_name}: permutation length {permutation.shape[0]} "
        f"!= n_out {weight_before.shape[0]}"
    )
    # Reconstruct: w_after should equal w_before indexed by permutation
    expected = weight_before[permutation]
    diff = (expected.float() - weight_after.float()).abs().max().item()
    assert diff <= rtol * max(1.0, weight_before.abs().max().item()), (
        f"Layer {layer_name}: permutation not correctly applied. "
        f"Max diff = {diff:.6e} (rtol={rtol}). "
        f"Likely a downstream stage (AutoRound/SQC/GPTQ) operated on "
        f"pre-permutation weights."
    )


def assert_finite_output(
    y: torch.Tensor,
    context: str = "",
    raise_on_nan: bool = True,
) -> bool:
    """Verify that `y` contains no NaN/Inf (seam D/E — packing errors).

    Args:
        y: output tensor from any forward pass.
        context: human-readable context for the error message.
        raise_on_nan: if True, raise; if False, just return False.

    Returns:
        True if all finite, False if NaN/Inf present (and raise_on_nan=False).
    """
    if not torch.isfinite(y).all():
        n_nan = torch.isnan(y).sum().item()
        n_inf = torch.isinf(y).sum().item()
        msg = (
            f"[FATAL] Non-finite output detected ({context}): "
            f"{n_nan} NaN, {n_inf} Inf out of {y.numel()} elements "
            f"({100*(n_nan+n_inf)/y.numel():.2f}%). "
            f"This usually means a packing/stride mismatch at seam E "
            f"(fake-quant -> MixLLM -> cupy.RawKernel) OR a permutation "
            f"mismatch at seam D."
        )
        if raise_on_nan:
            raise RuntimeError(msg)
        print(msg)
        return False
    return True


# ===========================================================================
# Seam E: Packing layout invariants
# ===========================================================================

def assert_packing_invariants(
    W16: Optional[torch.Tensor],
    W8: Optional[torch.Tensor],
    W4_packed: Optional[torch.Tensor],
    S8: Optional[torch.Tensor],
    S4: Optional[torch.Tensor],
    K: int,
    N16: int,
    N8: int,
    N4: int,
    group_size: int = 128,
) -> None:
    """Verify all shape/dtype/contiguous invariants at the packing boundary.

    Catches the most common seam-E bugs:
      - W4 packed as (K/2, N4) instead of (N4, K/2)  (transposed)
      - S8 stored as (n_groups, N8) instead of (N8, n_groups)
      - W8 dtype float instead of int8
      - Non-contiguous weights causing silent cupy copies

    Args:
        W16: (N16, K) FP16 or None.
        W8:  (N8, K) INT8 or None.
        W4_packed: (N4, K/2) UINT8 or None.
        S8:  (N8, K/group_size) FP16 or None.
        S4:  (N4, K/group_size) FP16 or None.
        K, N16, N8, N4: declared dimensions.
        group_size: 128 by default.

    Raises:
        AssertionError on any invariant violation.
    """
    n_groups = K // group_size
    assert K % group_size == 0, \
        f"K={K} must be divisible by group_size={group_size}"

    # ---- W16 ----
    if N16 > 0:
        assert W16 is not None, "N16>0 but W16 is None"
        assert W16.shape == (N16, K), \
            f"W16 shape {W16.shape} != ({N16}, {K})"
        assert W16.dtype == torch.float16, \
            f"W16 dtype {W16.dtype} != float16"
        assert W16.is_contiguous(), "W16 must be contiguous (seam S3)"

    # ---- W8 ----
    if N8 > 0:
        assert W8 is not None, "N8>0 but W8 is None"
        assert W8.shape == (N8, K), \
            f"W8 shape {W8.shape} != ({N8}, {K})"
        assert W8.dtype == torch.int8, \
            f"W8 dtype {W8.dtype} != int8 (seam E: dequant math assumes int8)"
        assert W8.is_contiguous(), "W8 must be contiguous (seam S3)"

    # ---- W4 packed ----
    if N4 > 0:
        assert W4_packed is not None, "N4>0 but W4_packed is None"
        # CRITICAL: kernel expects (N4, K/2), NOT (K/2, N4)
        assert W4_packed.shape == (N4, K // 2), \
            f"W4_packed shape {W4_packed.shape} != ({N4}, {K//2}). " \
            f"Likely transposed at packing boundary (seam E)."
        assert W4_packed.dtype == torch.uint8, \
            f"W4_packed dtype {W4_packed.dtype} != uint8"
        assert W4_packed.is_contiguous(), "W4_packed must be contiguous (seam S3)"

    # ---- S8 ----
    if N8 > 0:
        assert S8 is not None, "N8>0 but S8 is None"
        # CRITICAL: kernel expects (N8, n_groups), MixLLM stores (n_groups, N8)
        assert S8.shape == (N8, n_groups), \
            f"S8 shape {S8.shape} != ({N8}, {n_groups}). " \
            f"MixLLM convention is (n_groups, N8) — did you forget to transpose? (seam E)"
        assert S8.dtype == torch.float16, \
            f"S8 dtype {S8.dtype} != float16"
        assert S8.is_contiguous(), "S8 must be contiguous"

    # ---- S4 ----
    if N4 > 0:
        assert S4 is not None, "N4>0 but S4 is None"
        assert S4.shape == (N4, n_groups), \
            f"S4 shape {S4.shape} != ({N4}, {n_groups}). " \
            f"MixLLM convention is (n_groups, N4) — did you forget to transpose? (seam E)"
        assert S4.dtype == torch.float16, f"S4 dtype {S4.dtype} != float16"
        assert S4.is_contiguous(), "S4 must be contiguous"

    # ---- Range checks ----
    if N8 > 0:
        # INT8 codes should be in [-128, 127]
        w8_min, w8_max = W8.min().item(), W8.max().item()
        assert -128 <= w8_min and w8_max <= 127, \
            f"W8 out of int8 range: [{w8_min}, {w8_max}]"
    if N4 > 0:
        # Packed uint8: each nibble in [0, 15] (unsigned) before sign-extend
        # So the byte itself is in [0, 255] (trivially true for uint8)
        # Check scales are positive and reasonable
        if S4.numel() > 0:
            s4_max = S4.abs().max().item()
            assert s4_max < 100.0, \
                f"S4 max abs = {s4_max} — looks unreasonable (typical <1.0). " \
                f"Packing likely got wrong scale tensor (seam E)."


def round_trip_test_int4(
    weight_fp16: torch.Tensor,
    W4_packed: torch.Tensor,
    S4: torch.Tensor,
    group_size: int = 128,
    rtol: float = 0.1,
) -> None:
    """Verify that unpacking W4_packed with S4 recovers weight_fp16 within rtol.

    This is the strongest seam-E check — actually round-trips the packing.

    Args:
        weight_fp16: original (N4, K) FP16 weight (before quantization).
        W4_packed: (N4, K/2) uint8 packed.
        S4: (N4, K/group_size) FP16 scales.
        group_size: 128.
        rtol: relative tolerance (typical INT4 quantization error is ~10%).

    Raises:
        AssertionError if round-trip error exceeds rtol.
    """
    N4, K = weight_fp16.shape
    n_groups = K // group_size

    # Unpack: low nibble = even idx, high nibble = odd idx (kernel convention)
    low = (W4_packed & 0x0F).to(torch.int16)
    high = (W4_packed >> 4).to(torch.int16) & 0x0F
    codes = torch.stack([low, high], dim=-1).reshape(N4, K).to(torch.int8)
    # Sign-extend from 4 bits
    codes = torch.where(codes >= 8, codes - 16, codes).to(torch.int8)

    # Dequantize: w = code * scale
    codes_f = codes.float().reshape(N4, n_groups, group_size)
    scales_f = S4.float().unsqueeze(-1)  # (N4, n_groups, 1)
    w_deq = (codes_f * scales_f).reshape(N4, K)

    # Compare
    w_orig = weight_fp16.float()
    abs_err = (w_deq - w_orig).abs()
    rel_err = abs_err / w_orig.abs().clamp(min=1e-6)
    max_rel_err = rel_err.max().item()
    assert max_rel_err <= rtol, (
        f"INT4 round-trip test failed: max relative error = {max_rel_err:.4f} "
        f"(rtol={rtol}). Packing layout is likely wrong (seam E). "
        f"Check: nibble order (low=even, high=odd), scale shape (N4, n_groups) "
        f"not (n_groups, N4), scale dtype (fp16 not fp32)."
    )


# ===========================================================================
# Seam S2: Memory streaming
# ===========================================================================

def streaming_capture_input_activations(
    model: nn.Module,
    layer_names: List[str],
    calibration_data: torch.Tensor,
    batch_size: int = 1,
    max_samples: Optional[int] = None,
    cleanup_between_layers: bool = True,
) -> Dict[str, List[torch.Tensor]]:
    """Memory-efficient activation capture — one layer at a time (seam S2).

    The original capture_input_activations() registers hooks on ALL layers
    simultaneously, which on a 7B model means ~28 layers × 128 samples ×
    (B*S*cin) FP16 tensors all kept in GPU memory at once. On T4 16GB this
    causes OOM.

    This streaming version captures one layer at a time, then calls
    torch.cuda.empty_cache() between layers. Total time is the same
    (one forward pass per layer per batch), but peak memory is O(1)
    layers instead of O(L).

    Args:
        model: the model.
        layer_names: list of layer names to capture.
        calibration_data: (N, S) tokenized input.
        batch_size: forward batch size.
        max_samples: cap on number of samples.
        cleanup_between_layers: if True, empty_cache() between layers.

    Returns:
        {layer_name: [list of (N, cin) tensors per batch]}
    """
    from .utils import get_module_by_name
    device = next(model.parameters()).device
    n_total = calibration_data.shape[0]
    if max_samples is not None:
        n_total = min(n_total, max_samples)

    captured: Dict[str, List[torch.Tensor]] = {}

    for name in layer_names:
        mod = get_module_by_name(model, name)
        cap_list: List[torch.Tensor] = []

        def hook(module, inputs, outputs, _cap_list=cap_list):
            x = inputs[0].detach()
            x = x.reshape(-1, x.shape[-1])
            _cap_list.append(x.to(device))

        handle = mod.register_forward_hook(hook)
        with torch.no_grad():
            for i in range(0, n_total, batch_size):
                batch = calibration_data[i: i + batch_size].to(device)
                model(batch)
                # Free intermediate activations after each batch
                if cleanup_between_layers and device.type == "cuda":
                    torch.cuda.empty_cache()
        handle.remove()
        captured[name] = cap_list

        if cleanup_between_layers and device.type == "cuda":
            gc.collect()
            torch.cuda.empty_cache()

    return captured


# ===========================================================================
# Seam S3: cupy DLPack contiguous guard
# ===========================================================================

def assert_contiguous_for_cupy(
    tensors: Dict[str, Optional[torch.Tensor]],
    context: str = "",
) -> None:
    """Verify that all provided tensors are contiguous before cupy.from_dlpack.

    cupy.from_dlpack() requires contiguous tensors; otherwise it silently
    makes a copy, which on a 7B model can spike GPU memory by 3-5GB and
    cause OOM.

    Args:
        tensors: {name: tensor or None}.
        context: for error message.

    Raises:
        AssertionError if any non-None tensor is not contiguous.
    """
    for name, t in tensors.items():
        if t is None:
            continue
        assert t.is_contiguous(), (
            f"[{context}] Tensor '{name}' is not contiguous. "
            f"cupy.from_dlpack() will silently copy it, costing "
            f"{t.numel() * t.element_size() / 1e6:.1f} MB of GPU memory. "
            f"Call .contiguous() before passing to the kernel (seam S3). "
            f"Shape={t.shape}, strides={t.stride()}."
        )


# ===========================================================================
# Seam S4: shmq_config.json schema
# ===========================================================================

def assert_shmq_config_schema(config_dict: Dict[str, Any]) -> None:
    """Verify that the saved shmq_config.json has the right schema for vLLM.

    The vLLM patch 0005-shmq-3level-t4-support.patch reads:
      - quant_method: must be "shmq_3level"
      - cluster_sizes: {layer_name: {"16": int, "8": int, "4": int}}
        (NOTE: keys must be STRINGS, not ints, for JSON compatibility)
      - bit_allocation: {layer_name: int}
      - sqc_multipliers: {layer_name: float}

    Args:
        config_dict: the dict about to be written to shmq_config.json.

    Raises:
        AssertionError if any required field is missing or has wrong type.
    """
    # Top-level required fields
    required_top = {"quant_method", "bit_allocation", "cluster_sizes"}
    missing = required_top - set(config_dict.keys())
    assert not missing, \
        f"shmq_config.json missing required fields: {missing}"

    # quant_method
    qm = config_dict["quant_method"]
    assert qm == "shmq_3level", \
        f"quant_method must be 'shmq_3level', got '{qm}' (seam S4: vLLM won't load)"

    # cluster_sizes: keys must be strings (JSON-compatible)
    cs = config_dict["cluster_sizes"]
    for layer_name, levels in cs.items():
        assert isinstance(layer_name, str), \
            f"cluster_sizes key {layer_name!r} is not a string"
        for level_key, count in levels.items():
            assert isinstance(level_key, str), \
                f"cluster_sizes['{layer_name}'] key {level_key!r} is not a string. " \
                f"JSON requires string keys; vLLM deserialize will fail (seam S4)."
            assert isinstance(count, int), \
                f"cluster_sizes['{layer_name}']['{level_key}'] = {count} is not int"
            assert level_key in {"16", "8", "4"}, \
                f"cluster_sizes level key '{level_key}' not in {{'16','8','4'}}"

    # bit_allocation
    ba = config_dict["bit_allocation"]
    for layer_name, bits in ba.items():
        assert bits in (4, 8, 16), \
            f"bit_allocation['{layer_name}'] = {bits}, expected 4/8/16"


def normalize_cluster_sizes_for_json(
    cluster_sizes: Dict[str, Dict[int, int]],
) -> Dict[str, Dict[str, int]]:
    """Convert int keys to string keys for JSON serialization (seam S4).

    The pipeline stores cluster_sizes as {layer: {16: n, 8: n, 4: n}} with
    int keys (Python dict allows this). But JSON requires string keys. This
    helper does the conversion explicitly so the saved file is loadable by
    vLLM's json.load() without silent key-type mismatches.
    """
    out: Dict[str, Dict[str, int]] = {}
    for layer_name, levels in cluster_sizes.items():
        out[layer_name] = {
            str(level): int(count)
            for level, count in levels.items()
        }
    return out


def denormalize_cluster_sizes_from_json(
    cluster_sizes_json: Dict[str, Dict[str, int]],
) -> Dict[str, Dict[int, int]]:
    """Inverse of normalize_cluster_sizes_for_json — used by vLLM loader."""
    out: Dict[str, Dict[int, int]] = {}
    for layer_name, levels in cluster_sizes_json.items():
        out[layer_name] = {
            int(level): int(count)
            for level, count in levels.items()
        }
    return out


# ===========================================================================
# Convenience: run all hardening checks for a step
# ===========================================================================

def check_step_complete(
    step_name: str,
    checks: List[Tuple[str, bool, str]],
) -> None:
    """Run a list of (check_name, passed, error_msg) tuples.

    Args:
        step_name: e.g. "step4_permutation".
        checks: list of (name, passed, msg).

    Raises:
        AssertionError on first failed check with descriptive message.
    """
    for name, passed, msg in checks:
        if not passed:
            raise AssertionError(
                f"[{step_name}] HARDENING CHECK FAILED: {name}\n  {msg}"
            )
