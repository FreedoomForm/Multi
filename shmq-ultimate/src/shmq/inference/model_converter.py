"""Model converter: replace nn.Linear with SHMQQuantLinear for REAL INT4 inference.

After the SHMQ pipeline (steps 0-8) the model has:
  - Smoothed weights (SmoothQuant)
  - Per-channel permutation applied to weight columns (decoupled permutation)
  - RMSNorm with fused permutation
  - Fake-quantized weights (FP16 storage with quant→dequant once)

This module swaps each fake-quantized nn.Linear with a SHMQQuantLinear that
stores REAL packed INT4/INT8 codes and dispatches to the custom CUDA kernel
(or CPU fallback) at inference time. After this conversion:
  - Memory footprint drops by ~3.3× (4.8 bits vs 16 bits per weight)
  - Inference uses the SHMQ parallel two-bit matmul kernel (no dequant overhead)
  - Throughput increases by ~2.86× (per SHMQ paper Table 3)

Conversion logic:
  For each Linear layer L with weight W (cout, cin):
    bits = bit_allocation[L_name]        # 4 or 8 (from ILP / step 3)
    if bits == 8:
        K = cin                          # entire layer is INT8 (inter-layer W8A8)
    else: # bits == 4
        K = round(cin * intra_layer_hp_ratio)  # intra-layer mix: top K channels as INT8
        K = round_up_to_group_size(K)          # ensure K is a multiple of group_size
        K = clamp(K, 0, cin - group_size)      # ensure both halves have >=1 group
    SHMQQuantLinear.from_weight(W, n_sensitive=K, group_size, bias, perm)
    replace_module(model, L_name, shmq_linear)

The permutation has ALREADY been applied to W in step 4 (decoupled permutation),
so the first K columns of W are the sensitive cluster (Csen) and the remaining
(cin - K) are the insensitive cluster (Cinsen).
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import time
import torch
import torch.nn as nn

from ..utils import get_module_by_name, set_module_by_name
from .shmq_quant_linear import SHMQQuantLinear
from .weight_packing import pack_shmq_linear


def _round_to_group(x: int, g: int) -> int:
    """Round x up to the nearest multiple of g."""
    return ((x + g - 1) // g) * g


def convert_model_to_real_int4(
    model: nn.Module,
    layer_names: List[str],
    bit_allocation: Dict[str, int],
    permutation_indices: Optional[Dict[str, torch.Tensor]] = None,
    group_size: int = 128,
    intra_layer_hp_ratio: float = 0.125,
    verbose: bool = True,
) -> Dict[str, Dict]:
    """Replace every nn.Linear in `layer_names` with a SHMQQuantLinear.

    Args:
        model: HuggingFace LLM, already through SHMQ steps 0-8 (fake-quantized
               weights with permutation already applied to weight columns).
        layer_names: list of dotted module names to convert (must be nn.Linear).
        bit_allocation: {name: 4 or 8} from ILP (step 3).
        permutation_indices: {name: (cin,) tensor} from step 4 (kept for debug).
        group_size: 128 (must match what was used during SHMQ quantization).
        intra_layer_hp_ratio: fraction of channels to keep at INT8 inside a
                4-bit layer (the SHMQ paper's U_b = 0.125 default).
        verbose: print progress.

    Returns:
        {name: {"n_sensitive": K, "n_insensitive": cin-K, "bits": int,
                 "avg_bits": float, "params": int}}
    """
    if verbose:
        print("\n" + "=" * 70)
        print("STEP 9: Convert fake-quant → REAL INT4/INT8 inference modules")
        print("=" * 70)
    t0 = time.time()
    summary: Dict[str, Dict] = {}

    n_total_layers = len(layer_names)
    n_converted = 0
    total_params = 0
    total_int4_params = 0
    total_int8_params = 0

    for name in layer_names:
        mod = get_module_by_name(model, name)
        if not isinstance(mod, nn.Linear):
            if verbose:
                print(f"  [skip] {name} (not nn.Linear, got {type(mod).__name__})")
            continue

        W = mod.weight.data  # (cout, cin) — already permuted by step 4
        cout, cin = W.shape
        bits = bit_allocation.get(name, 4)
        assert bits in (4, 8), f"Invalid bits={bits} for layer {name}"

        # Determine K (number of INT8 / sensitive channels)
        if bits == 8:
            K = cin  # all INT8
        else:
            K = _round_to_group(int(round(cin * intra_layer_hp_ratio)), group_size)
            # Both halves must be either 0 or >= group_size
            if K < group_size:
                K = 0
            if cin - K < group_size:
                K = cin  # all INT8 if there's no room for an INT4 group
            K = min(K, cin)

        # Pull permutation indices (if available) for the buffer
        perm = permutation_indices.get(name) if permutation_indices else None

        # Pull precomputed GPTQ integer codes (if Step 8 stored them on the module)
        pre_codes = getattr(mod, "_shmq_int_codes", None)
        pre_scales = getattr(mod, "_shmq_scales", None)
        pre_nbits = getattr(mod, "_shmq_n_bits", None)

        # Build the SHMQQuantLinear from the permuted weight
        bias = mod.bias.data if mod.bias is not None else None
        shmq_lin = SHMQQuantLinear.from_weight(
            weight=W,
            n_sensitive=K,
            group_size=group_size,
            bias=bias,
            perm=perm,
            precomputed_codes=pre_codes,
            precomputed_scales=pre_scales,
            precomputed_n_bits=pre_nbits,
            device=W.device,
        )

        # Replace in model
        set_module_by_name(model, name, shmq_lin)
        n_converted += 1

        # Stats
        n_params = cout * cin
        n_8 = cout * K
        n_4 = cout * (cin - K)
        total_params += n_params
        total_int8_params += n_8
        total_int4_params += n_4
        avg_bits = (8 * n_8 + 4 * n_4) / max(n_params, 1)
        summary[name] = {
            "n_sensitive":   K,
            "n_insensitive": cin - K,
            "bits":          bits,
            "avg_bits":      avg_bits,
            "params":        n_params,
            "in_features":   cin,
            "out_features":  cout,
        }

    t1 = time.time()
    overall_avg_bits = (8 * total_int8_params + 4 * total_int4_params) / max(total_params, 1)
    if verbose:
        print(f"\n[step9] Converted {n_converted}/{n_total_layers} Linear layers in {t1-t0:.1f}s")
        print(f"[step9] Total params: {total_params/1e6:.1f}M "
              f"(INT8: {total_int8_params/1e6:.1f}M, INT4: {total_int4_params/1e6:.1f}M)")
        print(f"[step9] Average bits per weight: {overall_avg_bits:.3f}")
        print(f"[step9] Memory footprint: {total_int8_params + total_int4_params/2:,} bytes "
              f"vs FP16 {2*total_params:,} bytes "
              f"(compression: {2*total_params / max(total_int8_params + total_int4_params/2, 1):.2f}x)")
        # Also report backend
        from .kernel_loader import is_cuda_kernel_available
        backend = "CUDA custom kernel" if is_cuda_kernel_available() else "CPU fallback (PyTorch)"
        print(f"[step9] Inference backend: {backend}")
    return summary


def convert_model_back_to_fake_quant(
    model: nn.Module,
    layer_names: List[str],
) -> None:
    """Inverse: replace every SHMQQuantLinear with an nn.Linear holding the
    dequantized (fake-quant) weight. Useful for debug / accuracy comparison.
    """
    for name in layer_names:
        mod = get_module_by_name(model, name)
        if not isinstance(mod, SHMQQuantLinear):
            continue
        W_dq = mod.dequantize_weight()  # (cout, cin) float32, SHMQ-permuted
        new_lin = nn.Linear(mod.in_features, mod.out_features,
                            bias=mod.bias is not None,
                            device=W_dq.device, dtype=torch.float16)
        new_lin.weight.data = W_dq.to(torch.float16)
        if mod.bias is not None:
            new_lin.bias.data = mod.bias.to(torch.float16).clone()
        set_module_by_name(model, name, new_lin)
