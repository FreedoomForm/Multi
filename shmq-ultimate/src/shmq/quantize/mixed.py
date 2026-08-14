"""Mixed INT4/INT8 quantization — the final step (Step 8).

Applies per-layer bit-width (from ILP) to quantize each Linear:
- 4-bit layers: GPTQ with 4-bit (after AutoRound V baking)
- 8-bit layers: RTN with 8-bit (no GPTQ needed — 8-bit is already near-lossless)

Activations are quantized to 8-bit per-token at inference time (W4.8A8 format).

Side effect: stores `_shmq_int_codes`, `_shmq_scales`, `_shmq_n_bits` on each
Linear module for downstream use by Step 9 (real INT4 inference packing).
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from ..utils import get_module_by_name, symmetric_quantize_weights, symmetric_quantize_activations
from .gptq import GPTQQuantizer


def _store_codes_on_module(mod: nn.Linear, int_codes: torch.Tensor,
                            scales: torch.Tensor, n_bits: int) -> None:
    """Cache integer codes + scales on the module for Step 9 reuse."""
    mod._shmq_int_codes = int_codes.to("cpu").to(torch.int8)
    mod._shmq_scales = scales.to("cpu").to(torch.float16)
    mod._shmq_n_bits = int(n_bits)


def _rtn_quantize_to_codes(weight: torch.Tensor, n_bits: int, group_size: int
                            ) -> Tuple[torch.Tensor, torch.Tensor]:
    """RTN quantization that returns INTEGER codes + scales (not fake-quant)."""
    out_features, in_features = weight.shape
    n_groups = in_features // group_size
    max_q = 2 ** (n_bits - 1) - 1
    w_g = weight.reshape(out_features, n_groups, group_size)
    max_abs = w_g.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = (max_abs / max_q).to(torch.float16)
    codes = (w_g / scale.to(weight.dtype)).round().clamp(-max_q, max_q).to(torch.int8)
    codes = codes.reshape(out_features, in_features)
    scales = scale.squeeze(-1)
    return codes, scales


class MixedPrecisionQuantizer:
    """Apply mixed INT4/INT8 quantization to a model.

    Usage:
        q = MixedPrecisionQuantizer(group_size=128, percdamp=0.01)
        q.apply(model, layer_names, n_bits_per_layer, captured_activations)
    """

    def __init__(self, group_size: int = 128, percdamp: float = 0.01,
                 blocksize: int = 128, activation_bits: int = 8):
        self.group_size = group_size
        self.percdamp = percdamp
        self.blocksize = blocksize
        self.activation_bits = activation_bits

    def apply(self, model: nn.Module,
              layer_names: List[str],
              n_bits_per_layer: Dict[str, int],
              captured_activations: Optional[Dict[str, List[torch.Tensor]]] = None,
              use_gptq_for_4bit: bool = True,
              sqc_multipliers: Optional[Dict[str, float]] = None) -> Dict[str, Dict[str, torch.Tensor]]:
        """Apply mixed-precision quantization to the model.

        Args:
            model: HuggingFace LLM (in-place modification)
            layer_names: list of layer names
            n_bits_per_layer: {layer_name: 4 or 8 or 16}
            captured_activations: {layer_name: [list of (N, cin) input activations]}
                                  — required for GPTQ on 4-bit layers
            use_gptq_for_4bit: if True, use GPTQ for 4-bit layers; if False, use RTN
            sqc_multipliers: {layer_name: float} — scale multiplier from SQC calibration
                             (Step 7). Applied as base_scale *= multiplier before
                             GPTQ/RTN. None = no SQC (multiplier=1.0).

        Returns:
            {layer_name: {"qweight": tensor, "scale": tensor, "n_bits": int}}
        """
        results: Dict[str, Dict[str, torch.Tensor]] = {}
        n_4bit = sum(1 for n in layer_names if n_bits_per_layer.get(n, 4) == 4)
        n_8bit = sum(1 for n in layer_names if n_bits_per_layer.get(n, 4) == 8)
        print(f"[mixed_quantize] Quantizing {len(layer_names)} layers: "
              f"{n_4bit} at 4-bit, {n_8bit} at 8-bit")

        for name in layer_names:
            mod = get_module_by_name(model, name)
            n_bits = n_bits_per_layer.get(name, 4)
            sqc_mult = (sqc_multipliers or {}).get(name, 1.0)
            if sqc_mult != 1.0:
                print(f"[mixed_quantize] applying SQC mult={sqc_mult:.4f} to {name}")

            if n_bits == 16:
                # 16-bit: keep FP16 (no quantization, just mark for inference path)
                mod._shmq_n_bits = 16
                mod._shmq_scales = None
                mod._shmq_int_codes = None
                results[name] = {"qweight": mod.weight.data.clone(), "scale": None, "n_bits": 16}
                continue

            if n_bits == 8:
                # 8-bit: RTN (near-lossless, no GPTQ needed)
                int_codes, scales = _rtn_quantize_to_codes(
                    mod.weight.data, n_bits=8, group_size=self.group_size,
                )
                # SQC: rescale codes if multiplier != 1.0
                if sqc_mult != 1.0:
                    scales = scales * sqc_mult
                    int_codes, _ = _rtn_quantize_to_codes(
                        mod.weight.data, n_bits=8, group_size=self.group_size,
                    )
                    # re-quantize with adjusted scale
                    n_groups = mod.weight.data.shape[1] // self.group_size
                    w_g = mod.weight.data.reshape(-1, n_groups, self.group_size).float()
                    codes = (w_g / scales.to(w_g.dtype).repeat_interleave(self.group_size, dim=1).reshape(w_g.shape)).round().clamp(-127, 127).to(torch.int8)
                    int_codes = codes.reshape(mod.weight.data.shape)
                qweight = (int_codes.to(torch.float32) *
                           scales.to(torch.float32).repeat_interleave(self.group_size, dim=1)
                           ).to(mod.weight.dtype)
                mod.weight.data = qweight
                _store_codes_on_module(mod, int_codes, scales, n_bits=8)
                results[name] = {"qweight": qweight, "scale": scales, "n_bits": 8}
            else:
                # 4-bit: GPTQ (if activations available) or RTN
                if use_gptq_for_4bit and captured_activations and name in captured_activations:
                    gptq = GPTQQuantizer(mod, n_bits=4, group_size=self.group_size,
                                          percdamp=self.percdamp, blocksize=self.blocksize)
                    # SQC: scale the per-group scales BEFORE GPTQ sees them
                    if sqc_mult != 1.0:
                        gptq.scale = gptq.scale * sqc_mult
                    for a in captured_activations[name]:
                        gptq.add_batch(a)
                    if captured_activations[name]:
                        qweight = gptq.quantize()
                        scale = gptq.scale
                        gptq.free()
                    else:
                        int_codes, scales = _rtn_quantize_to_codes(
                            mod.weight.data, n_bits=4, group_size=self.group_size,
                        )
                        if sqc_mult != 1.0:
                            scales = scales * sqc_mult
                            int_codes, _ = _rtn_quantize_to_codes(
                                mod.weight.data, n_bits=4, group_size=self.group_size,
                            )
                        qweight = (int_codes.to(torch.float32) *
                                   scales.to(torch.float32).repeat_interleave(self.group_size, dim=1)
                                   ).to(mod.weight.dtype)
                        mod.weight.data = qweight
                        _store_codes_on_module(mod, int_codes, scales, n_bits=4)
                        scale = scales
                else:
                    int_codes, scales = _rtn_quantize_to_codes(
                        mod.weight.data, n_bits=4, group_size=self.group_size,
                    )
                    if sqc_mult != 1.0:
                        scales = scales * sqc_mult
                        int_codes, _ = _rtn_quantize_to_codes(
                            mod.weight.data, n_bits=4, group_size=self.group_size,
                        )
                    qweight = (int_codes.to(torch.float32) *
                               scales.to(torch.float32).repeat_interleave(self.group_size, dim=1)
                               ).to(mod.weight.dtype)
                    mod.weight.data = qweight
                    _store_codes_on_module(mod, int_codes, scales, n_bits=4)
                    scale = scales
                results[name] = {"qweight": qweight, "scale": scale, "n_bits": 4}

        return results

    @staticmethod
    def quantize_activations_for_inference(x: torch.Tensor, n_bits: int = 8) -> torch.Tensor:
        """Per-token symmetric activation quantization (W4.8A8 format).

        Args:
            x: (B, S, cin) tensor
            n_bits: 8

        Returns:
            Fake-quantized (dequantized) activation tensor.
        """
        qact, _ = symmetric_quantize_activations(x, n_bits=n_bits)
        return qact
