"""Mixed INT4/INT8 quantization — the final step (Step 8).

Applies per-layer bit-width (from ILP) to quantize each Linear:
- 4-bit layers: GPTQ with 4-bit (after AutoRound V baking)
- 8-bit layers: RTN with 8-bit (no GPTQ needed — 8-bit is already near-lossless)

Activations are quantized to 8-bit per-token at inference time (W4.8A8 format).
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from ..utils import get_module_by_name, symmetric_quantize_weights, symmetric_quantize_activations
from .gptq import GPTQQuantizer


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
              use_gptq_for_4bit: bool = True) -> Dict[str, Dict[str, torch.Tensor]]:
        """Apply mixed-precision quantization to the model.

        Args:
            model: HuggingFace LLM (in-place modification)
            layer_names: list of layer names
            n_bits_per_layer: {layer_name: 4 or 8}
            captured_activations: {layer_name: [list of (N, cin) input activations]}
                                  — required for GPTQ on 4-bit layers
            use_gptq_for_4bit: if True, use GPTQ for 4-bit layers; if False, use RTN

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

            if n_bits == 8:
                # 8-bit: RTN (near-lossless, no GPTQ needed)
                qweight, scale = symmetric_quantize_weights(
                    mod.weight.data, n_bits=8, group_size=self.group_size,
                )
                mod.weight.data = qweight.to(mod.weight.dtype)
                results[name] = {"qweight": qweight, "scale": scale, "n_bits": 8}
            else:
                # 4-bit: GPTQ (if activations available) or RTN
                if use_gptq_for_4bit and captured_activations and name in captured_activations:
                    gptq = GPTQQuantizer(mod, n_bits=4, group_size=self.group_size,
                                          percdamp=self.percdamp, blocksize=self.blocksize)
                    for a in captured_activations[name]:
                        gptq.add_batch(a)
                    if captured_activations[name]:
                        qweight = gptq.quantize()
                        scale = gptq.scale
                        gptq.free()
                    else:
                        qweight, scale = symmetric_quantize_weights(
                            mod.weight.data, n_bits=4, group_size=self.group_size,
                        )
                        mod.weight.data = qweight.to(mod.weight.dtype)
                else:
                    qweight, scale = symmetric_quantize_weights(
                        mod.weight.data, n_bits=4, group_size=self.group_size,
                    )
                    mod.weight.data = qweight.to(mod.weight.dtype)
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
