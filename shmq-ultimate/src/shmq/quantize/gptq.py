"""GPTQ quantizer (per-element Hessian weight update) — from SliM-LLM/AutoGPTQ.

Reference: SliM-LLM slim-llm/slim_gptq.py (SliMGPTQ class),
           AutoGPTQ auto_gptq/quantization/gptq.py (GPTQ.fasterquant)

Algorithm (OBS-style):
    H = X^T X + λ I    (Hessian, shape (cin, cin))
    H^{-1} via Cholesky
    For each block of `block_size` columns:
        Q_block = round(W_block / scale)
        err = (W_block - Q_block) / H^{-1}_{block, block}
        W_remaining -= err @ H^{-1}_{block, remaining}  (propagate error)

This is the heart of post-training quantization: by updating remaining weights
based on the quantization error of current weights, we get a much better solution
than naive round-to-nearest.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from ..utils import get_module_by_name, symmetric_quantize_weights


class GPTQQuantizer:
    """GPTQ quantizer for a single Linear layer.

    Usage:
        gptq = GPTQQuantizer(layer, n_bits=4, group_size=128, percdamp=0.01)
        gptq.add_batch(X)  # accumulate input activations
        gptq.quantize()    # apply GPTQ
        gptq.free()        # release Hessian memory
    """

    def __init__(self, layer: nn.Linear, n_bits: int = 4, group_size: int = 128,
                 percdamp: float = 0.01, blocksize: int = 128):
        self.layer = layer
        self.n_bits = n_bits
        self.group_size = group_size
        self.percdamp = percdamp
        self.blocksize = blocksize

        W = layer.weight.data.float()  # (cout, cin)
        self.cout, self.cin = W.shape
        self.n_groups = self.cin // self.group_size

        # Hessian accumulator
        self.H = torch.zeros(self.cin, self.cin, dtype=torch.float32,
                             device=W.device)
        self.n_samples = 0

        # Pre-compute scale
        max_q = 2 ** (n_bits - 1) - 1
        w_grouped = W.reshape(self.cout, self.n_groups, self.group_size)
        max_abs = w_grouped.abs().amax(dim=-1)  # (cout, n_groups)
        self.scale = (max_abs.clamp(min=1e-8) / max_q)  # (cout, n_groups)
        self.max_q = max_q

    def add_batch(self, X: torch.Tensor):
        """Accumulate input activations into the Hessian.

        Args:
            X: (N, cin) tensor of input activations
        """
        if X.dim() == 3:
            X = X.reshape(-1, X.shape[-1])
        X = X.float()
        self.H += X.T @ X
        self.n_samples += X.shape[0]

    def quantize(self) -> torch.Tensor:
        """Apply GPTQ to the layer's weight.

        Returns:
            qweight: (cout, cin) fake-quantized (dequantized) weight
        """
        W = self.layer.weight.data.float().clone()  # (cout, cin)
        H = self.H.clone()

        # Dampening
        damp = self.percdamp * H.diag().mean()
        H.add_(torch.eye(self.cin, device=H.device, dtype=H.dtype) * damp)

        # Cholesky inverse
        try:
            L = torch.linalg.cholesky(H)
            Hinv = torch.cholesky_inverse(L)
            # Take sqrt for stable division (GPTQ convention)
            Hinv_sqrt = torch.linalg.cholesky(Hinv)
        except Exception as e:
            print(f"[gptq] WARNING: Cholesky failed ({e}); falling back to direct inverse")
            Hinv = torch.linalg.inv(H)
            Hinv_sqrt = torch.linalg.cholesky(Hinv) if self._is_pd(Hinv) else torch.sqrt(Hinv.diag().diag())

        # Apply GPTQ block by block (along cin axis)
        for i in range(0, self.cin, self.blocksize):
            i_end = min(i + self.blocksize, self.cin)
            block_idx = torch.arange(i, i_end)

            # Quantize this block
            W_block = W[:, i:i_end]  # (cout, block_size)
            # Compute scale for this block (per-group within the block)
            # For simplicity, assume group_size divides blocksize
            n_groups_in_block = (i_end - i) // self.group_size
            for g in range(n_groups_in_block):
                g_start = i + g * self.group_size
                g_end = g_start + self.group_size
                w_g = W[:, g_start:g_end]
                s = self.scale[:, self._group_index(g_start)]  # (cout,)
                q_g = (w_g / s.unsqueeze(-1)).round().clamp(-self.max_q, self.max_q)
                W[:, g_start:g_end] = q_g * s.unsqueeze(-1)

            # Propagate error to remaining columns
            err_block = (W_block - W[:, i:i_end]) / Hinv_sqrt[i:i_end, i:i_end].diag().unsqueeze(0)
            if i_end < self.cin:
                W[:, i_end:] -= err_block @ Hinv[i:i_end, i_end:]

        # Update layer weight
        self.layer.weight.data = W.to(self.layer.weight.dtype)
        return W

    def _group_index(self, col_idx: int) -> int:
        return col_idx // self.group_size

    def _is_pd(self, M: torch.Tensor, eps: float = 1e-8) -> bool:
        try:
            torch.linalg.cholesky(M + eps * torch.eye(M.shape[0], device=M.device))
            return True
        except Exception:
            return False

    def free(self):
        """Release Hessian memory."""
        self.H = None


def apply_gptq_to_model(model: nn.Module, layer_names: List[str],
                         captured_activations: Dict[str, List[torch.Tensor]],
                         n_bits_per_layer: Dict[str, int],
                         group_size: int = 128,
                         percdamp: float = 0.01,
                         blocksize: int = 128) -> Dict[str, torch.Tensor]:
    """Apply GPTQ to each layer in the model.

    Args:
        model: HuggingFace LLM
        layer_names: list of layer names to quantize
        captured_activations: {layer_name: [list of (N, cin) input activations]}
        n_bits_per_layer: {layer_name: 4 or 8} — bit-width per layer
        group_size: 128
        percdamp: 0.01
        blocksize: 128

    Returns:
        {layer_name: quantized weight tensor}
    """
    results = {}
    for name in layer_names:
        mod = get_module_by_name(model, name)
        n_bits = n_bits_per_layer.get(name, 4)

        # If 8-bit, just do RTN (GPTQ for 8-bit doesn't help much)
        if n_bits == 8:
            qweight, _ = symmetric_quantize_weights(mod.weight.data, n_bits=8,
                                                    group_size=group_size)
            mod.weight.data = qweight.to(mod.weight.dtype)
            results[name] = mod.weight.data.clone()
            continue

        # 4-bit: apply GPTQ
        gptq = GPTQQuantizer(mod, n_bits=n_bits, group_size=group_size,
                              percdamp=percdamp, blocksize=blocksize)
        acts = captured_activations.get(name, [])
        for a in acts:
            gptq.add_batch(a)
        if acts:
            qweight = gptq.quantize()
            results[name] = qweight
        else:
            # No activations, fall back to RTN
            qweight, _ = symmetric_quantize_weights(mod.weight.data, n_bits=n_bits,
                                                    group_size=group_size)
            mod.weight.data = qweight.to(mod.weight.dtype)
            results[name] = mod.weight.data.clone()
        gptq.free()
    return results
