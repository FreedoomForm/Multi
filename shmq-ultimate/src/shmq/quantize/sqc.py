"""SQC (Salience-Weighted Quantizer Calibration) — from SliM-LLM.

Reference: SliM-LLM slim-llm/utils/mixed_quantizer.py::Quantizer.fit

Algorithm:
1. Identify salient weights via z-score > threshold (default 2.0).
2. Grid-search scale multiplier p ∈ [0.9, 1.1] (50 points each side):
     - Compute error with scale * p
     - Salient weights get extra penalty (λ_salience * err_s)
3. Pick the best scale multiplier.

This optimizes the per-group scale factor with salience awareness — better than
vanilla min-max quantization. SHMQ paper doesn't specify a quantizer calibration,
so we use SQC to fill the gap.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from ..utils import get_module_by_name


class SQCCalibrator:
    """Salience-Weighted Quantizer Calibration.

    Usage:
        calibrator = SQCCalibrator(zscore_threshold=2.0, scale_range=(0.9, 1.1))
        calibrator.calibrate_layer(weight, sensitivity)  # returns optimal scale multiplier
    """

    def __init__(self,
                 zscore_threshold: float = 2.0,
                 scale_range: Tuple[float, float] = (0.9, 1.1),
                 search_points: int = 50,
                 salience_lambda: float = 1.0):
        self.zscore_threshold = zscore_threshold
        self.scale_range = scale_range
        self.search_points = search_points
        self.salience_lambda = salience_lambda

    def _identify_salient_weights(self, weight: torch.Tensor,
                                   sensitivity: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return a boolean mask of salient weights.

        If sensitivity is provided, use it to identify salient weights (top z-score).
        Otherwise, use weight magnitude z-score.
        """
        if sensitivity is not None:
            s = sensitivity.float()
            mean_s = s.mean()
            std_s = s.std().clamp(min=1e-8)
            z = (s - mean_s) / std_s
            return z.abs() > self.zscore_threshold
        else:
            w = weight.float().abs()
            mean_w = w.mean()
            std_w = w.std().clamp(min=1e-8)
            z = (w - mean_w) / std_w
            return z.abs() > self.zscore_threshold

    def calibrate_layer(self, weight: torch.Tensor, n_bits: int = 4,
                        group_size: int = 128,
                        sensitivity: Optional[torch.Tensor] = None) -> float:
        """Find the optimal scale multiplier for a layer's weight.

        Args:
            weight: (cout, cin) tensor
            n_bits: 4 or 8
            group_size: 128
            sensitivity: (cout, cin) per-element sensitivity (from OBS) — optional

        Returns:
            best_scale_multiplier (float in [scale_range[0], scale_range[1]])
        """
        cout, cin = weight.shape
        n_groups = cin // group_size
        w = weight.float().reshape(cout, n_groups, group_size)

        # Original scale (max-abs per group)
        max_abs = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)  # (cout, n_groups, 1)
        max_q = 2 ** (n_bits - 1) - 1
        base_scale = max_abs / max_q

        # Identify salient weights
        salient_mask = self._identify_salient_weights(weight, sensitivity)  # (cout, cin)
        salient_mask_g = salient_mask.reshape(cout, n_groups, group_size)

        # Grid search
        lo, hi = self.scale_range
        candidates = torch.linspace(lo, hi, self.search_points * 2 + 1)
        # Always include 1.0
        if 1.0 not in candidates.tolist():
            candidates = torch.cat([candidates, torch.tensor([1.0])])

        best_loss = float("inf")
        best_mult = 1.0

        for mult in candidates:
            scale = base_scale * mult
            # Quantize-dequantize
            q = (w / scale).round().clamp(-max_q, max_q)
            qdq = q * scale
            err = (qdq - w) ** 2  # (cout, n_groups, group_size)

            # Salient penalty
            err_salient = err * salient_mask_g.float() * self.salience_lambda
            err_non_salient = err * (~salient_mask_g).float()
            total_loss = (err_salient.sum() + err_non_salient.sum()).item()

            if total_loss < best_loss:
                best_loss = total_loss
                best_mult = float(mult)

        return best_mult

    def calibrate_model(self, model: nn.Module, layer_names: List[str],
                        sensitivities: Optional[Dict[str, torch.Tensor]] = None,
                        n_bits_per_layer: Optional[Dict[str, int]] = None,
                        group_size: int = 128) -> Dict[str, float]:
        """Calibrate the scale multiplier for each layer.

        Returns: {layer_name: best_scale_multiplier}
        """
        results = {}
        for name in layer_names:
            mod = get_module_by_name(model, name)
            W = mod.weight.data
            n_bits = n_bits_per_layer.get(name, 4) if n_bits_per_layer else 4
            sens = sensitivities.get(name) if sensitivities else None
            mult = self.calibrate_layer(W, n_bits=n_bits, group_size=group_size,
                                         sensitivity=sens)
            results[name] = mult
        return results
