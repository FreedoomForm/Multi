"""Intra-layer per-element OBS sensitivity for SHMQ-Ultimate.

Implements SHMQ Eq. 10 / Eq. 24:
    S^l_{i,j} = (1/2) * (w^l_{i,j} - Q(w^l_{i,j}))^2 / [(X X^T + λ·mean(diag(X X^T))·I)^{-1}]_{j,j}

where:
    - X = calibration input activation matrix for layer l, shape (n_samples, cin)
    - X X^T would be (n_samples, n_samples) — too big! Use X^T X which is (cin, cin).
    - The notation in the paper uses "X X^T" but the standard GPTQ formulation uses H = X^T X.
      For a Linear layer W of shape (cout, cin), the Hessian of the loss w.r.t. W is
      H = X^T X (Gram of input activations), shape (cin, cin).
    - dampening: λ * mean(diag(H)) added to H diagonal before inversion (Levenberg-Marquardt)
    - Inversion via Cholesky decomposition (Krishnamoorthy & Menon 2013)

Reference: SHMQ paper Eq. 10, 24, Appendix A.1.3.
Also: SliM-LLM slim_gptq.py::SliMGPTQ.add_batch + get_salience.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from ..utils import get_module_by_name, symmetric_quantize_weights


class OBSHessian:
    """Per-layer OBS (Optimal Brain Surgeon) Hessian computation.

    Usage:
        obs = OBSHessian(dampening=0.1, use_mean_diag=True)
        obs.add_batch(X)        # X: (n_samples, cin) input activations
        H = obs.get_hessian()   # (cin, cin)
        Hinv = obs.get_hessian_inverse()  # (cin, cin)
        sens = obs.compute_sensitivity(W, Q_W)  # (cout, cin) per-element sensitivity
    """

    def __init__(self, dampening: float = 0.1, use_mean_diag: bool = True):
        self.dampening = dampening
        self.use_mean_diag = use_mean_diag
        self.H: Optional[torch.Tensor] = None  # (cin, cin)
        self.n_samples: int = 0

    def add_batch(self, X: torch.Tensor):
        """Accumulate input activations X into the Hessian.

        Args:
            X: (n_samples, cin) tensor of input activations
        """
        if X.dim() == 3:
            # (B, S, cin) -> (B*S, cin)
            X = X.reshape(-1, X.shape[-1])
        X = X.float()
        if self.H is None:
            self.H = X.T @ X  # (cin, cin)
        else:
            self.H += X.T @ X
        self.n_samples += X.shape[0]

    def get_hessian(self) -> torch.Tensor:
        """Return the (cin, cin) Hessian H = X^T X with optional dampening."""
        if self.H is None:
            raise RuntimeError("Call add_batch() first.")
        H = self.H.float().clone()
        if self.use_mean_diag:
            # λ * mean(diag(H)) added to diagonal
            damp = self.dampening * H.diag().mean().item()
        else:
            damp = self.dampening
        H.add_(torch.eye(H.shape[0], device=H.device, dtype=H.dtype) * damp)
        return H

    def get_hessian_inverse(self) -> torch.Tensor:
        """Return H^{-1} via Cholesky decomposition.

        H is positive definite (X^T X + λI), so Cholesky is fast and stable.
        """
        H = self.get_hessian()
        try:
            L = torch.linalg.cholesky(H)
            Hinv = torch.cholesky_inverse(L)
            return Hinv
        except Exception as e:
            print(f"[obs] WARNING: Cholesky failed ({e}); falling back to direct inverse")
            return torch.linalg.inv(H)

    def compute_sensitivity(self, W: torch.Tensor, Q_W: torch.Tensor) -> torch.Tensor:
        """Compute per-element sensitivity S_{i,j} (SHMQ Eq. 10).

        Args:
            W: (cout, cin) original weights
            Q_W: (cout, cin) quantized weights

        Returns:
            S: (cout, cin) per-element sensitivity
                S_{i,j} = (1/2) * (W - Q_W)^2_{i,j} / [H^{-1}]_{j,j}
        """
        δW = (W - Q_W).float()  # (cout, cin)
        Hinv = self.get_hessian_inverse()  # (cin, cin)
        Hinv_diag = Hinv.diag()  # (cin,)
        # Per-element: S_{i,j} = 0.5 * δW_{i,j}^2 / Hinv_diag[j]
        # Broadcast Hinv_diag over rows (cout)
        S = 0.5 * (δW ** 2) / Hinv_diag.unsqueeze(0).clamp(min=1e-10)
        return S


def compute_intra_layer_obs_sensitivity(
    model: nn.Module,
    layer_names: List[str],
    calibration_data: torch.Tensor,
    n_bits: int = 4,
    group_size: int = 128,
    dampening: float = 0.1,
    use_mean_diag: bool = True,
    batch_size: int = 1,
    max_samples: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Compute per-element OBS sensitivity for each layer.

    Returns: Dict {layer_name: (cout, cin) sensitivity tensor}
    """
    device = next(model.parameters()).device
    n_total = calibration_data.shape[0]
    if max_samples is not None:
        n_total = min(n_total, max_samples)

    # Hook all layers to capture input activations
    captured: Dict[str, List[torch.Tensor]] = {n: [] for n in layer_names}
    handles: List = []

    def make_hook(name: str):
        def hook(module: nn.Module, inputs, outputs):
            x = inputs[0].detach()  # (B, S, cin)
            x = x.reshape(-1, x.shape[-1])  # (B*S, cin)
            captured[name].append(x.to(device))
        return hook

    for name in layer_names:
        mod = get_module_by_name(model, name)
        handles.append(mod.register_forward_hook(make_hook(name)))

    print(f"[obs] Capturing input activations from {n_total} samples...")
    with torch.no_grad():
        for i in range(0, n_total, batch_size):
            batch = calibration_data[i : i + batch_size].to(device)
            model(batch)

    for h in handles:
        h.remove()

    # Compute per-layer OBS sensitivity
    print(f"[obs] Computing per-element OBS sensitivity for {len(layer_names)} layers...")
    sensitivities: Dict[str, torch.Tensor] = {}
    for name in layer_names:
        mod = get_module_by_name(model, name)
        W = mod.weight.data  # (cout, cin)
        Q_W, _ = symmetric_quantize_weights(W, n_bits=n_bits, group_size=group_size)

        obs = OBSHessian(dampening=dampening, use_mean_diag=use_mean_diag)
        for x_batch in captured[name]:
            obs.add_batch(x_batch)
        if obs.H is None:
            # No activations captured — fall back to identity
            print(f"[obs] WARNING: no activations for {name}, using zero sensitivity")
            sensitivities[name] = torch.zeros_like(W, dtype=torch.float32)
            continue
        sens = obs.compute_sensitivity(W, Q_W)
        sensitivities[name] = sens.detach().cpu()

    return sensitivities
