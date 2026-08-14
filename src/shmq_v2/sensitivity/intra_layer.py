"""SHMQ Intra-layer sensitivity (paper §3.2.2, Eq. 10-11).

Intra-layer sensitivity is computed PER INPUT CHANNEL (K-axis) using the
diagonal of the inverse Hessian. SHMQ uses XX^T + λI (Eq. 10) for the
intra-layer Hessian (cheaper than Fisher, doesn't require explicit inverse).

Equations (paper):
  Eq. 10:  S^l_{i,j} = (1/2) (w - Q(w))² / [ (X X^T + λ·mean(diag(X X^T))·I)^{-1} ]_{j,j}
  Eq. 11:  S_IntraMQ_j = Σ_{i∈cout} |S^l_{i,j}|   (Manhattan channel sensitivity)

For SHMQ-Ultimate v2, we compute a SIMPLIFIED sensitivity suitable for
permutation ordering (we don't need exact values — only the relative
ordering of channels for the decoupled permutation). The exact loss values
used for bit allocation come from MixLLM's Fisher-based loss estimation,
which is computed independently along the N-axis.

So this module produces ONLY the per-channel K-axis sensitivity ranking
needed by `decoupled_permutation()`.
"""
from __future__ import annotations
import torch
from typing import Optional


def compute_intra_layer_sensitivity(
    weight: torch.Tensor,        # [out_features, in_features]
    activation: torch.Tensor,    # [n_tokens, in_features]
    lambda_damp: float = 0.1,    # λ in Eq. 10
    group_size: int = 128,
) -> torch.Tensor:
    """Compute per-input-channel (K-axis) sensitivity S_IntraMQ (SHMQ Eq. 11).

    Algorithm (per SHMQ paper §3.2.2):
      1. Compute H_intra = X^T X + λ·mean(diag(X^T X))·I   [cin × cin]
         - X^T X instead of X X^T (per paper: cin × cin is feasible, cout × cout too big)
         - λ dampening prevents ill-conditioning
         - mean(diag(X^T X)) normalizes the dampening to data scale
      2. Compute diag(H_intra^{-1})  [cin]
         - We only need the diagonal, so use a Cholesky-based efficient extraction
      3. Compute quantization error per element: (w - Q(w))²
         - Use simple RTN quantization for the estimate (cheap, doesn't need GPTQ)
      4. S^l_{i,j} = (1/2) (w - Q(w))²_{i,j} / diag(H_intra^{-1})_j  [cout × cin]
      5. S_IntraMQ_j = Σ_i |S^l_{i,j}|   (Manhattan norm, SHMQ Eq. 11)  [cin]

    Returns:
        sensitivity: 1D tensor of shape [in_features], S_IntraMQ values.
                     Higher = more sensitive channel → goes in Csen.

    Notes:
      - For very large cin (e.g., 4096+), the cin×cin Hessian is 64MB+ in FP16.
        We compute it in chunks if needed.
      - The Cholesky-based inverse-diagonal is O(cin³/3) — expensive but
        acceptable for one-time calibration.
    """
    out_features, in_features = weight.shape
    device = weight.device
    dtype = weight.dtype if weight.dtype != torch.float16 else torch.float32

    weight = weight.to(dtype)
    activation = activation.to(dtype)

    # === Step 1: H_intra = X^T X + λ·mean(diag(X^T X))·I  [cin × cin] ===
    # X is [n_tokens, cin], so X^T X is [cin, cin]
    # Compute in chunks for memory efficiency on large cin
    chunk_size = max(1, 4096 // max(1, in_features // 1024))
    H = torch.zeros(in_features, in_features, device=device, dtype=dtype)
    for start in range(0, activation.shape[0], chunk_size):
        chunk = activation[start:start + chunk_size]  # [chunk, cin]
        H += chunk.t() @ chunk
    # Dampening
    diag_mean = H.diag().mean().clamp(min=1e-8)
    H.diagonal().add_(lambda_damp * diag_mean)

    # === Step 2: diag(H_intra^{-1}) via Cholesky ===
    # H = L L^T  →  H^{-1} = L^{-T} L^{-1}
    # diag(H^{-1})_j = ||L^{-1}[:, j]||²
    # Solve L Y = I via triangular solve, then diag(H^{-1}) = (Y**2).sum(dim=0)
    try:
        L = torch.linalg.cholesky(H)
        # Solve L Y = I → Y = L^{-1}
        Y = torch.linalg.solve_triangular(L, torch.eye(in_features, device=device, dtype=dtype),
                                           upper=False)
        diag_H_inv = (Y ** 2).sum(dim=0)  # [cin]
    except RuntimeError:
        # Fallback: pseudo-inverse diagonal (less accurate but robust)
        # diag((H + εI)^{-1}) ≈ 1 / (diag(H) + ε)
        diag_H_inv = 1.0 / (H.diag().clamp(min=1e-8))

    diag_H_inv = diag_H_inv.clamp(min=1e-10)  # avoid division by zero

    # === Step 3: quantization error per element (w - Q(w))² ===
    # Use simple per-group RTN quantization at 4-bit (worst case for ranking)
    # Group size = group_size, symmetric quantization
    n_groups = in_features // group_size
    assert in_features % group_size == 0, \
        f"in_features {in_features} not divisible by group_size {group_size}"

    weight_grouped = weight.view(out_features, n_groups, group_size)
    max_val = weight_grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    # 4-bit symmetric: scale = max_val / 7
    scale = max_val / 7.0
    Q = (weight_grouped / scale).round().clamp(-8, 7) * scale
    quant_error = (weight_grouped - Q) ** 2  # [out_features, n_groups, group_size]
    quant_error = quant_error.view(out_features, in_features)

    # === Step 4: S^l_{i,j} = (1/2) (w - Q(w))²_{i,j} / diag(H^{-1})_j ===
    S = 0.5 * quant_error / diag_H_inv.unsqueeze(0)  # [out_features, in_features]

    # === Step 5: S_IntraMQ_j = Σ_i |S^l_{i,j}|   (Manhattan norm, Eq. 11) ===
    sensitivity = S.abs().sum(dim=0)  # [in_features]

    return sensitivity.to(torch.float32)


def compute_parallel_layer_sensitivity(
    sensitivities_per_layer: list,  # list of [cin] tensors, one per parallel layer (e.g., q/k/v)
) -> torch.Tensor:
    """Average sensitivities across parallel layers (SHMQ parallel constraint).

    Per SHMQ paper §3.2.4 (parallel layer constraint):
      - q/k/v share the SAME K-axis permutation (so their prior RMSNorm can be fused)
      - up/gate share the SAME K-axis permutation (so their prior SiLU can be fused)
      - Sensitivities are AVERAGED across the parallel layers before computing
        the permutation, so all layers agree on channel ordering.

    Args:
        sensitivities_per_layer: list of 1D tensors of the same shape [cin].
                                 For Qwen2.5-7B, q/k/v have cin = hidden_size = 3584.

    Returns:
        averaged: 1D tensor of shape [cin], the averaged sensitivity.
    """
    assert len(sensitivities_per_layer) > 0
    shape = sensitivities_per_layer[0].shape
    for s in sensitivities_per_layer:
        assert s.shape == shape, f"Shape mismatch: {s.shape} vs {shape}"
    stacked = torch.stack(sensitivities_per_layer, dim=0)
    return stacked.mean(dim=0)
