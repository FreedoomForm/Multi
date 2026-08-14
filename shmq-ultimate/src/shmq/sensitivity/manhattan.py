"""Manhattan norm aggregation for channel sensitivity (SHMQ Eq. 11).

Implements:
    S_IntraMQ_j = ||S^l_{:,j}||_1 = Σ_i∈cout |S^l_{i,j}|

This collapses the per-element sensitivity matrix (cout, cin) into a per-channel
sensitivity vector (cin,). The TopK most sensitive channels (by this metric) are
designated as "sensitive" (Csen) and get INT8; the rest (Cinsen) get INT4.

Reference: SHMQ paper Eq. 11, Section 3.2.2.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import torch


def aggregate_manhattan_channel_sensitivity(
    per_element_sensitivity: torch.Tensor,
) -> torch.Tensor:
    """Aggregate per-element sensitivity to per-channel via Manhattan norm.

    Args:
        per_element_sensitivity: (cout, cin) tensor of S^l_{i,j}

    Returns:
        (cin,) tensor of S_IntraMQ_j = Σ_i |S^l_{i,j}|
    """
    if per_element_sensitivity.dim() != 2:
        raise ValueError(f"Expected 2D tensor (cout, cin), got shape {per_element_sensitivity.shape}")
    return per_element_sensitivity.abs().sum(dim=0)  # (cin,)


def identify_sensitive_channels(
    channel_sensitivity: torch.Tensor,
    high_precision_ratio: float,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Identify the top-K sensitive channels (SHMQ Eq. 12).

    Csen = I(S_IntraMQ, K) where K = ⌊cin * U_l⌉

    Args:
        channel_sensitivity: (cin,) tensor
        high_precision_ratio: U_l in [0, 1] — fraction of channels to designate as sensitive

    Returns:
        (sen_indices, insen_indices, K) where:
            sen_indices: (K,) LongTensor — indices of sensitive channels
            insen_indices: (cin - K,) LongTensor — indices of insensitive channels
            K: int — number of sensitive channels
    """
    cin = channel_sensitivity.numel()
    K = int(round(cin * high_precision_ratio))
    K = max(0, min(K, cin))
    if K == 0:
        return torch.empty(0, dtype=torch.long), torch.arange(cin, dtype=torch.long), 0
    if K == cin:
        return torch.arange(cin, dtype=torch.long), torch.empty(0, dtype=torch.long), K
    sen_indices = torch.topk(channel_sensitivity, K, largest=True).indices
    mask = torch.ones(cin, dtype=torch.bool)
    mask[sen_indices] = False
    insen_indices = torch.where(mask)[0]
    return sen_indices, insen_indices, K
