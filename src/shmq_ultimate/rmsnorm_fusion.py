"""Permutation fusion into RMSNorm (SHMQ zero-overhead trick).

For linears directly preceded by an RMSNorm (q/k/v after input_layernorm,
gate/up after post_attention_layernorm) the channel permutation is folded
into the norm's output gather, so inference pays no extra cost:

    gather(rmsnorm(x) * gamma, perm) == rmsnorm(x)[..., perm] * gamma[perm]

PermutedRMSNorm computes the variance over ALL channels (permutation
invariant) then emits the permuted, gamma-scaled output.
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from .model_utils import ATTN_IN, MLP_IN, BlockInfo, layer_key
from .permutation import ChannelPartition


class PermutedRMSNorm(nn.Module):
    """RMSNorm with a fused output permutation."""

    def __init__(self, weight: torch.Tensor, eps: float, perm: torch.Tensor):
        super().__init__()
        gamma = weight.detach().clone()[perm]
        self.weight = nn.Parameter(gamma)
        self.variance_epsilon = float(eps)
        self.register_buffer("perm", perm.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        var = xf.pow(2).mean(dim=-1, keepdim=True)
        xf = xf * torch.rsqrt(var + self.variance_epsilon)
        out = xf[..., self.perm] * self.weight.float()
        return out.to(dtype)


def _norm_eps(norm: nn.Module) -> float:
    for attr in ("variance_epsilon", "eps"):
        if hasattr(norm, attr):
            return float(getattr(norm, attr))
    return 1e-6


def fuse_permutations(blocks: List[BlockInfo], parts: Dict[str, ChannelPartition]) -> int:
    """Replace RMSNorms with PermutedRMSNorm where the following parallel
    group has a non-identity permutation.  Returns number of fused norms."""
    fused = 0
    for blk in blocks:
        # attention input norm -> q/k/v
        key = layer_key(blk.index, ATTN_IN[0])
        p = parts.get(key)
        if p is not None and not p.is_identity() and blk.input_norm is not None:
            new = PermutedRMSNorm(blk.input_norm.weight, _norm_eps(blk.input_norm), p.perm)
            blk.module.input_layernorm = new
            blk.input_norm = new
            fused += 1
        # post-attention norm -> gate/up
        key = layer_key(blk.index, MLP_IN[0])
        p = parts.get(key)
        if p is not None and not p.is_identity() and blk.post_attn_norm is not None:
            new = PermutedRMSNorm(blk.post_attn_norm.weight, _norm_eps(blk.post_attn_norm), p.perm)
            blk.module.post_attention_layernorm = new
            blk.post_attn_norm = new
            fused += 1
    return fused
