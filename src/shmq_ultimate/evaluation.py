"""Evaluation utilities: WikiText-2 perplexity + model memory accounting."""
from __future__ import annotations

import torch
import torch.nn as nn


@torch.no_grad()
def wikitext2_perplexity(model, ids: torch.Tensor, seq_len: int = 2048,
                         max_windows: int = 0) -> float:
    """Standard non-overlapping-window perplexity (GPTQ protocol)."""
    model.eval()
    n = ids.shape[1] // seq_len
    if max_windows:
        n = min(n, max_windows)
    nlls = []
    for i in range(n):
        batch = ids[:, i * seq_len:(i + 1) * seq_len]
        out = model(batch, labels=batch)
        nlls.append(out.loss.float() * seq_len)
    if not nlls:
        return float("nan")
    return float(torch.exp(torch.stack(nlls).sum() / (len(nlls) * seq_len)))


def linear_weight_bytes(model) -> int:
    """Bytes used by linear-layer weights (FP or quantized buffers)."""
    from .inference.quant_linear import SHMQUltimateLinear
    total = 0
    for m in model.modules():
        if isinstance(m, SHMQUltimateLinear):
            total += m.memory_bytes()
        elif isinstance(m, nn.Linear):
            total += m.weight.numel() * m.weight.element_size()
            if m.bias is not None:
                total += m.bias.numel() * m.bias.element_size()
    return total
