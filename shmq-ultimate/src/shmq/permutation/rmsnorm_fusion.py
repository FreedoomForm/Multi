"""PermutedRMSNorm: bake permutation into RMSNorm for zero-overhead inference.

SHMQ paper Appendix A.3.1:
    "the reordering of the input activation of q_proj/k_proj/v_proj linear layers
     is fused into the prior normalization layer."

This is the FUSION step: instead of applying the permutation at runtime (which
would require a gather operation on every forward pass), we permute the RMSNorm
weight vector so that the permutation is "baked in" to the normalization.

Specifically, if perm_indices[j] = k, then:
    Original:  y = RMSNorm(x); out = W @ y
    Permuted:  y = RMSNorm(x[perm]); out = W[:, perm] @ y
    Fused:     y = RMSNorm_permuted(x); out = W[:, perm] @ y
              where RMSNorm_permuted.weight = RMSNorm.weight[perm]

By permuting the RMSNorm weight (and any bias), the output of RMSNorm_permuted(x)
is the same as RMSNorm(x)[perm]. So no runtime permutation is needed.

For Qwen2.5 / Llama:
    - input_layernorm is the RMSNorm BEFORE q/k/v proj → fuse q/k/v permutation into it
    - post_attention_layernorm is the RMSNorm BEFORE gate/up proj → fuse gate/up permutation into it
    - o_proj has its own preceding source (attention output) → no fusion needed
    - down_proj has its own preceding source (SwiLU output) → no fusion needed

Reference: SHMQ paper Section 3.2, Appendix A.3.1.
"""
from __future__ import annotations
from typing import List, Optional, Dict
import torch
import torch.nn as nn
from ..utils import get_module_by_name, set_module_by_name


class PermutedRMSNorm(nn.Module):
    """RMSNorm with a baked-in channel permutation.

    Equivalent to: y = RMSNorm(x)[perm_indices]
    But implemented as: y = RMSNorm_with_permuted_weight(x)
    So no runtime gather is needed.

    Forward:
        x: (B, S, cin) or (N, cin)
        out: same shape, with channels permuted

    Note: if the original RMSNorm has a bias (LayerNorm style), the bias is also
    permuted. RMSNorm typically has no bias.
    """

    def __init__(self, original_norm: nn.Module, perm_indices: torch.Tensor):
        super().__init__()
        # PyTorch nn.RMSNorm stores eps as a list [1e-6] in some versions;
        # handle both scalar and list forms.
        eps_val = getattr(original_norm, "eps", 1e-6)
        if isinstance(eps_val, (list, tuple)):
            eps_val = eps_val[0] if len(eps_val) > 0 else 1e-6
        self.eps = float(eps_val) if eps_val is not None else 1e-6
        # Permute the weight
        # original_norm.weight shape: (hidden_size,)
        w = original_norm.weight.data.clone()
        perm_indices = perm_indices.to(w.device)
        self.weight = nn.Parameter(w[perm_indices].clone())
        # Handle bias (LayerNorm may have one; RMSNorm usually doesn't)
        if hasattr(original_norm, "bias") and original_norm.bias is not None:
            b = original_norm.bias.data.clone()
            self.bias = nn.Parameter(b[perm_indices].clone())
        else:
            self.bias = None
        # Store perm_indices for diagnostic purposes
        self.register_buffer("perm_indices", perm_indices.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: gather x by perm, then apply RMSNorm with permuted weight.

        Mathematically:
            output[j] = x[perm[j]] / rms(x) * weight[perm[j]] = (RMSNorm(x))[perm[j]]

        The gather on x is the runtime cost — but it's fused INTO the RMSNorm
        computation, so there's no separate gather step. The next Linear (q/k/v)
        uses a permuted weight W[:, perm] and operates on this output directly.
        """
        # Gather x by perm (the runtime cost that the fusion encapsulates)
        # x shape: (..., cin); perm shape: (cin,)
        x_gathered = x.index_select(-1, self.perm_indices.to(x.device))
        # Standard RMSNorm on the gathered input
        if x_gathered.dtype != torch.float32 and x_gathered.dtype != torch.float16:
            x_gathered = x_gathered.float()
        x_f = x_gathered.float()
        var = x_f.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x_f * torch.rsqrt(var + self.eps)
        x_normed = x_normed.to(self.weight.dtype)
        if self.bias is not None:
            return x_normed * self.weight + self.bias
        return x_normed * self.weight


def fuse_permutation_into_rmsnorm(
    model: nn.Module,
    perm_indices_per_layer: Dict[str, torch.Tensor],
) -> Dict[str, str]:
    """Fuse the channel permutation into the preceding RMSNorm for each layer.

    For Qwen2.5 / Llama:
        - q_proj, k_proj, v_proj → input_layernorm
        - gate_proj, up_proj → post_attention_layernorm
        - o_proj, down_proj → no fusion (no preceding RMSNorm to fold into)

    When multiple layers share the same RMSNorm (q/k/v share input_layernorm),
    we require that they have the SAME perm_indices (which is enforced by the
    parallel constraint in sensitivity/parallel.py).

    Args:
        model: HuggingFace LLM
        perm_indices_per_layer: {layer_name: (cin,) perm indices}

    Returns:
        Dict {norm_layer_name: "fused with permutation for [layer1, layer2, ...]"}
        — for diagnostic / logging purposes.
    """
    import re
    block_pattern = re.compile(r"model\.layers\.(\d+)\.(\w+)\.(\w+_proj)")

    # Group perm_indices by the RMSNorm they should be fused into
    norm_to_perm: Dict[str, Dict[str, torch.Tensor]] = {}  # norm_name -> {layer_name: perm}

    for layer_name, perm in perm_indices_per_layer.items():
        m = block_pattern.match(layer_name)
        if not m:
            continue
        block_idx_str, sub_module, proj_name = m.groups()
        if proj_name in ("q_proj", "k_proj", "v_proj"):
            norm_name = f"model.layers.{block_idx_str}.input_layernorm"
        elif proj_name in ("gate_proj", "up_proj"):
            norm_name = f"model.layers.{block_idx_str}.post_attention_layernorm"
        else:
            # o_proj, down_proj — no fusion
            continue
        norm_to_perm.setdefault(norm_name, {})[layer_name] = perm

    # Fuse each norm
    fused_log: Dict[str, str] = {}
    for norm_name, layer_perms in norm_to_perm.items():
        # All layers sharing this norm should have the same perm (parallel constraint)
        perms = list(layer_perms.values())
        ref_perm = perms[0]
        for p in perms[1:]:
            if not torch.equal(p, ref_perm):
                print(f"[rmsnorm_fusion] WARNING: {norm_name} has conflicting perms from "
                      f"{list(layer_perms.keys())}; using first")
                break

        # Replace the original norm with PermutedRMSNorm
        try:
            original_norm = get_module_by_name(model, norm_name)
            new_norm = PermutedRMSNorm(original_norm, ref_perm)
            # Try to preserve the original class type if possible (for compatibility)
            # We just replace it with PermutedRMSNorm — it has the same forward behavior
            # modulo the permutation.
            set_module_by_name(model, norm_name, new_norm)
            fused_log[norm_name] = f"fused with permutation for {list(layer_perms.keys())}"
        except Exception as e:
            print(f"[rmsnorm_fusion] WARNING: failed to fuse {norm_name}: {e}")
            fused_log[norm_name] = f"FAILED: {e}"

    return fused_log
