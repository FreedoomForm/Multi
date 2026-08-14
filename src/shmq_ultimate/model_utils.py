"""Model loading + architecture introspection for Llama/Qwen-family models.

Identifies:
  - transformer blocks and their Linear layers
  - parallel groups (q/k/v share input; up/gate share input) — SHMQ §3.2 Eq. 4
  - the RMSNorm that feeds each Linear group (for permutation fusion §3.2.2)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn


ATTN_IN = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]
ATTN_OUT = ["self_attn.o_proj"]
MLP_IN = ["mlp.gate_proj", "mlp.up_proj"]
MLP_OUT = ["mlp.down_proj"]
ALL_ROLES = ATTN_IN + ATTN_OUT + MLP_IN + MLP_OUT


@dataclass
class BlockInfo:
    index: int
    prefix: str
    module: nn.Module
    linears: Dict[str, nn.Linear] = field(default_factory=dict)
    input_norm: Optional[nn.Module] = None
    post_attn_norm: Optional[nn.Module] = None


def load_model(model_name: str, dtype: str = "float32", device: str = "cpu",
               max_blocks: Optional[int] = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch_dtype = getattr(torch, dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype, low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if max_blocks is not None:
        layers = get_decoder_layers(model)
        del layers[max_blocks:]
        model.config.num_hidden_layers = max_blocks
    model.to(device)
    model.eval()
    return model, tokenizer


def get_decoder_layers(model) -> nn.ModuleList:
    for path in ("model.layers", "transformer.h", "model.decoder.layers"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            if isinstance(obj, nn.ModuleList):
                return obj
        except AttributeError:
            continue
    raise ValueError("Could not locate decoder layers")


def collect_blocks(model) -> List[BlockInfo]:
    layers = get_decoder_layers(model)
    blocks = []
    for i, blk in enumerate(layers):
        info = BlockInfo(index=i, prefix=f"model.layers.{i}", module=blk)
        for role in ALL_ROLES:
            obj = blk
            ok = True
            for part in role.split("."):
                if not hasattr(obj, part):
                    ok = False
                    break
                obj = getattr(obj, part)
            if ok and isinstance(obj, nn.Linear):
                info.linears[role] = obj
        info.input_norm = getattr(blk, "input_layernorm", None)
        info.post_attn_norm = getattr(blk, "post_attention_layernorm", None)
        blocks.append(info)
    return blocks


def parallel_groups(block: BlockInfo) -> List[List[str]]:
    """Groups of layer-roles sharing the same input activation (SHMQ Eq. 4)."""
    groups = []
    g1 = [r for r in ATTN_IN if r in block.linears]
    if g1:
        groups.append(g1)
    g2 = [r for r in MLP_IN if r in block.linears]
    if g2:
        groups.append(g2)
    for r in ATTN_OUT + MLP_OUT:
        if r in block.linears:
            groups.append([r])
    return groups


def layer_key(block_idx: int, role: str) -> str:
    return f"model.layers.{block_idx}.{role}"
