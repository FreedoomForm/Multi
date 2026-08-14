"""Model loading and layer identification for SHMQ-Ultimate.

Loads a HuggingFace LLM (default Qwen2.5-7B-Instruct) and provides utilities to
identify linear layers, parallel groups (q/k/v, up/gate), and per-layer metadata
needed by the SHMQ pipeline.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


# ----------------------------------------------------------------------
# Layer metadata
# ----------------------------------------------------------------------
@dataclass
class LayerInfo:
    """Metadata for a single Linear layer in the model."""
    name: str               # full parameter name (e.g. "model.layers.0.self_attn.q_proj")
    module: nn.Linear       # the actual nn.Linear module
    suffix: str             # last component of name (e.g. "q_proj")
    block_idx: int          # transformer block index (-1 for non-block layers)
    group: str              # "attention" | "ffn" | "other"
    is_parallel: bool       # whether this layer belongs to a parallel group
    parallel_group_key: Optional[str]  # e.g. "layers.0.attention" — all in same key share bits

    @property
    def weight(self) -> torch.Tensor:
        return self.module.weight

    @property
    def in_features(self) -> int:
        return self.module.in_features

    @property
    def out_features(self) -> int:
        return self.module.out_features


# ----------------------------------------------------------------------
# Model loader
# ----------------------------------------------------------------------
class ModelLoader:
    """Load a HuggingFace LLM and identify its linear layers.

    Tested against Qwen2.5-Instruct; structurally compatible with Llama, Mistral,
    and other RMSNorm-based transformers (use the Qwen2/Llama modeling code).
    """

    def __init__(self, model_name: str, device: str = "cpu",
                 dtype: torch.dtype = torch.float16,
                 trust_remote_code: bool = False):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        self.model = None
        self.tokenizer = None
        self.layers: List[LayerInfo] = []
        self._layer_by_name: Dict[str, LayerInfo] = {}

    # ------------------------------------------------------------------
    # Load model + tokenizer
    # ------------------------------------------------------------------
    def load(self) -> Tuple[nn.Module, "AutoTokenizer"]:
        """Load model and tokenizer from HuggingFace hub."""
        print(f"[ModelLoader] Loading {self.model_name} (device={self.device}, dtype={self.dtype})")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=self.trust_remote_code,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            device_map=self.device if self.device != "cpu" else None,
            trust_remote_code=self.trust_remote_code,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        # Identify layers
        self._identify_layers()
        return self.model, self.tokenizer

    # ------------------------------------------------------------------
    # Identify linear layers + parallel groups
    # ------------------------------------------------------------------
    def _identify_layers(self,
                         attention_suffixes: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj"),
                         ffn_suffixes: Tuple[str, ...] = ("gate_proj", "up_proj", "down_proj"),
                         excluded_keywords: Tuple[str, ...] = ("embed", "lm_head", "norm", "layernorm")):
        """Walk the model and identify all Linear layers, classify by group."""
        # Pattern: model.layers.{idx}.<group>.<suffix>
        block_pattern = re.compile(r"layers\.(\d+)\.")

        for name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            # Skip excluded layers
            if any(kw in name.lower() for kw in excluded_keywords):
                continue

            suffix = name.split(".")[-1]
            # Determine group
            if suffix in attention_suffixes:
                group = "attention"
            elif suffix in ffn_suffixes:
                group = "ffn"
            else:
                group = "other"

            # Determine block index
            m = block_pattern.search(name)
            block_idx = int(m.group(1)) if m else -1

            # Determine parallel group key (block_idx + group + output_size)
            # q/k/v share one key; up/gate share another; o/down are NOT parallel
            # (they have a single preceding norm/activation, but their INPUT comes from
            #  a single source, so the parallel constraint does not apply).
            #
            # GQA AWARENESS: For models with Grouped Query Attention (Qwen2.5,
            # Llama-3, Mistral, etc.), q_proj has more output channels than
            # k_proj/v_proj. The SHMQ parallel constraint requires layers in
            # the same group to have the SAME output size (for permutation
            # fusion to work). So we split q/k/v into:
            #   - "qkv" group if all three have the same out_features (MHA)
            #   - "q" group alone + "kv" group together if GQA
            # Similarly for up/gate (though they always match in current LLMs).
            is_parallel = False
            parallel_group_key = None
            if block_idx >= 0:
                out_features = module.weight.shape[0]
                if suffix in ("q_proj", "k_proj", "v_proj"):
                    is_parallel = True
                    # GQA: k_proj and v_proj have same out_features, q_proj is larger.
                    # Use output_size in the key to keep same-size layers together.
                    parallel_group_key = f"layers.{block_idx}.attn_out{out_features}"
                elif suffix in ("up_proj", "gate_proj"):
                    is_parallel = True
                    parallel_group_key = f"layers.{block_idx}.ffn_out{out_features}"
                # o_proj and down_proj are NOT parallel (single preceding source).

            info = LayerInfo(
                name=name, module=module, suffix=suffix,
                block_idx=block_idx, group=group,
                is_parallel=is_parallel, parallel_group_key=parallel_group_key,
            )
            self.layers.append(info)
            self._layer_by_name[name] = info

        print(f"[ModelLoader] Identified {len(self.layers)} Linear layers "
              f"({sum(1 for l in self.layers if l.is_parallel)} parallel, "
              f"{sum(1 for l in self.layers if not l.is_parallel)} non-parallel)")

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def get_layer(self, name: str) -> LayerInfo:
        return self._layer_by_name[name]

    def get_layers_by_group(self, group: str) -> List[LayerInfo]:
        return [l for l in self.layers if l.group == group]

    def get_parallel_groups(self) -> Dict[str, List[LayerInfo]]:
        """Return dict mapping parallel_group_key -> list of LayerInfo.

        Example: {"layers.0.attention_qkv": [q_proj, k_proj, v_proj], ...}
        """
        groups: Dict[str, List[LayerInfo]] = {}
        for layer in self.layers:
            if layer.is_parallel:
                groups.setdefault(layer.parallel_group_key, []).append(layer)
        return groups

    def get_blocks(self) -> List[int]:
        """Return sorted list of unique block indices."""
        return sorted({l.block_idx for l in self.layers if l.block_idx >= 0})

    def get_layers_in_block(self, block_idx: int) -> List[LayerInfo]:
        return [l for l in self.layers if l.block_idx == block_idx]

    # ------------------------------------------------------------------
    # Transformer block access (for forward hooks)
    # ------------------------------------------------------------------
    def get_transformer_blocks(self) -> List[nn.Module]:
        """Return the list of transformer blocks (model.layers)."""
        # Works for Qwen2, Llama, Mistral — they all use model.layers
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return list(self.model.model.layers)
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return list(self.model.transformer.h)  # GPT-2 style
        else:
            raise RuntimeError("Could not find transformer blocks in model. "
                               "Expected model.model.layers (Llama/Qwen) or "
                               "model.transformer.h (GPT-2).")

    def get_input_embeddings(self) -> nn.Module:
        return self.model.get_input_embeddings()

    def get_output_embeddings(self) -> nn.Module:
        return self.model.get_output_embeddings()
