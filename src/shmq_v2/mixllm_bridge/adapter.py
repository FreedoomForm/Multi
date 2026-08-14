"""MixLLM Bridge — wraps MixLLM as the foundation of SHMQ-Ultimate v2.

MixLLM provides:
  - Bit allocation (N-axis, global loss distance)
  - Custom CUDA kernel (CUTLASS MMA mixed INT4/INT8)
  - vLLM patch for production inference
  - Calibration pipeline (WikiText2)

We wrap MixLLM's public API and add SHMQ-specific pre/post-processing:
  - BEFORE MixLLM: SmoothQuant + K-axis permutation + RMSNorm fusion
  - DURING MixLLM: AutoRound-informed rounding (interleaved with GPTQ)
  - AFTER MixLLM: nothing (the output is a MixLLM-quantized model with
                  SHMQ permutation baked into the RMSNorm weights)
"""
from __future__ import annotations
import sys
import os
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

# Add external/MixLLM to sys.path so we can import mixllm as a package
_MIXLLM_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "external", "MixLLM")
_MIXLLM_PATH = os.path.abspath(_MIXLLM_PATH)
if _MIXLLM_PATH not in sys.path:
    sys.path.insert(0, _MIXLLM_PATH)

try:
    from mixllm.quantization.quantizer import MixLLMConfig, Quantizer, QuantConfig
    from mixllm.quantization.searcher import MixLLMSearcher
    import mixllm.utils.modeling as mixllm_modeling
    from mixllm.utils.datautils import DataUtils
    MIXLLM_AVAILABLE = True
except ImportError as e:
    MIXLLM_AVAILABLE = False
    _IMPORT_ERROR = str(e)


def ensure_mixllm_available() -> None:
    """Raise a clear error if MixLLM is not importable."""
    if not MIXLLM_AVAILABLE:
        raise ImportError(
            f"MixLLM is not available. Cloned to {_MIXLLM_PATH}?\n"
            f"Original error: {_IMPORT_ERROR}\n"
            f"Try: cd {_MIXLLM_PATH} && pip install -e ."
        )


def build_mixllm_config(
    bit_percent: Dict[int, int],
    group_size: int = 128,
    weight_4bit_asymmetric: bool = True,
    weight_8bit_asymmetric: bool = False,
    gptq_4bit: bool = True,
    gptq_8bit: bool = True,
    clip_shrink_4bit: bool = True,
    clip_shrink_8bit: bool = True,
    gptq_group_reorder: bool = True,
    activation_bit_width: int = 8,
    activation_asymmetric: bool = False,
) -> "MixLLMConfig":
    """Build a MixLLMConfig from SHMQv2Config parameters.

    This is a thin wrapper that translates our config dataclass into
    MixLLM's expected format.
    """
    ensure_mixllm_available()

    weight_config = {
        4: {
            "group_size": group_size,
            "asymmetric": weight_4bit_asymmetric,
            "gptq": gptq_4bit,
            "gptq_group_reorder": gptq_group_reorder,
            "clip_shrink": clip_shrink_4bit,
        },
        8: {
            "group_size": group_size,
            "asymmetric": weight_8bit_asymmetric,
            "gptq": gptq_8bit,
            "gptq_group_reorder": gptq_group_reorder,
            "clip_shrink": clip_shrink_8bit,
        },
    }
    activation_config = {
        "bit_width": activation_bit_width,
        "group_size": group_size,
        "asymmetric": activation_asymmetric,
    }
    return MixLLMConfig(
        bit_percent=bit_percent,
        weight_config=weight_config,
        activation_config=activation_config,
    )


def get_calibration_data(
    model_name: str,
    n_samples: int = 128,
    seed: int = 0,
    device: str = "cuda",
) -> torch.Tensor:
    """Load WikiText2 calibration data using MixLLM's data utils."""
    ensure_mixllm_available()

    seqlen = mixllm_modeling.get_seqlen(model_name)
    trainloader, _ = DataUtils.get_loaders(
        "wikitext2",
        nsamples=n_samples,
        seed=seed,
        seqlen=seqlen,
        model=model_name,
    )
    calib_data, _ = DataUtils.trainloader_to_tensor(trainloader, device=device)
    return calib_data


def run_mixllm_allocation(
    model_name: str,
    calib_data: torch.Tensor,
    mixllm_config: "MixLLMConfig",
    device: str = "cuda",
) -> "MixLLMConfig":
    """Run MixLLM's global loss-distance bit allocation.

    This is the SEARCH phase — it determines which output channels (N-axis)
    get INT4 vs INT8. The model is NOT modified; only `mixllm_config.linear_config_map`
    is populated with per-layer QuantConfig.
    """
    ensure_mixllm_available()

    return MixLLMSearcher.search_mix_config(
        pretrained_model_name_or_path=model_name,
        calib_input=calib_data,
        mixllm_config=mixllm_config,
        device=device,
    )


def run_mixllm_quantize(
    model: nn.Module,
    calib_data: torch.Tensor,
    mixllm_config: "MixLLMConfig",
) -> nn.Module:
    """Run MixLLM's fake quantization (modifies model in-place).

    For real INT4/INT8 packing (for vLLM), use `Quantizer.quantize_model_fake`
    with `fake=False` instead. This is the "fake" path that produces a
    model with quantized weights stored as FP16 (for evaluation only).
    """
    ensure_mixllm_available()

    Quantizer.quantize_model_fake(model, calib_data, mixllm_config)
    return model


def get_named_linears(model: nn.Module) -> List[Tuple[str, nn.Linear]]:
    """Get all named Linear layers in the transformer blocks (MixLLM helper)."""
    ensure_mixllm_available()
    return mixllm_modeling.get_named_linears_in_transformer_layers(model)


def capture_activations(
    model: nn.Module,
    calib_data: torch.Tensor,
    target_linears: List[Tuple[str, nn.Linear]],
    device: str = "cuda",
    max_tokens: int = 4096,
) -> Dict[str, torch.Tensor]:
    """Run a forward pass and capture input activations for each target Linear.

    Used for:
      - SmoothQuant scale computation
      - SHMQ intra-layer sensitivity computation
      - AutoRound optimization

    Args:
        model: the model to forward through
        calib_data: [batch, seqlen] calibration tokens
        target_linears: list of (name, linear_module) to capture inputs for
        device: where to store captured activations
        max_tokens: cap total tokens to limit memory usage

    Returns:
        activations: dict of layer_name → [n_tokens, in_features] tensor
    """
    activations: Dict[str, torch.Tensor] = {}
    hooks = []

    def make_hook(name: str):
        def hook(module, inp, out):
            x = inp[0].detach()
            # Flatten to [n_tokens, in_features]
            x = x.reshape(-1, x.shape[-1])
            # Cap memory: keep at most max_tokens rows
            if x.shape[0] > max_tokens:
                x = x[:max_tokens]
            activations[name] = x.to(device)
        return hook

    for name, linear in target_linears:
        linear.name = name  # MixLLM convention
        h = linear.register_forward_hook(make_hook(name))
        hooks.append(h)

    try:
        with torch.no_grad():
            # Forward just enough data to populate hooks
            # Use a small batch to save time
            batch_size = min(4, calib_data.shape[0])
            model(calib_data[:batch_size].to(device), use_cache=False)
    finally:
        for h in hooks:
            h.remove()

    return activations
