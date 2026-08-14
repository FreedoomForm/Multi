"""MixLLM integration for SHMQ-Ultimate.

This package wraps Microsoft's MixLLM CUDA kernel and extends it with
an FP16 path for the new {16, 8, 4} 3-level scheme.

Public API:
    is_mixllm_available()           — check if MixLLM is importable
    SHMQMixLLMLinear                — combined FP16 + INT8 + INT4 linear layer
    convert_model_to_mixllm()       — replace nn.Linear with SHMQMixLLMLinear
    pack_int4_weights, pack_int8_weights  — weight packing helpers
"""
from .adapter import (
    is_mixllm_available,
    SHMQMixLLMConfig,
    SHMQMixLLMLinear,
    ConversionSummary,
    convert_linear_to_mixllm,
    convert_model_to_mixllm,
    pack_int4_weights,
    pack_int8_weights,
)

__all__ = [
    "is_mixllm_available",
    "SHMQMixLLMConfig",
    "SHMQMixLLMLinear",
    "ConversionSummary",
    "convert_linear_to_mixllm",
    "convert_model_to_mixllm",
    "pack_int4_weights",
    "pack_int8_weights",
]
