"""SHMQ-Ultimate — Quantize subpackage.

Implements:
- SQC (Salience-Weighted Quantizer Calibration) — from SliM-LLM
- GPTQ backend (per-element Hessian weight update) — from SliM-LLM/AutoGPTQ
- Mixed INT4/INT8 quantization with per-channel bit allocation
"""
from .sqc import SQCCalibrator
from .gptq import GPTQQuantizer
from .mixed import MixedPrecisionQuantizer

__all__ = ["SQCCalibrator", "GPTQQuantizer", "MixedPrecisionQuantizer"]
