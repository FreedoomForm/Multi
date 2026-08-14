"""Configuration for the SHMQ-Ultimate pipeline.

Defaults follow the SHMQ paper (EMNLP Industry 2025) Appendix A plus the
SHMQ-Ultimate plan extensions:
  - 3 precision levels {16, 8, 4} instead of the paper's 2 levels {8, 4}
  - ILP bit allocation (HAWQ-V3) instead of proportion mapping (Eq. 8)
  - SmoothQuant / AutoRound / SQC add-ons
  - PolyQ ISA-aware quanta matching
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class SHMQUltimateConfig:
    # ---- model / data ------------------------------------------------------
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    device: str = "cpu"                  # "cuda" on GPU boxes
    dtype: str = "float32"               # "float16" on GPU
    n_samples: int = 128                 # SHMQ paper: 128 calibration samples
    sequence_length: int = 2048          # SHMQ paper: 2048 tokens each
    calib_dataset: str = "wikitext2"
    seed: int = 0
    max_blocks: Optional[int] = None     # limit transformer blocks (testing)

    # ---- precision levels (SHMQ-Ultimate: 3 native GPU levels) -------------
    bit_levels: tuple = (16, 8, 4)
    target_avg_bits: float = 4.8         # SHMQ W4.8A8 weight budget
    activation_bits: int = 8             # A8, per-token symmetric

    # ---- SHMQ core ----------------------------------------------------------
    intra_hp_base_ratio: float = 0.125   # UB = 12.5% (App. A)
    hessian_lambda: float = 0.1          # lambda in Eq. 10/24
    group_size: int = 128
    inter_hessian: str = "fisher"        # "fisher" (paper default) | "xxt"
    enable_parallel_constraint: bool = True
    enable_permutation: bool = True
    enable_rmsnorm_fusion: bool = True

    # ---- PolyQ ISA-aware quanta matching (PolyQ §3.3) ------------------------
    quanta_int8: int = 128
    quanta_int4: int = 64

    # ---- SmoothQuant ---------------------------------------------------------
    enable_smoothquant: bool = True
    smoothquant_alpha: float = 0.5

    # ---- AutoRound (SignSGD learnable rounding) ------------------------------
    enable_autoround: bool = True
    autoround_iters: int = 200
    autoround_lr: float = 5e-3

    # ---- SQC (SliM-LLM salience-weighted quantizer calibration) --------------
    enable_sqc: bool = True
    sqc_grid: int = 20
    sqc_range: float = 0.1

    # ---- GPTQ / OBS ----------------------------------------------------------
    enable_gptq: bool = True
    gptq_blocksize: int = 128
    gptq_percdamp: float = 0.01

    # ---- output --------------------------------------------------------------
    output_dir: str = "./quantized_models/shmq_ultimate"

    def save(self, path: str) -> None:
        d = asdict(self)
        d["bit_levels"] = list(self.bit_levels)
        with open(path, "w") as f:
            json.dump(d, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "SHMQUltimateConfig":
        with open(path) as f:
            d = json.load(f)
        if "bit_levels" in d:
            d["bit_levels"] = tuple(d["bit_levels"])
        return cls(**d)
