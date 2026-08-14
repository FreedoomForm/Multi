"""SHMQ-Ultimate v2 — Configuration dataclass.

Builds on MixLLM as foundation (MixLLMConfig untouched) and adds SHMQ-specific
parameters for K-axis permutation + RMSNorm fusion.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, List


@dataclass
class SHMQv2Config:
    """Full configuration for SHMQ-Ultimate v2 pipeline.

    MixLLM parameters (kept as-is, mirror MixLLMConfig):
        bit_percent: {8: 10, 4: 90} means W4.4A8 (10% INT8, 90% INT4)
        group_size: 128 (MixLLM kernel hard-coded group size)
        activation_bit_width: 8 (W4A8 format)

    SHMQ-specific parameters (NEW in v2):
        enable_permutation: apply SHMQ K-axis decoupled permutation
        enable_rmsnorm_fusion: fuse input permutation into prior RMSNorm
        enable_parallel_constraint: q/k/v share perm; up/gate share perm
        enable_smoothquant: activation outlier migration pre-processing
        enable_autoround: learnable SignSGD rounding (200 steps)
        intra_hessian_lambda: dampening for XX^T + λI (default 0.1, SHMQ paper)
        hp_ratio: fraction of input channels marked sensitive (Eq. 12)
                  (typically matches MixLLM's INT8 ratio, default 0.10)

    Pipeline control:
        skip_steps: set of step indices (1-11) to skip
        device: "cuda" for production, "cpu" for smoke tests
    """
    # === MixLLM foundation (untouched) ===
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    bit_percent: Dict[int, int] = field(default_factory=lambda: {8: 10, 4: 90})
    group_size: int = 128
    activation_bit_width: int = 8
    activation_asymmetric: bool = False  # MixLLM requires False
    weight_4bit_asymmetric: bool = True   # MixLLM default
    weight_8bit_asymmetric: bool = False  # MixLLM default
    gptq_4bit: bool = True
    gptq_8bit: bool = True
    clip_shrink_4bit: bool = True
    clip_shrink_8bit: bool = True
    gptq_group_reorder: bool = True

    # === Calibration ===
    calib_dataset: str = "wikitext2"
    n_samples: int = 128
    seed: int = 0
    sequence_length: int = 2048

    # === SHMQ-specific (NEW) ===
    enable_permutation: bool = True
    enable_rmsnorm_fusion: bool = True
    enable_parallel_constraint: bool = True
    enable_smoothquant: bool = True
    enable_autoround: bool = True
    intra_hessian_lambda: float = 0.1        # λ for XX^T + λI (SHMQ Eq. 10)
    hp_ratio: float = 0.10                    # high-precision K-channel ratio
    autoround_iters: int = 200                # SignSGD iterations per block
    autoround_lr: float = 1e-3
    autoround_block_size: int = 128           # Block size for AutoRound V optimization

    # === SmoothQuant ===
    smoothquant_alpha: float = 0.5            # migration strength
    smoothquant_subset: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj"
    ])  # skip down_proj (its input is post-SiLU, no outliers)

    # === Pipeline control ===
    device: str = "cuda"
    dtype: str = "float16"
    skip_steps: set = field(default_factory=set)
    save_dir: Optional[str] = None

    # === Evaluation ===
    eval_ppl: bool = True
    eval_ppl_dataset: str = "wikitext2"
    eval_zeroshot: bool = False
    eval_zeroshot_tasks: List[str] = field(default_factory=lambda: [
        "hellaswag", "arc_challenge", "piqa", "winogrande"
    ])

    def validate(self) -> None:
        """Validate configuration consistency."""
        assert sum(self.bit_percent.values()) == 100, \
            f"bit_percent must sum to 100, got {sum(self.bit_percent.values())}"
        assert all(b in (4, 8) for b in self.bit_percent), \
            f"Only 2 levels {{4, 8}} supported (MixLLM native), got {list(self.bit_percent)}"
        assert self.group_size == 128, \
            f"MixLLM kernel hard-codes group_size=128, got {self.group_size}"
        assert self.activation_bit_width == 8, \
            f"MixLLM kernel only supports A8, got {self.activation_bit_width}"
        if self.enable_rmsnorm_fusion and not self.enable_permutation:
            raise ValueError("RMSNorm fusion requires permutation enabled")
        if self.enable_parallel_constraint and not self.enable_permutation:
            raise ValueError("Parallel constraint requires permutation enabled")

    def summary(self) -> str:
        bits = "+".join(f"{b}:{p}%" for b, p in sorted(self.bit_percent.items(), reverse=True))
        return (
            f"SHMQ-Ultimate v2 config:\n"
            f"  Model: {self.model_name}\n"
            f"  Bit allocation (MixLLM, N-axis): {bits}\n"
            f"  Activation: W{self.activation_bit_width}A{self.activation_bit_width}\n"
            f"  Group size: {self.group_size}\n"
            f"  SHMQ permutation (K-axis): {self.enable_permutation}\n"
            f"  RMSNorm fusion: {self.enable_rmsnorm_fusion}\n"
            f"  Parallel constraint: {self.enable_parallel_constraint}\n"
            f"  SmoothQuant: {self.enable_smoothquant} (α={self.smoothquant_alpha})\n"
            f"  AutoRound: {self.enable_autoround} (iters={self.autoround_iters})\n"
            f"  HP ratio (K-axis): {self.hp_ratio}\n"
            f"  Device: {self.device}, dtype: {self.dtype}\n"
            f"  Calibration: {self.calib_dataset} n={self.n_samples} L={self.sequence_length}"
        )


# === Paper defaults ===

PAPER_QWEN7B = SHMQv2Config(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    bit_percent={8: 10, 4: 90},  # MixLLM W4.4A8 (paper §4.2)
    hp_ratio=0.10,                # SHMQ default Ub (paper §4.2)
    intra_hessian_lambda=0.1,     # SHMQ default (paper §4.2)
    n_samples=128,
    sequence_length=2048,
    autoround_iters=200,          # AutoRound paper default
)

QUICK_TEST = SHMQv2Config(
    model_name="Qwen/Qwen2.5-0.5B",
    bit_percent={8: 10, 4: 90},
    n_samples=4,
    sequence_length=128,
    device="cpu",
    dtype="float32",  # CPU smoke test
    enable_autoround=False,  # skip AutoRound for speed
    autoround_iters=2,
    skip_steps=set(),  # run all
)
