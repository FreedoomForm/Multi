"""Configuration for SHMQ-Ultimate.

All hyperparameters from the SHMQ paper (Section 4.1 + Appendix A.3.1).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict


@dataclass
class SHMQConfig:
    """Hyperparameters for SHMQ-Ultimate.

    Reference: SHMQ paper (EMNLP Industry 2025), Section 4.1 + Appendix A.3.1.
    """

    # ------------------------------------------------------------------
    # Quantization format
    # ------------------------------------------------------------------
    #: Bit-width levels — 3 levels {4, 8, 16} for SHMQ-Ultimate.
    #: 16-bit = FP16/BF16 (kept original), 8-bit = INT8, 4-bit = INT4.
    bit_levels: Tuple[int, ...] = (4, 8, 16)

    #: Target ratio of FP16 channels across the model (e.g. 0.05 = 5% FP16).
    target_hp_ratio_16: float = 0.05

    #: Target ratio of INT8 channels across the model (e.g. 0.20 = 20% INT8).
    target_hp_ratio_8: float = 0.20

    #: Base (floor) ratio of INT8 channels — guarantees UB floor (paper: 0.125).
    base_hp_ratio_8: float = 0.125

    #: Target average bits per parameter (computed from ratios if None).
    #: If None, computed as 4*(1-r16-r8) + 8*r8 + 16*r16.
    target_avg_bits: Optional[float] = None

    #: Per-layer intra-cluster ratios (for 4-bit layers).
    #: Fraction of channels within a 4-bit layer that go to FP16 / INT8.
    intra_layer_hp_ratio_16: float = 0.05
    intra_layer_hp_ratio_8: float = 0.20

    #: Legacy field kept for backward compatibility (used by 2-level code paths).
    target_hp_ratio: float = 0.20

    #: Legacy field kept for backward compatibility.
    base_hp_ratio: float = 0.125

    #: Group size for per-group symmetric quantization (paper: 128).
    group_size: int = 128

    #: Whether weight quantization is symmetric (paper: True).
    weight_symmetric: bool = True

    #: Whether activation quantization is symmetric (paper: True).
    activation_symmetric: bool = True

    #: Activation bit-width (paper: 8 for W4.8A8).
    activation_bits: int = 8

    # ------------------------------------------------------------------
    # Sensitivity computation (SHMQ Eq. 6, 7, 10, 11)
    # ------------------------------------------------------------------
    #: Inter-layer Hessian mode: "fisher" (paper default, Eq. 7) or "pyhessian"
    #: (HAWQ-V3 Hutchinson trace — used as ablation in paper Appendix A.2).
    inter_layer_hessian: str = "fisher"

    #: Dampening factor lambda (paper: 0.1, Eq. 10).
    dampening: float = 0.1

    #: Use mean(diag(H)) * lambda for dampening (paper Eq. 10).
    #: If False, use lambda * I (identity).
    use_mean_diag_dampening: bool = True

    # ------------------------------------------------------------------
    # Calibration data
    # ------------------------------------------------------------------
    #: Calibration dataset name (paper: WikiText-2).
    calibration_dataset: str = "wikitext2"

    #: Number of calibration samples (paper: 128).
    n_samples: int = 128

    #: Sequence length per sample (paper: 2048).
    sequence_length: int = 2048

    #: Calibration batch size.
    batch_size: int = 1

    # ------------------------------------------------------------------
    # SmoothQuant pre-processing
    # ------------------------------------------------------------------
    #: SmoothQuant alpha (paper-recommended: 0.5 default; 0.6-0.9 for W8A8 production).
    #: For W4A8, start with 0.5 (more weight-side migration helps W4).
    smooth_alpha: float = 0.5

    #: SmoothQuant scale clamp floor (avoid divide-by-zero).
    smooth_scale_min: float = 1e-5

    # ------------------------------------------------------------------
    # AutoRound (Step 6)
    # ------------------------------------------------------------------
    #: Enable AutoRound SignSGD learnable rounding.
    enable_autoround: bool = True

    #: Number of SignSGD steps (paper-recommended: 200).
    autoround_iters: int = 200

    #: AutoRound learning rate (paper: 5e-3 via 1/iters with iters=200).
    autoround_lr: Optional[float] = None  # None -> 1.0/iters

    #: AutoRound LR schedule (paper: linear decay to 0).
    autoround_lr_schedule: str = "linear"

    #: AutoRound block size for processing (paper: 128).
    autoround_block_size: int = 128

    # ------------------------------------------------------------------
    # SQC calibration (Step 7)
    # ------------------------------------------------------------------
    #: Enable SQC (Salience-Weighted Quantizer Calibration from SliM-LLM).
    enable_sqc: bool = True

    #: SQC salient z-score threshold (SliM-LLM default: 2.0).
    sqc_zscore_threshold: float = 2.0

    #: SQC scale multiplier search range.
    sqc_scale_range: Tuple[float, float] = (0.9, 1.1)

    #: SQC scale multiplier search points per side.
    sqc_scale_search_points: int = 50

    #: SQC salience penalty weight.
    sqc_salience_lambda: float = 1.0

    # ------------------------------------------------------------------
    # GPTQ (Step 8)
    # ------------------------------------------------------------------
    #: GPTQ block size (SliM-LLM default: 128).
    gptq_block_size: int = 128

    #: GPTQ dampening percent (SliM-LLM default: 0.01).
    gptq_percdamp: float = 0.01

    #: GPTQ activation order (False — we use our own decoupled permutation).
    gptq_actorder: bool = False

    # ------------------------------------------------------------------
    # Parallel layer constraint (SHMQ Appendix A.3.1)
    # ------------------------------------------------------------------
    #: Group of layer name suffixes that must share bit allocation.
    #: q/k/v proj share; up/gate proj share.
    parallel_groups: Dict[str, List[str]] = field(default_factory=lambda: {
        "attention": ["q_proj", "k_proj", "v_proj"],
        "ffn": ["up_proj", "gate_proj"],
    })

    #: Layers excluded from quantization (kept FP16).
    excluded_layer_keywords: List[str] = field(default_factory=lambda: [
        "embed_tokens", "lm_head", "norm", "layernorm", "rmsnorm",
    ])

    # ------------------------------------------------------------------
    # Decoupled permutation (SHMQ Eq. 12, Section 3.2.3)
    # ------------------------------------------------------------------
    #: Permutation metric: "act_weight_linf" (paper: product of activation
    #: and weight L-infinity norms).
    permutation_metric: str = "act_weight_linf"

    #: Whether to use magnitude-based sort within Csen/Cinsen clusters.
    permutation_sort_by_magnitude: bool = True

    # ------------------------------------------------------------------
    # ILP solver (Step 3)
    # ------------------------------------------------------------------
    #: ILP solver: "CBC" (default, PULP bundled) or "GLPK" (requires system install).
    ilp_solver: str = "CBC"

    #: ILP time limit (seconds).
    ilp_time_limit: int = 30

    #: Use 3-level {4,8,16} ILP solver (default True).
    #: If False, fall back to 2-level {4,8} solver (legacy).
    use_3level_ilp: bool = True

    # ------------------------------------------------------------------
    # PolyQ ISA-aware quanta matching (Step 3.5)
    # ------------------------------------------------------------------
    #: Enable ISA-aware quanta matching (PolyQ).
    enable_isa_matching: bool = True

    #: Tensor-core tile sizes for each precision.
    isa_tile_16: int = 128
    isa_tile_8:  int = 128
    isa_tile_4:  int = 64

    #: When ISA matching creates a surplus, prefer upgrading channels
    #: to higher precision (True) or downgrading to lower precision (False).
    isa_prefer_upgrade: bool = True

    # ------------------------------------------------------------------
    # Model / runtime
    # ------------------------------------------------------------------
    #: Model name or path.
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"

    #: Device for sensitivity computation ("cpu" or "cuda").
    device: str = "cpu"

    #: Data type for sensitivity computation.
    dtype: str = "float16"

    #: Whether to use flash attention (GPU only).
    use_flash_attention: bool = False

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def __post_init__(self):
        assert self.inter_layer_hessian in ("fisher", "pyhessian"), \
            f"inter_layer_hessian must be 'fisher' or 'pyhessian', got {self.inter_layer_hessian}"
        assert set(self.bit_levels).issubset({4, 8, 16}), \
            f"bit_levels must be subset of {{4, 8, 16}}, got {self.bit_levels}"
        if self.use_3level_ilp:
            assert set(self.bit_levels) == {4, 8, 16}, \
                f"3-level ILP requires bit_levels=(4,8,16), got {self.bit_levels}"
            r16 = self.target_hp_ratio_16
            r8 = self.target_hp_ratio_8
            assert 0.0 <= r16 and 0.0 <= r8 and r16 + r8 <= 1.0, \
                f"ratios r16={r16} + r8={r8} must be in [0, 1]"
        else:
            assert set(self.bit_levels) == {4, 8}, \
                f"2-level ILP requires bit_levels=(4,8), got {self.bit_levels}"
        assert 0.0 <= self.base_hp_ratio_8 <= 1.0, "base_hp_ratio_8 must be in [0, 1]"
        assert 0.0 <= self.target_hp_ratio_8 <= 1.0, "target_hp_ratio_8 must be in [0, 1]"
        if self.autoround_lr is None:
            object.__setattr__(self, "autoround_lr", 1.0 / self.autoround_iters)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    @property
    def Ut(self) -> float:
        """Target high-precision ratio."""
        return self.target_hp_ratio

    @property
    def Ub(self) -> float:
        """Base high-precision ratio."""
        return self.base_hp_ratio

    @property
    def lambda_(self) -> float:
        """Dampening factor (Python keyword workaround)."""
        return self.dampening

    @property
    def computed_target_avg_bits(self) -> float:
        """Compute target average bits from r16/r8 ratios."""
        if self.target_avg_bits is not None:
            return self.target_avg_bits
        r16 = self.target_hp_ratio_16
        r8 = self.target_hp_ratio_8
        r4 = 1.0 - r16 - r8
        return 4.0 * r4 + 8.0 * r8 + 16.0 * r16

    def summary(self) -> str:
        """Human-readable summary."""
        if self.use_3level_ilp:
            fmt = (f"W{self.computed_target_avg_bits:.2f}A{self.activation_bits} "
                   f"(4-bit: {(1-self.target_hp_ratio_16-self.target_hp_ratio_8)*100:.1f}% "
                   f"+ 8-bit: {self.target_hp_ratio_8*100:.1f}% "
                   f"+ 16-bit: {self.target_hp_ratio_16*100:.1f}%)")
        else:
            fmt = (f"W{4 + 4 * self.target_hp_ratio:.1f}A{self.activation_bits} "
                   f"(W4A{self.activation_bits} + {self.target_hp_ratio*100:.1f}% W8A{self.activation_bits})")
        return (
            f"SHMQ-Ultimate Config:\n"
            f"  Format: {fmt}\n"
            f"  Bit levels: {self.bit_levels}\n"
            f"  Group size: {self.group_size}\n"
            f"  UB (base HP ratio 8): {self.base_hp_ratio_8}\n"
            f"  Ut (target HP ratio 8): {self.target_hp_ratio_8}\n"
            f"  Ut16 (target HP ratio 16): {self.target_hp_ratio_16}\n"
            f"  Target avg bits: {self.computed_target_avg_bits:.3f}\n"
            f"  lambda (dampening): {self.dampening}\n"
            f"  Inter-layer Hessian: {self.inter_layer_hessian}\n"
            f"  SmoothQuant alpha: {self.smooth_alpha}\n"
            f"  AutoRound: iters={self.autoround_iters}, lr={self.autoround_lr:.4f}\n"
            f"  SQC: {self.enable_sqc}\n"
            f"  ISA matching: {self.enable_isa_matching}\n"
            f"  Calibration: {self.n_samples} samples x {self.sequence_length} tokens "
            f"from {self.calibration_dataset}\n"
            f"  Model: {self.model_name}\n"
            f"  Device: {self.device}\n"
        )
