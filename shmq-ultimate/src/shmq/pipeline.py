"""SHMQ-Ultimate pipeline orchestrator.

Runs the full 9-step SHMQ-Ultimate pipeline:
1. SmoothQuant pre-processing
2. Inter-layer Fisher sensitivity + intra-layer OBS + Manhattan + parallel constraint
3. ILP bit allocation {4, 8}
4. Decoupled permutation
5. Permutation fusion into RMSNorm
6. AutoRound SignSGD (200 steps per block)
7. SQC calibration
8. GPTQ + mixed INT4/INT8 quantization
9. (Inference is done by the user after quantization)
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
import time
import torch
import torch.nn as nn

from .config import SHMQConfig
from .model_loader import ModelLoader, LayerInfo
from .calibration import get_calibration_data, iter_batches
from .smooth import smooth_lm, get_act_scales
from .sensitivity import (
    compute_inter_layer_fisher_sensitivity,
    compute_inter_layer_pyhessian_trace,
    compute_intra_layer_obs_sensitivity,
    aggregate_manhattan_channel_sensitivity,
    average_inter_layer_parallel_sensitivity,
    concatenate_intra_layer_parallel_sensitivity,
)
from .ilp import solve_ilp_bit_allocation, ILPResult
from .permutation import (
    compute_permutation_metric,
    apply_permutation_to_parallel_layers,
    fuse_permutation_into_rmsnorm,
    capture_input_activations,
)
from .autoround import autoround_block
from .quantize import SQCCalibrator, MixedPrecisionQuantizer
from .utils import symmetric_quantize_weights, compute_quant_error


class SHMQPipeline:
    """Main SHMQ-Ultimate pipeline orchestrator.

    Usage:
        config = SHMQConfig(model_name="Qwen/Qwen2.5-7B-Instruct")
        pipeline = SHMQPipeline(config)
        pipeline.run()
        # model is now quantized; save or evaluate
    """

    def __init__(self, config: SHMQConfig):
        self.config = config
        self.model_loader: Optional[ModelLoader] = None
        self.model: Optional[nn.Module] = None
        self.tokenizer = None
        self.calibration_data: Optional[torch.Tensor] = None
        self.layer_infos: List[LayerInfo] = []
        self.layer_names: List[str] = []
        self.parallel_groups: Dict[str, List[str]] = {}

        # Results from each step
        self.act_scales: Dict[str, torch.Tensor] = {}
        self.inter_layer_sensitivities: Dict[str, float] = {}
        self.intra_layer_sensitivities: Dict[str, torch.Tensor] = {}
        self.channel_sensitivities: Dict[str, torch.Tensor] = {}
        self.ilp_result: Optional[ILPResult] = None
        self.bit_allocation: Dict[str, int] = {}
        self.permutation_indices: Dict[str, torch.Tensor] = {}
        self.permutation_metrics: Dict[str, torch.Tensor] = {}
        self.captured_activations: Dict[str, List[torch.Tensor]] = {}
        self.sqc_multipliers: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Step 0: Load model + calibration data
    # ------------------------------------------------------------------
    def step0_load(self):
        print("\n" + "=" * 70)
        print("STEP 0: Load model and calibration data")
        print("=" * 70)
        print(self.config.summary())

        dtype = getattr(torch, self.config.dtype)
        self.model_loader = ModelLoader(
            model_name=self.config.model_name,
            device=self.config.device, dtype=dtype,
        )
        self.model, self.tokenizer = self.model_loader.load()

        # Filter out excluded layers
        excluded = tuple(self.config.excluded_layer_keywords)
        self.layer_infos = [l for l in self.model_loader.layers
                            if not any(kw in l.name.lower() for kw in excluded)]
        self.layer_names = [l.name for l in self.layer_infos]
        self.parallel_groups = self._build_parallel_groups()

        # Load calibration data
        self.calibration_data = get_calibration_data(
            self.config.calibration_dataset, self.tokenizer,
            nsamples=self.config.n_samples, seqlen=self.config.sequence_length,
            device=self.config.device,
        )
        print(f"[step0] Loaded {len(self.layer_names)} layers, "
              f"{len(self.parallel_groups)} parallel groups, "
              f"{self.calibration_data.shape[0]} calibration samples")

    def _build_parallel_groups(self) -> Dict[str, List[str]]:
        """Build parallel groups dict from LayerInfo.

        Returns: {group_key: [layer_name1, layer_name2, ...]}
        """
        groups: Dict[str, List[str]] = {}
        for layer in self.layer_infos:
            if layer.is_parallel:
                groups.setdefault(layer.parallel_group_key, []).append(layer.name)
        return groups

    # ------------------------------------------------------------------
    # Step 1: SmoothQuant pre-processing
    # ------------------------------------------------------------------
    def step1_smoothquant(self):
        print("\n" + "=" * 70)
        print("STEP 1: SmoothQuant pre-processing")
        print("=" * 70)
        t0 = time.time()
        # Identify layers to smooth (q/k/v/gate/up only)
        layers_to_smooth = [n for n in self.layer_names
                            if any(s in n for s in ("q_proj", "k_proj", "v_proj",
                                                      "gate_proj", "up_proj"))]
        # Capture activation scales
        self.act_scales = get_act_scales(
            self.model, layers_to_smooth, self.calibration_data,
            batch_size=self.config.batch_size,
        )
        # Apply smoothing
        smooth_scales = smooth_lm(
            self.model, layers_to_smooth, self.act_scales,
            alpha=self.config.smooth_alpha,
            scale_min=self.config.smooth_scale_min,
        )
        t1 = time.time()
        print(f"[step1] Smoothed {len(layers_to_smooth)} layers in {t1-t0:.1f}s "
              f"(alpha={self.config.smooth_alpha})")

    # ------------------------------------------------------------------
    # Step 2: Sensitivity computation (Fisher + OBS + Manhattan + parallel)
    # ------------------------------------------------------------------
    def step2_sensitivity(self):
        print("\n" + "=" * 70)
        print("STEP 2: Sensitivity computation (inter-layer Fisher + intra-layer OBS)")
        print("=" * 70)
        t0 = time.time()

        # 2a. Inter-layer sensitivity
        if self.config.inter_layer_hessian == "fisher":
            self.inter_layer_sensitivities = compute_inter_layer_fisher_sensitivity(
                self.model, self.layer_names, self.calibration_data,
                n_bits=4, group_size=self.config.group_size,
                batch_size=self.config.batch_size,
            )
        else:
            self.inter_layer_sensitivities = compute_inter_layer_pyhessian_trace(
                self.model, self.layer_names, self.calibration_data,
                batch_size=self.config.batch_size,
                max_samples=4,
            )
        # Apply parallel constraint (average within group)
        self.inter_layer_sensitivities = average_inter_layer_parallel_sensitivity(
            self.inter_layer_sensitivities, self.parallel_groups,
        )

        t1 = time.time()
        print(f"[step2a] Inter-layer sensitivity computed for {len(self.inter_layer_sensitivities)} layers "
              f"in {t1-t0:.1f}s")

        # 2b. Intra-layer per-element OBS sensitivity
        self.intra_layer_sensitivities = compute_intra_layer_obs_sensitivity(
            self.model, self.layer_names, self.calibration_data,
            n_bits=4, group_size=self.config.group_size,
            dampening=self.config.dampening,
            use_mean_diag=self.config.use_mean_diag_dampening,
            batch_size=self.config.batch_size,
        )

        t2 = time.time()
        print(f"[step2b] Intra-layer OBS sensitivity computed in {t2-t1:.1f}s")

        # 2c. Manhattan norm + parallel constraint (concat then Manhattan)
        self.channel_sensitivities = concatenate_intra_layer_parallel_sensitivity(
            self.intra_layer_sensitivities, self.parallel_groups,
        )

        t3 = time.time()
        print(f"[step2c] Channel sensitivity (Manhattan + parallel concat) in {t3-t2:.1f}s")

    # ------------------------------------------------------------------
    # Step 3: ILP bit allocation
    # ------------------------------------------------------------------
    def step3_ilp(self):
        print("\n" + "=" * 70)
        print("STEP 3: ILP bit allocation {4, 8}")
        print("=" * 70)
        t0 = time.time()

        # Compute per-layer n_params and quant errors (||W-Q4||^2 and ||W-Q8||^2)
        n_params: Dict[str, int] = {}
        qerr_4bit: Dict[str, float] = {}
        qerr_8bit: Dict[str, float] = {}
        for name in self.layer_names:
            mod = self.model_loader.get_layer(name).module
            n_params[name] = mod.weight.numel()
            qerr_4bit[name] = compute_quant_error(mod.weight.data, n_bits=4,
                                                  group_size=self.config.group_size)
            qerr_8bit[name] = compute_quant_error(mod.weight.data, n_bits=8,
                                                  group_size=self.config.group_size)

        self.ilp_result = solve_ilp_bit_allocation(
            layer_names=self.layer_names,
            sensitivities=self.inter_layer_sensitivities,
            n_params=n_params,
            quant_error_4bit=qerr_4bit,
            quant_error_8bit=qerr_8bit,
            target_hp_ratio=self.config.target_hp_ratio,
            base_hp_ratio=self.config.base_hp_ratio,
            parallel_groups=self.parallel_groups,
            solver=self.config.ilp_solver,
            time_limit=self.config.ilp_time_limit,
            verbose=False,
        )
        self.bit_allocation = self.ilp_result.bit_allocation
        t1 = time.time()
        print(self.ilp_result.summary())
        print(f"[step3] ILP solved in {t1-t0:.2f}s")

    # ------------------------------------------------------------------
    # Step 4: Decoupled permutation
    # ------------------------------------------------------------------
    def step4_permutation(self):
        print("\n" + "=" * 70)
        print("STEP 4: Decoupled permutation")
        print("=" * 70)
        t0 = time.time()
        # Capture input activations (needed for permutation metric AND GPTQ later)
        self.captured_activations = capture_input_activations(
            self.model, self.layer_names, self.calibration_data,
            batch_size=self.config.batch_size,
        )
        # Compute permutation metric per layer
        from .utils import get_module_by_name
        for name in self.layer_names:
            mod = get_module_by_name(self.model, name)
            acts = self.captured_activations.get(name, [])
            self.permutation_metrics[name] = compute_permutation_metric(
                mod.weight.data, acts,
            )
        # Apply decoupled permutation (with parallel constraint)
        self.permutation_indices = apply_permutation_to_parallel_layers(
            self.model, self.parallel_groups,
            self.channel_sensitivities, self.permutation_metrics,
            self.bit_allocation, group_size=self.config.group_size,
        )
        t1 = time.time()
        print(f"[step4] Permutation applied to {len(self.permutation_indices)} layers "
              f"in {t1-t0:.1f}s")

    # ------------------------------------------------------------------
    # Step 5: Permutation fusion into RMSNorm
    # ------------------------------------------------------------------
    def step5_rmsnorm_fusion(self):
        print("\n" + "=" * 70)
        print("STEP 5: Permutation fusion into RMSNorm")
        print("=" * 70)
        t0 = time.time()
        fused_log = fuse_permutation_into_rmsnorm(
            self.model, self.permutation_indices,
        )
        t1 = time.time()
        print(f"[step5] Fused permutation into {len(fused_log)} RMSNorms in {t1-t0:.1f}s")
        for n, msg in list(fused_log.items())[:5]:
            print(f"  - {n}: {msg}")
        if len(fused_log) > 5:
            print(f"  ... ({len(fused_log) - 5} more)")

    # ------------------------------------------------------------------
    # Step 6: AutoRound SignSGD
    # ------------------------------------------------------------------
    def step6_autoround(self):
        if not self.config.enable_autoround:
            print("\n[step6] AutoRound disabled, skipping")
            return
        print("\n" + "=" * 70)
        print("STEP 6: AutoRound SignSGD learnable rounding")
        print("=" * 70)
        t0 = time.time()
        # Get blocks
        blocks = self.model_loader.get_transformer_blocks()
        block_to_layers: Dict[int, List[str]] = {i: [] for i in range(len(blocks))}
        for layer in self.layer_infos:
            if layer.block_idx >= 0:
                block_to_layers[layer.block_idx].append(layer.name)

        n_processed = 0
        for block_idx, layer_names_in_block in block_to_layers.items():
            if not layer_names_in_block:
                continue
            wrappers = autoround_block(
                self.model, block_idx, layer_names_in_block,
                self.calibration_data,
                n_bits=4, group_size=self.config.group_size,
                iters=self.config.autoround_iters,
                lr=self.config.autoround_lr,
                batch_size=self.config.batch_size,
                max_samples=min(8, self.config.n_samples),
                n_bits_per_layer=self.bit_allocation,
                verbose=False,
            )
            n_processed += len(wrappers)
        t1 = time.time()
        print(f"[step6] AutoRound applied to {n_processed} 4-bit layers in {t1-t0:.1f}s")

    # ------------------------------------------------------------------
    # Step 7: SQC calibration
    # ------------------------------------------------------------------
    def step7_sqc(self):
        if not self.config.enable_sqc:
            print("\n[step7] SQC disabled, skipping")
            return
        print("\n" + "=" * 70)
        print("STEP 7: SQC (Salience-Weighted Quantizer Calibration)")
        print("=" * 70)
        t0 = time.time()
        sqc = SQCCalibrator(
            zscore_threshold=self.config.sqc_zscore_threshold,
            scale_range=self.config.sqc_scale_range,
            search_points=self.config.sqc_scale_search_points,
            salience_lambda=self.config.sqc_salience_lambda,
        )
        self.sqc_multipliers = sqc.calibrate_model(
            self.model, self.layer_names,
            sensitivities=self.intra_layer_sensitivities,
            n_bits_per_layer=self.bit_allocation,
            group_size=self.config.group_size,
        )
        # Apply multipliers to weights (multiply weight by multiplier to use the calibrated scale)
        # Note: SQC just picks a better scale multiplier; we don't need to modify weights.
        # The multiplier is used during final quantization.
        t1 = time.time()
        print(f"[step7] SQC calibrated {len(self.sqc_multipliers)} layers in {t1-t0:.1f}s")

    # ------------------------------------------------------------------
    # Step 8: GPTQ + mixed INT4/INT8 quantization
    # ------------------------------------------------------------------
    def step8_quantize(self):
        print("\n" + "=" * 70)
        print("STEP 8: GPTQ + mixed INT4/INT8 quantization")
        print("=" * 70)
        t0 = time.time()
        quantizer = MixedPrecisionQuantizer(
            group_size=self.config.group_size,
            percdamp=self.config.gptq_percdamp,
            blocksize=self.config.gptq_block_size,
            activation_bits=self.config.activation_bits,
        )
        # Re-capture activations (the model has been modified by permutation + autoround)
        # — but since we already have them from step 4 and they don't change much,
        # we can reuse. Note: ideally we'd re-capture, but for efficiency we skip.
        # WARNING: if permutation was applied, the activation columns are also permuted.
        # Since the RMSNorm fusion handles this, the activations captured AFTER
        # fusion would have the permuted order. We need to re-capture post-fusion.
        # For simplicity, we re-capture here.
        post_fusion_activations = capture_input_activations(
            self.model, self.layer_names, self.calibration_data,
            batch_size=self.config.batch_size,
        )
        results = quantizer.apply(
            self.model, self.layer_names,
            self.bit_allocation, post_fusion_activations,
            use_gptq_for_4bit=True,
        )
        t1 = time.time()
        n_4bit = sum(1 for r in results.values() if r["n_bits"] == 4)
        n_8bit = sum(1 for r in results.values() if r["n_bits"] == 8)
        print(f"[step8] Quantized {len(results)} layers ({n_4bit} INT4, {n_8bit} INT8) "
              f"in {t1-t0:.1f}s")

    # ------------------------------------------------------------------
    # Run all steps
    # ------------------------------------------------------------------
    def run(self, skip_steps: Optional[List[int]] = None):
        """Run the full SHMQ-Ultimate pipeline.

        Args:
            skip_steps: list of step numbers to skip (0-8). Default: None (run all).
        """
        skip_steps = skip_steps or []
        if 0 not in skip_steps:
            self.step0_load()
        if 1 not in skip_steps:
            self.step1_smoothquant()
        if 2 not in skip_steps:
            self.step2_sensitivity()
        if 3 not in skip_steps:
            self.step3_ilp()
        if 4 not in skip_steps:
            self.step4_permutation()
        if 5 not in skip_steps:
            self.step5_rmsnorm_fusion()
        if 6 not in skip_steps:
            self.step6_autoround()
        if 7 not in skip_steps:
            self.step7_sqc()
        if 8 not in skip_steps:
            self.step8_quantize()
        print("\n" + "=" * 70)
        print("SHMQ-Ultimate pipeline COMPLETE")
        print("=" * 70)
        print(f"Model: {self.config.model_name}")
        print(f"Format: W{4 + 4*self.config.target_hp_ratio:.1f}A{self.config.activation_bits}")
        if self.ilp_result:
            print(self.ilp_result.summary())

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------
    def save_model(self, output_dir: str):
        """Save the quantized model and tokenizer."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        print(f"[save] Saving quantized model to {output_dir}")
        self.model.save_pretrained(output_dir, safe_serialization=True)
        self.tokenizer.save_pretrained(output_dir)
        # Save SHMQ metadata
        import json
        meta = {
            "config": self.config.__dict__,
            "bit_allocation": self.bit_allocation,
            "sqc_multipliers": self.sqc_multipliers,
        }
        with open(os.path.join(output_dir, "shmq_config.json"), "w") as f:
            json.dump(meta, f, indent=2, default=str)
        print(f"[save] Done.")
