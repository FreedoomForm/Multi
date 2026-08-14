"""SHMQ-Ultimate pipeline orchestrator (3-level {4, 8, 16} + MixLLM kernel).

Runs the full 11-step SHMQ-Ultimate pipeline:
1.  SmoothQuant pre-processing
2.  Inter-layer Fisher sensitivity + intra-layer OBS + Manhattan + parallel
3.  3-level ILP bit allocation {4, 8, 16}  (HAWQ-V3, PULP)
3.5 PolyQ ISA-aware quanta matching       (round cluster sizes to tensor-core tiles)
4.  Decoupled permutation into 3 clusters C16/C8/C4  (SHMQ Eq. 12, extended)
5.  Permutation fusion into RMSNorm       (SHMQ §3.2 + PolyQ layout propagation)
6.  AutoRound SignSGD (200 steps per block)
7.  SQC calibration
8.  GPTQ + mixed INT4/INT8 quantization   (for the INT4/INT8 channels only)
9.  Convert fake-quant → real MixLLM kernel + FP16 path
    (replaces the old custom CUDA kernel with MixLLM's production kernel)
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
from .ilp import solve_ilp_3level, ILPResult3L, compute_target_avg_bits
from .polyq import apply_isa_matching, ISAMatchResult
from .permutation import (
    compute_permutation_metric,
    apply_permutation_to_parallel_layers,
    apply_permutation_to_parallel_layers_3level,
    fuse_permutation_into_rmsnorm,
    capture_input_activations,
)
from .autoround import autoround_block
from .quantize import SQCCalibrator, MixedPrecisionQuantizer
from .utils import symmetric_quantize_weights, compute_quant_error, get_module_by_name
from .mixllm import convert_model_to_mixllm, ConversionSummary, is_mixllm_available


class SHMQPipeline:
    """Main SHMQ-Ultimate pipeline orchestrator (3-level {4,8,16} + MixLLM).

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
        self.ilp_result: Optional[ILPResult3L] = None
        self.isa_result: Optional[ISAMatchResult] = None
        self.bit_allocation: Dict[str, int] = {}
        self.permutation_indices: Dict[str, torch.Tensor] = {}
        self.cluster_sizes: Dict[str, Dict[int, int]] = {}
        self.permutation_metrics: Dict[str, torch.Tensor] = {}
        self.captured_activations: Dict[str, List[torch.Tensor]] = {}
        self.sqc_multipliers: Dict[str, float] = {}
        self.conversion_summary: Optional[ConversionSummary] = None

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
        print(f"[step0] MixLLM kernel available: {is_mixllm_available()}")

    def _build_parallel_groups(self) -> Dict[str, List[str]]:
        """Build parallel groups dict from LayerInfo."""
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
        layers_to_smooth = [n for n in self.layer_names
                            if any(s in n for s in ("q_proj", "k_proj", "v_proj",
                                                      "gate_proj", "up_proj"))]
        self.act_scales = get_act_scales(
            self.model, layers_to_smooth, self.calibration_data,
            batch_size=self.config.batch_size,
        )
        smooth_scales = smooth_lm(
            self.model, layers_to_smooth, self.act_scales,
            alpha=self.config.smooth_alpha,
            scale_min=self.config.smooth_scale_min,
        )
        t1 = time.time()
        print(f"[step1] Smoothed {len(layers_to_smooth)} layers in {t1-t0:.1f}s "
              f"(alpha={self.config.smooth_alpha})")

    # ------------------------------------------------------------------
    # Step 2: Sensitivity (Fisher + OBS + Manhattan + parallel)
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
    # Step 3: 3-level ILP bit allocation
    # ------------------------------------------------------------------
    def step3_ilp(self):
        print("\n" + "=" * 70)
        print("STEP 3: 3-level ILP bit allocation {4, 8, 16}")
        print("=" * 70)
        t0 = time.time()

        n_params: Dict[str, int] = {}
        qerr_4bit: Dict[str, float] = {}
        qerr_8bit: Dict[str, float] = {}
        qerr_16bit: Dict[str, float] = {}
        for name in self.layer_names:
            mod = self.model_loader.get_layer(name).module
            n_params[name] = mod.weight.numel()
            qerr_4bit[name]  = compute_quant_error(mod.weight.data, n_bits=4,
                                                    group_size=self.config.group_size)
            qerr_8bit[name]  = compute_quant_error(mod.weight.data, n_bits=8,
                                                    group_size=self.config.group_size)
            qerr_16bit[name] = 0.0  # FP16 is lossless

        target_avg = self.config.computed_target_avg_bits
        min_avg = 4 + 4 * self.config.base_hp_ratio_8  # floor: at least base ratio at 8-bit

        self.ilp_result = solve_ilp_3level(
            layer_names=self.layer_names,
            sensitivities=self.inter_layer_sensitivities,
            n_params=n_params,
            quant_error_4bit=qerr_4bit,
            quant_error_8bit=qerr_8bit,
            quant_error_16bit=qerr_16bit,
            target_avg_bits=target_avg,
            min_avg_bits=min_avg,
            parallel_groups=self.parallel_groups,
            solver=self.config.ilp_solver,
            time_limit=self.config.ilp_time_limit,
            verbose=False,
        )
        self.bit_allocation = self.ilp_result.bit_allocation
        t1 = time.time()
        print(self.ilp_result.summary())
        print(f"[step3] 3-level ILP solved in {t1-t0:.2f}s")
        print(f"[step3] Target avg bits: {target_avg:.3f}, achieved: {self.ilp_result.total_bits:.3f}")

    # ------------------------------------------------------------------
    # Step 3.5: PolyQ ISA-aware quanta matching
    # ------------------------------------------------------------------
    def step3_5_isa_matching(self):
        if not self.config.enable_isa_matching:
            print("\n[step3.5] ISA matching disabled, skipping")
            return
        print("\n" + "=" * 70)
        print("STEP 3.5: PolyQ ISA-aware quanta matching")
        print("=" * 70)
        t0 = time.time()
        # Build initial ratios per layer based on bit_allocation
        out_features: Dict[str, int] = {}
        initial_ratios: Dict[str, Dict[int, float]] = {}
        for name in self.layer_names:
            mod = self.model_loader.get_layer(name).module
            n_out = mod.weight.shape[0]
            out_features[name] = n_out
            bits = self.bit_allocation.get(name, 4)
            if bits == 16:
                initial_ratios[name] = {16: 1.0, 8: 0.0, 4: 0.0}
            elif bits == 8:
                initial_ratios[name] = {16: 0.0, 8: 1.0, 4: 0.0}
            else:
                initial_ratios[name] = {
                    16: self.config.intra_layer_hp_ratio_16,
                    8:  self.config.intra_layer_hp_ratio_8,
                    4:  1.0 - self.config.intra_layer_hp_ratio_16 - self.config.intra_layer_hp_ratio_8,
                }
        self.isa_result = apply_isa_matching(
            layer_names=self.layer_names,
            out_features=out_features,
            initial_ratios=initial_ratios,
            avg_bits_budget=self.config.computed_target_avg_bits,
            prefer_upgrade=self.config.isa_prefer_upgrade,
            verbose=False,
        )
        t1 = time.time()
        print(self.isa_result.summary())
        print(f"[step3.5] ISA matching done in {t1-t0:.2f}s")

    # ------------------------------------------------------------------
    # Step 4: 3-level decoupled permutation
    # ------------------------------------------------------------------
    def step4_permutation(self):
        print("\n" + "=" * 70)
        print("STEP 4: 3-level decoupled permutation (C16/C8/C4)")
        print("=" * 70)
        t0 = time.time()
        # Capture input activations (needed for permutation metric AND GPTQ later)
        self.captured_activations = capture_input_activations(
            self.model, self.layer_names, self.calibration_data,
            batch_size=self.config.batch_size,
        )
        # Compute permutation metric per layer
        for name in self.layer_names:
            mod = get_module_by_name(self.model, name)
            acts = self.captured_activations.get(name, [])
            self.permutation_metrics[name] = compute_permutation_metric(
                mod.weight.data, acts,
            )
        # Apply 3-level decoupled permutation (with parallel constraint)
        # Use ISA-matched cluster sizes if available
        cluster_sizes_for_perm = self.isa_result.cluster_sizes if self.isa_result else None
        # The 3-level permutation function uses intra-layer ratios for 4-bit layers,
        # but if we have ISA-matched cluster sizes, we should use them.
        # For now, the function uses ratios — the ISA matching happens at the
        # MixLLM conversion step (step 9) where we pass cluster_sizes explicitly.
        self.permutation_indices, self.cluster_sizes = apply_permutation_to_parallel_layers_3level(
            model=self.model,
            parallel_groups=self.parallel_groups,
            channel_sensitivities=self.channel_sensitivities,
            permutation_metrics=self.permutation_metrics,
            bit_allocation=self.bit_allocation,
            intra_layer_hp_ratio_16=self.config.intra_layer_hp_ratio_16,
            intra_layer_hp_ratio_8=self.config.intra_layer_hp_ratio_8,
            tile_16=self.config.isa_tile_16,
            tile_8=self.config.isa_tile_8,
            tile_4=self.config.isa_tile_4,
            all_layer_names=self.layer_names,
        )
        # If ISA matching was done, override cluster_sizes with ISA-matched sizes
        if self.isa_result is not None:
            self.cluster_sizes = self.isa_result.cluster_sizes
        t1 = time.time()
        n_c16 = sum(cs.get(16, 0) for cs in self.cluster_sizes.values())
        n_c8  = sum(cs.get(8, 0)  for cs in self.cluster_sizes.values())
        n_c4  = sum(cs.get(4, 0)  for cs in self.cluster_sizes.values())
        print(f"[step4] 3-level permutation applied to {len(self.permutation_indices)} layers "
              f"in {t1-t0:.1f}s")
        print(f"[step4] Cluster totals: C16={n_c16}, C8={n_c8}, C4={n_c4} channels")

    # ------------------------------------------------------------------
    # Step 5: Permutation fusion into RMSNorm
    # ------------------------------------------------------------------
    def step5_rmsnorm_fusion(self):
        print("\n" + "=" * 70)
        print("STEP 5: Permutation fusion into RMSNorm (PolyQ layout propagation)")
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
        t1 = time.time()
        print(f"[step7] SQC calibrated {len(self.sqc_multipliers)} layers in {t1-t0:.1f}s")

    # ------------------------------------------------------------------
    # Step 8: GPTQ + mixed INT4/INT8 quantization (fake quant)
    # ------------------------------------------------------------------
    def step8_quantize(self):
        print("\n" + "=" * 70)
        print("STEP 8: GPTQ + mixed INT4/INT8 fake quantization")
        print("=" * 70)
        t0 = time.time()
        quantizer = MixedPrecisionQuantizer(
            group_size=self.config.group_size,
            percdamp=self.config.gptq_percdamp,
            blocksize=self.config.gptq_block_size,
            activation_bits=self.config.activation_bits,
        )
        # Re-capture activations post-fusion
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
        n_4bit  = sum(1 for r in results.values() if r["n_bits"] == 4)
        n_8bit  = sum(1 for r in results.values() if r["n_bits"] == 8)
        n_16bit = sum(1 for r in results.values() if r["n_bits"] == 16)
        print(f"[step8] Fake-quantized {len(results)} layers "
              f"(INT4: {n_4bit}, INT8: {n_8bit}, FP16: {n_16bit}) in {t1-t0:.1f}s")

    # ------------------------------------------------------------------
    # Step 9: Convert to real MixLLM kernel + FP16 path
    # ------------------------------------------------------------------
    def step9_mixllm_conversion(self):
        """Replace every nn.Linear with a SHMQMixLLMLinear that combines
        FP16 + INT8 + INT4 weight paths.

        This is what gives SHMQ-Ultimate its speedup — MixLLM's production
        CUDA kernel handles the INT4+INT8 GEMM, while FP16 channels go
        through standard cuBLAS.
        """
        print("\n" + "=" * 70)
        print("STEP 9: MixLLM kernel conversion (FP16 + INT8 + INT4)")
        print("=" * 70)
        t0 = time.time()
        self.conversion_summary = convert_model_to_mixllm(
            model=self.model,
            layer_names=self.layer_names,
            bit_allocation=self.bit_allocation,
            permutation_indices=self.permutation_indices,
            cluster_sizes=self.cluster_sizes,
            intra_layer_hp_ratio_8=self.config.intra_layer_hp_ratio_8,
            intra_layer_hp_ratio_16=self.config.intra_layer_hp_ratio_16,
            group_size=self.config.group_size,
            verbose=True,
        )
        t1 = time.time()
        print(self.conversion_summary)
        print(f"[step9] MixLLM conversion done in {t1-t0:.1f}s")

    # ------------------------------------------------------------------
    # Run all steps
    # ------------------------------------------------------------------
    def run(self, skip_steps: Optional[List[int]] = None):
        """Run the full SHMQ-Ultimate pipeline.

        Args:
            skip_steps: list of step numbers to skip (0-9). Default: None (run all).
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
        # Step 3.5 is invoked via step3_5_isa_matching (use string "3.5" to skip)
        if "3.5" not in skip_steps:
            self.step3_5_isa_matching()
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
        if 9 not in skip_steps:
            self.step9_mixllm_conversion()
        print("\n" + "=" * 70)
        print("SHMQ-Ultimate pipeline COMPLETE (3-level {4,8,16} + MixLLM kernel)")
        print("=" * 70)
        print(f"Model: {self.config.model_name}")
        if self.ilp_result:
            print(self.ilp_result.summary())
        if self.conversion_summary:
            print(self.conversion_summary)
        print(f"Inference backend: {'MixLLM CUDA kernel' if is_mixllm_available() else 'PyTorch fallback'}")

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
        import json
        meta = {
            "config": self.config.__dict__,
            "bit_allocation": self.bit_allocation,
            "cluster_sizes": {k: {str(bk): bv for bk, bv in v.items()}
                              for k, v in self.cluster_sizes.items()},
            "sqc_multipliers": self.sqc_multipliers,
            "mixllm_available": is_mixllm_available(),
        }
        with open(os.path.join(output_dir, "shmq_config.json"), "w") as f:
            json.dump(meta, f, indent=2, default=str)
        print(f"[save] Done.")
