"""SHMQ-Ultimate v2 — Main Pipeline Orchestrator.

11-step pipeline that combines:
  - MixLLM (foundation): bit allocation, kernel, vLLM integration
  - SHMQ (innovation): K-axis permutation, RMSNorm fusion, parallel constraint
  - SmoothQuant (pre-processing): activation outlier migration
  - AutoRound (calibration): learnable rounding via SignSGD

Steps:
  0. Load model + calibration data
  1. SmoothQuant: activation outlier migration (modifies weights + norm weights)
  2. Capture activations for sensitivity + AutoRound
  3. SHMQ intra-layer sensitivity (K-axis, per layer)
  4. SHMQ parallel constraint: average sensitivities for q/k/v, up/gate
  5. SHMQ decoupled permutation (Eq. 12): compute K-axis perm per group
  6. Apply K-axis permutation to weights (gather along K)
  7. SHMQ RMSNorm fusion: replace RMSNorm with PermutedRMSNorm
  8. MixLLM bit allocation (N-axis, global loss distance) — KEEP AS IS
  9. AutoRound: optimize V per block (200 steps SignSGD)
 10. MixLLM quantization (GPTQ + clip shrink, applies N-axis split)
 11. Save model for vLLM (or evaluate PPL / zero-shot)

CRITICAL DESIGN INSIGHT:
  - Steps 1-7 are SHMQ-specific PRE-processing on FP16 weights
  - Steps 8-10 are MixLLM's NATIVE pipeline (untouched)
  - The K-axis permutation (step 6) is transparent to MixLLM's kernel
    (kernel walks K in groups of 128, agnostic to channel ordering)
  - The RMSNorm fusion (step 7) ensures activations arrive pre-permuted,
    so MixLLM's kernel receives the correct (permuted) input
"""
from __future__ import annotations
import os
import sys
import time
import logging
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from .config import SHMQv2Config
from .preprocessing.smoothquant import apply_smoothquant_to_model
from .sensitivity.intra_layer import (
    compute_intra_layer_sensitivity,
    compute_parallel_layer_sensitivity,
)
from .permutation.parallel import (
    group_linears_by_parallel_constraint,
    compute_group_sensitivity,
    assign_group_permutation,
)
from .permutation.decoupled import (
    decoupled_permutation,
    apply_k_permutation_to_weight,
    compute_permutation_metric,
)
from .permutation.rmsnorm_fusion import (
    PermutedRMSNorm,
    replace_rmsnorm_with_permuted,
)
from .autoround.sign_sgd import autoround_optimize, autoround_bake
from .mixllm_bridge.adapter import (
    ensure_mixllm_available,
    build_mixllm_config,
    get_calibration_data,
    run_mixllm_allocation,
    run_mixllm_quantize,
    get_named_linears,
    capture_activations,
)

logger = logging.getLogger("shmq_v2")


class SHMQv2Pipeline:
    """Orchestrates the 11-step SHMQ-Ultimate v2 pipeline."""

    def __init__(self, config: SHMQv2Config):
        self.config = config
        self.config.validate()
        self.timings: Dict[str, float] = {}
        self.model: Optional[nn.Module] = None
        self.calib_data: Optional[torch.Tensor] = None
        self.mixllm_config = None
        self.layer_perms: Dict[str, torch.Tensor] = {}
        self.layer_sensitivities: Dict[str, torch.Tensor] = {}

    def run(self) -> nn.Module:
        """Run the full 11-step pipeline. Returns the quantized model."""
        ensure_mixllm_available()
        print(self.config.summary())
        print("\n" + "=" * 70)
        print("  SHMQ-Ultimate v2 — Pipeline Start")
        print("=" * 70)

        self._step0_load()
        if 1 not in self.config.skip_steps: self._step1_smoothquant()
        if 2 not in self.config.skip_steps: self._step2_capture_activations()
        if 3 not in self.config.skip_steps: self._step3_intra_sensitivity()
        if 4 not in self.config.skip_steps: self._step4_parallel_constraint()
        if 5 not in self.config.skip_steps: self._step5_decoupled_permutation()
        if 6 not in self.config.skip_steps: self._step6_apply_permutation()
        if 7 not in self.config.skip_steps: self._step7_rmsnorm_fusion()
        if 8 not in self.config.skip_steps: self._step8_mixllm_allocation()
        if 9 not in self.config.skip_steps: self._step9_autoround()
        if 10 not in self.config.skip_steps: self._step10_mixllm_quantize()
        if 11 not in self.config.skip_steps: self._step11_save_or_eval()

        self._print_summary()
        return self.model

    # === Steps ===

    def _step0_load(self) -> None:
        """Step 0: Load model + calibration data."""
        t0 = time.time()
        print("\n[Step 0] Loading model and calibration data...")

        from .mixllm_bridge.adapter import mixllm_modeling
        torch_dtype = torch.float16 if self.config.dtype == "float16" else torch.float32

        self.model = mixllm_modeling.get_model(
            self.config.model_name,
            torch_dtype=torch_dtype,
            device_map=self.config.device if self.config.device != "cpu" else None,
        )

        self.calib_data = get_calibration_data(
            self.config.model_name,
            n_samples=self.config.n_samples,
            seed=self.config.seed,
            device=self.config.device,
        )

        # Build MixLLM config
        self.mixllm_config = build_mixllm_config(
            bit_percent=self.config.bit_percent,
            group_size=self.config.group_size,
            weight_4bit_asymmetric=self.config.weight_4bit_asymmetric,
            weight_8bit_asymmetric=self.config.weight_8bit_asymmetric,
            gptq_4bit=self.config.gptq_4bit,
            gptq_8bit=self.config.gptq_8bit,
            clip_shrink_4bit=self.config.clip_shrink_4bit,
            clip_shrink_8bit=self.config.clip_shrink_8bit,
            gptq_group_reorder=self.config.gptq_group_reorder,
            activation_bit_width=self.config.activation_bit_width,
            activation_asymmetric=self.config.activation_asymmetric,
        )

        self.timings["step0_load"] = time.time() - t0
        print(f"  Model: {self.config.model_name}")
        print(f"  Calibration: {self.calib_data.shape} ({self.calib_data.numel()} tokens)")
        print(f"  Time: {self.timings['step0_load']:.1f}s")

    def _step1_smoothquant(self) -> None:
        """Step 1: SmoothQuant activation outlier migration."""
        if not self.config.enable_smoothquant:
            print("\n[Step 1] SmoothQuant DISABLED, skipping.")
            return
        t0 = time.time()
        print("\n[Step 1] SmoothQuant: migrating activation outliers to weights...")

        # Need activations to compute scales — capture first
        named_linears = get_named_linears(self.model)
        activations = capture_activations(
            self.model, self.calib_data, named_linears, device=self.config.device
        )
        scales = apply_smoothquant_to_model(
            self.model, activations, alpha=self.config.smoothquant_alpha
        )
        self.timings["step1_smoothquant"] = time.time() - t0
        print(f"  Smoothed {len(scales)} layers (α={self.config.smoothquant_alpha})")
        print(f"  Time: {self.timings['step1_smoothquant']:.1f}s")

    def _step2_capture_activations(self) -> None:
        """Step 2: Capture activations for sensitivity + AutoRound."""
        t0 = time.time()
        print("\n[Step 2] Capturing activations for sensitivity computation...")
        named_linears = get_named_linears(self.model)
        self.activations = capture_activations(
            self.model, self.calib_data, named_linears, device=self.config.device
        )
        self.named_linears = named_linears
        self.timings["step2_capture"] = time.time() - t0
        print(f"  Captured activations for {len(self.activations)} layers")
        print(f"  Time: {self.timings['step2_capture']:.1f}s")

    def _step3_intra_sensitivity(self) -> None:
        """Step 3: SHMQ intra-layer sensitivity (K-axis, per layer)."""
        t0 = time.time()
        print("\n[Step 3] Computing SHMQ intra-layer sensitivity (K-axis)...")
        for name, linear in self.named_linears:
            if name not in self.activations:
                continue
            sens = compute_intra_layer_sensitivity(
                weight=linear.weight.data,
                activation=self.activations[name],
                lambda_damp=self.config.intra_hessian_lambda,
                group_size=self.config.group_size,
            )
            self.layer_sensitivities[name] = sens
        self.timings["step3_sensitivity"] = time.time() - t0
        print(f"  Computed sensitivity for {len(self.layer_sensitivities)} layers")
        print(f"  Time: {self.timings['step3_sensitivity']:.1f}s")

    def _step4_parallel_constraint(self) -> None:
        """Step 4: SHMQ parallel constraint — average sensitivities for q/k/v, up/gate."""
        if not self.config.enable_parallel_constraint:
            print("\n[Step 4] Parallel constraint DISABLED, skipping.")
            return
        t0 = time.time()
        print("\n[Step 4] Applying SHMQ parallel constraint (q/k/v, up/gate)...")
        groups = group_linears_by_parallel_constraint(
            [name for name, _ in self.named_linears]
        )
        self.group_sensitivities: Dict[str, torch.Tensor] = {}
        for group_key, members in groups.items():
            # Only include members that have sensitivities
            avail = [m for m in members if m in self.layer_sensitivities]
            if not avail:
                continue
            self.group_sensitivities[group_key] = compute_group_sensitivity(
                group_key, self.layer_sensitivities, avail
            )
        self.timings["step4_parallel"] = time.time() - t0
        n_groups = len(self.group_sensitivities)
        n_parallel = sum(1 for g in groups.values() if len(g) > 1)
        print(f"  {n_groups} groups ({n_parallel} parallel groups sharing permutations)")
        print(f"  Time: {self.timings['step4_parallel']:.1f}s")

    def _step5_decoupled_permutation(self) -> None:
        """Step 5: SHMQ decoupled permutation (Eq. 12)."""
        if not self.config.enable_permutation:
            print("\n[Step 5] Permutation DISABLED, skipping.")
            return
        t0 = time.time()
        print("\n[Step 5] Computing SHMQ decoupled permutation (Eq. 12)...")
        groups = group_linears_by_parallel_constraint(
            [name for name, _ in self.named_linears]
        )
        for group_key, members in groups.items():
            if group_key not in self.group_sensitivities:
                continue
            group_sens = self.group_sensitivities[group_key]
            # Compute magnitude metric for within-cluster sort
            # Use the first member's weight (parallel members share in_features)
            first_member = members[0]
            linear = dict(self.named_linears)[first_member]
            act = self.activations.get(first_member)
            if act is not None:
                mag = compute_permutation_metric(linear.weight.data, act)
            else:
                mag = group_sens.abs()
            # Compute permutation
            perm = decoupled_permutation(
                sensitivity=group_sens,
                hp_ratio=self.config.hp_ratio,
                group_size=self.config.group_size,
                magnitude=mag,
            )
            # Assign to all members
            group_perms = assign_group_permutation(members, perm)
            self.layer_perms.update(group_perms)
        self.timings["step5_permutation"] = time.time() - t0
        print(f"  Computed permutations for {len(self.layer_perms)} layers")
        print(f"  Time: {self.timings['step5_permutation']:.1f}s")

    def _step6_apply_permutation(self) -> None:
        """Step 6: Apply K-axis permutation to weights."""
        if not self.layer_perms:
            print("\n[Step 6] No permutations to apply, skipping.")
            return
        t0 = time.time()
        print("\n[Step 6] Applying K-axis permutation to weights...")
        n_applied = 0
        for name, linear in self.named_linears:
            if name not in self.layer_perms:
                continue
            perm = self.layer_perms[name]
            linear.weight.data = apply_k_permutation_to_weight(
                linear.weight.data, perm
            )
            n_applied += 1
        self.timings["step6_apply_perm"] = time.time() - t0
        print(f"  Applied permutation to {n_applied} layers")
        print(f"  Time: {self.timings['step6_apply_perm']:.1f}s")

    def _step7_rmsnorm_fusion(self) -> None:
        """Step 7: SHMQ RMSNorm fusion."""
        if not self.config.enable_rmsnorm_fusion:
            print("\n[Step 7] RMSNorm fusion DISABLED, skipping.")
            return
        t0 = time.time()
        print("\n[Step 7] Fusing permutation into RMSNorm...")
        # For each transformer block, find the RMSNorm and apply the
        # corresponding K-axis permutation (from q/k/v group)
        groups = group_linears_by_parallel_constraint(
            [name for name, _ in self.named_linears]
        )
        n_replaced = 0
        for group_key, members in groups.items():
            if group_key not in self.group_sensitivities:
                continue
            # Skip standalone groups (o_proj, down_proj) — their prior "norm"
            # is not a standard RMSNorm (o_proj takes attention output,
            # down_proj takes SiLU(gate)*up)
            if "standalone" in group_key:
                continue
            # Get permutation for this group
            # Find a member that has a perm
            perm = None
            for m in members:
                if m in self.layer_perms:
                    perm = self.layer_perms[m]
                    break
            if perm is None:
                continue
            # Determine which RMSNorm(s) to fuse into
            # For "attn_qkv" group: input_layernorm
            # For "mlp_gate_up" group: post_attention_layernorm
            layer_idx = group_key.split("_")[0]
            if "attn_qkv" in group_key:
                norm_suffix = "input_layernorm"
            elif "mlp_gate_up" in group_key:
                norm_suffix = "post_attention_layernorm"
            else:
                continue
            norm_name = f"model.layers.{layer_idx}.{norm_suffix}"
            try:
                norm_module = self.model.get_submodule(norm_name)
            except (AttributeError, ModuleNotFoundError):
                continue
            # Replace with PermutedRMSNorm
            hidden_size = norm_module.weight.shape[0]
            eps = getattr(norm_module, "eps", 1e-6)
            new_norm = PermutedRMSNorm(
                hidden_size=hidden_size,
                eps=eps,
                perm=perm.to(norm_module.weight.device),
            )
            new_norm.weight.data.copy_(norm_module.weight.data)
            # Set on parent
            path = norm_name.split(".")
            parent = self.model
            for p in path[:-1]:
                parent = getattr(parent, p)
            setattr(parent, path[-1], new_norm)
            n_replaced += 1
        self.timings["step7_rmsnorm"] = time.time() - t0
        print(f"  Replaced {n_replaced} RMSNorm modules with PermutedRMSNorm")
        print(f"  Time: {self.timings['step7_rmsnorm']:.1f}s")

    def _step8_mixllm_allocation(self) -> None:
        """Step 8: MixLLM bit allocation (N-axis, global loss distance)."""
        t0 = time.time()
        print("\n[Step 8] MixLLM bit allocation (global loss distance, N-axis)...")
        self.mixllm_config = run_mixllm_allocation(
            self.config.model_name,
            self.calib_data,
            self.mixllm_config,
            device=self.config.device,
        )
        self.timings["step8_mixllm_alloc"] = time.time() - t0
        n_layers = len(self.mixllm_config.linear_config_map)
        print(f"  Allocated bits for {n_layers} layers")
        print(f"  Time: {self.timings['step8_mixllm_alloc']:.1f}s")

    def _step9_autoround(self) -> None:
        """Step 9: AutoRound — optimize V per block (200 steps SignSGD)."""
        if not self.config.enable_autoround:
            print("\n[Step 9] AutoRound DISABLED, skipping.")
            return
        t0 = time.time()
        print(f"\n[Step 9] AutoRound: optimizing V via SignSGD ({self.config.autoround_iters} iters)...")
        n_optimized = 0
        for name, linear in self.named_linears:
            if name not in self.activations:
                continue
            # Determine target bit width from mixllm config
            # (For mixed-precision, we use the LOWER bit width for AutoRound
            #  since that's where rounding matters most.)
            target_bits = 4  # conservative: optimize for 4-bit
            V = autoround_optimize(
                linear=linear,
                activation=self.activations[name],
                bit_width=target_bits,
                group_size=self.config.group_size,
                n_iters=self.config.autoround_iters,
                lr=self.config.autoround_lr,
                asymmetric=self.config.weight_4bit_asymmetric,
                block_size=self.config.autoround_block_size,
            )
            # Bake V into weight
            baked = autoround_bake(
                linear, V, bit_width=target_bits,
                group_size=self.config.group_size,
                asymmetric=self.config.weight_4bit_asymmetric,
            )
            linear.weight.data = baked.to(linear.weight.dtype).to(linear.weight.device)
            n_optimized += 1
        self.timings["step9_autoround"] = time.time() - t0
        print(f"  Optimized V for {n_optimized} layers")
        print(f"  Time: {self.timings['step9_autoround']:.1f}s")

    def _step10_mixllm_quantize(self) -> None:
        """Step 10: MixLLM quantization (GPTQ + clip shrink)."""
        t0 = time.time()
        print("\n[Step 10] MixLLM quantization (GPTQ + clip shrink, fake)...")
        self.model = run_mixllm_quantize(
            self.model, self.calib_data, self.mixllm_config
        )
        self.timings["step10_mixllm_quant"] = time.time() - t0
        print(f"  Time: {self.timings['step10_mixllm_quant']:.1f}s")

    def _step11_save_or_eval(self) -> None:
        """Step 11: Save model for vLLM or evaluate PPL / zero-shot."""
        t0 = time.time()
        print("\n[Step 11] Save / Evaluate...")
        if self.config.save_dir:
            from .mixllm_bridge.adapter import mixllm_modeling
            from .nn.modules.utils import save_for_vllm  # type: ignore
            # (Saving for vLLM requires the LinearMixLLM4vLLM path, which needs
            #  fake=False in step 10. For now, just save the model state.)
            os.makedirs(self.config.save_dir, exist_ok=True)
            self.model.save_pretrained(self.config.save_dir)
            print(f"  Saved to {self.config.save_dir}")
        if self.config.eval_ppl:
            print("  Evaluating PPL on", self.config.eval_ppl_dataset)
            from .mixllm_bridge.adapter import mixllm_modeling
            from mixllm.evaluation.engine import evaluate_ppl, get_datasets
            _, testdata = get_datasets(
                self.config.eval_ppl_dataset,
                self.config.model_name,
                nsamples=self.config.n_samples,
                seed=self.config.seed,
                device=self.config.device,
            )
            ppls = evaluate_ppl(
                self.model, self.config.model_name,
                [(self.config.eval_ppl_dataset, testdata)],
                device=self.config.device,
            )
            self.ppl_results = ppls
            print(f"  PPL: {ppls}")
        self.timings["step11_eval"] = time.time() - t0
        print(f"  Time: {self.timings['step11_eval']:.1f}s")

    def _print_summary(self) -> None:
        total = sum(self.timings.values())
        print("\n" + "=" * 70)
        print("  SHMQ-Ultimate v2 — Pipeline Complete")
        print("=" * 70)
        for step, t in self.timings.items():
            print(f"  {step:30s} {t:8.2f}s")
        print(f"  {'TOTAL':30s} {total:8.2f}s")
        print("=" * 70)
