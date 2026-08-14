"""SHMQ-Ultimate 11-step quantization pipeline.

  1. SmoothQuant           — migrate activation outliers into weights
  2. Calibration capture   — H = XX^T + input samples per linear
  3. Inter-layer sens.     — Fisher (SHMQ Eq. 6-7) or XX^T fallback
  4. ILP bit allocation    — HAWQ-V3 PULP ILP over {16, 8, 4}
  5. Intra-layer OBS sens. — per-element Eq. 10 -> Manhattan channels Eq. 11
  6. Decoupled permutation — 3 clusters C16/C8/C4 (Eq. 12 extended)
  7. Permutation fusion    — fold perms into RMSNorm (zero overhead)
  8. AutoRound             — SignSGD learnable rounding (INT4 sub-matrix)
  9. SQC + mixed-bit GPTQ  — calibrated scales + OBS error compensation
 10. Weight repacking      — SHMQUltimateLinear (FP16 + INT8 + INT4 segments)
 11. Inference             — PyTorch reference here; CUDA kernel in benchmarks/gpu
"""
from __future__ import annotations

import os
import time
from typing import Dict, Optional

import torch

from .calibration import get_calibration_batches
from .config import SHMQUltimateConfig
from .ilp_allocation import average_bits, ilp_bit_allocation
from .inference.quant_linear import convert_linears
from .layer_quantizer import QuantizedLayerResult, quantize_layer
from .model_utils import collect_blocks, layer_key, load_model
from .permutation import (apply_permutation_to_weights, build_partitions,
                          magnitude_metric, permute_hessian)
from .rmsnorm_fusion import fuse_permutations
from .sensitivity import (capture_layer_stats, fisher_sensitivity,
                          intra_channel_sensitivity, xxt_sensitivity)
from .smoothquant import apply_smoothquant


class SHMQUltimatePipeline:
    def __init__(self, config: Optional[SHMQUltimateConfig] = None):
        self.cfg = config or SHMQUltimateConfig()
        self.model = None
        self.tokenizer = None
        self.blocks = None
        self.stats = None
        self.inter_sens: Dict[str, float] = {}
        self.bit_alloc: Dict[str, int] = {}
        self.partitions = {}
        self.results: Dict[str, QuantizedLayerResult] = {}
        self.report: Dict[str, object] = {}

    def log(self, msg: str):
        print(f"[shmq-ultimate] {msg}", flush=True)

    # ------------------------------------------------------------------ #
    def run(self, model=None, tokenizer=None):
        cfg = self.cfg
        t0 = time.time()

        if model is None:
            self.log(f"loading model {cfg.model_name} ...")
            model, tokenizer = load_model(cfg.model_name, cfg.dtype, cfg.device,
                                          cfg.max_blocks)
        self.model, self.tokenizer = model, tokenizer
        self.blocks = collect_blocks(model)
        self.log(f"{len(self.blocks)} transformer blocks, "
                 f"{sum(len(b.linears) for b in self.blocks)} linear layers")

        batches = get_calibration_batches(
            tokenizer, cfg.n_samples, cfg.sequence_length,
            cfg.calib_dataset, cfg.seed, cfg.device)

        # Step 1: SmoothQuant --------------------------------------------------
        if cfg.enable_smoothquant:
            n = apply_smoothquant(model, batches[: min(8, len(batches))],
                                  self.blocks, cfg.smoothquant_alpha)
            self.log(f"step 1  SmoothQuant: smoothed {n} norm->fc groups")

        # Step 2: calibration statistics --------------------------------------
        self.stats = capture_layer_stats(model, batches, self.blocks)
        self.log(f"step 2  captured XX^T stats for {len(self.stats)} layers")

        # Step 3: inter-layer sensitivity --------------------------------------
        if cfg.inter_hessian == "fisher":
            self.inter_sens = fisher_sensitivity(
                model, batches, self.blocks, bits_probe=4,
                group_size=cfg.group_size)
        else:
            self.inter_sens = xxt_sensitivity(self.stats, self.blocks,
                                              group_size=cfg.group_size)
        self.log(f"step 3  inter-layer sensitivity ({cfg.inter_hessian}) done")

        # Step 4: ILP bit allocation -------------------------------------------
        self.bit_alloc = ilp_bit_allocation(
            self.blocks, self.inter_sens, cfg.bit_levels,
            cfg.target_avg_bits, cfg.group_size,
            cfg.enable_parallel_constraint)
        avg = average_bits(self.bit_alloc, self.blocks)
        self.report["avg_bits"] = avg
        self.log(f"step 4  ILP allocation: avg {avg:.3f} bits "
                 f"(target {cfg.target_avg_bits})")

        # Step 5: intra-layer OBS channel sensitivity ---------------------------
        channel_sens: Dict[str, torch.Tensor] = {}
        magnitudes: Dict[str, torch.Tensor] = {}
        for blk in self.blocks:
            for role, lin in blk.linears.items():
                key = layer_key(blk.index, role)
                st = self.stats.get(key)
                if st is None:
                    continue
                H = st.hessian()
                channel_sens[key] = intra_channel_sensitivity(
                    lin.weight.data, H, 4, cfg.group_size, cfg.hessian_lambda)
                act = st.inputs[0] if st.inputs else None
                magnitudes[key] = magnitude_metric(lin.weight.data, act)
        self.log(f"step 5  OBS channel sensitivity for {len(channel_sens)} layers")

        # Step 6: decoupled 3-cluster permutation + PolyQ quanta ----------------
        self.partitions = build_partitions(
            self.blocks, channel_sens, magnitudes, self.bit_alloc,
            cfg.intra_hp_base_ratio, cfg.quanta_int8, cfg.quanta_int4,
            cfg.enable_permutation)
        apply_permutation_to_weights(self.blocks, self.partitions)
        n_perm = sum(1 for p in self.partitions.values() if not p.is_identity())
        self.log(f"step 6  permutation: {n_perm} layers permuted")

        # Step 7: permutation fusion into RMSNorm -------------------------------
        if cfg.enable_rmsnorm_fusion:
            n = fuse_permutations(self.blocks, self.partitions)
            self.log(f"step 7  fused {n} RMSNorms (zero-overhead permutation)")

        # Steps 8-9: per-layer quantization (SQC + AutoRound + mixed-bit GPTQ) --
        n_done = 0
        for blk in self.blocks:
            for role, lin in blk.linears.items():
                key = layer_key(blk.index, role)
                part = self.partitions[key]
                st = self.stats.get(key)
                H = st.hessian() if st is not None else torch.eye(lin.in_features)
                H = permute_hessian(H, part)
                acts = None
                if st is not None and st.inputs:
                    acts = [a[:, part.perm] for a in st.inputs]
                salience = torch.diagonal(H).clone()
                res = quantize_layer(
                    lin.weight.data, H, part,
                    group_size=cfg.group_size,
                    act_samples=acts, salience=salience,
                    enable_sqc=cfg.enable_sqc, sqc_grid=cfg.sqc_grid,
                    sqc_range=cfg.sqc_range,
                    enable_autoround=cfg.enable_autoround,
                    autoround_iters=cfg.autoround_iters,
                    autoround_lr=cfg.autoround_lr,
                    enable_gptq=cfg.enable_gptq,
                    gptq_blocksize=cfg.gptq_blocksize,
                    gptq_percdamp=cfg.gptq_percdamp)
                lin.weight.data = res.w_deq.to(lin.weight.dtype)
                self.results[key] = res
                n_done += 1
        self.log(f"steps 8-9  quantized {n_done} layers "
                 f"(SQC={cfg.enable_sqc}, AutoRound={cfg.enable_autoround}, "
                 f"GPTQ={cfg.enable_gptq})")

        self.report["elapsed_s"] = time.time() - t0
        self.log(f"pipeline done in {self.report['elapsed_s']:.1f}s")
        return model

    # ------------------------------------------------------------------ #
    def convert_for_inference(self) -> int:
        """Step 10: replace nn.Linear with SHMQUltimateLinear (packed)."""
        n = convert_linears(self.model, self.results, self.cfg.group_size)
        self.log(f"step 10  converted {n} layers to SHMQUltimateLinear")
        return n

    def save(self, output_dir: Optional[str] = None):
        out = output_dir or self.cfg.output_dir
        os.makedirs(out, exist_ok=True)
        self.cfg.save(os.path.join(out, "shmq_ultimate_config.json"))
        meta = {
            "bit_alloc": self.bit_alloc,
            "avg_bits": self.report.get("avg_bits"),
            "partitions": {
                k: {"n16": p.n16, "n8": p.n8, "n4": p.n4}
                for k, p in self.partitions.items()
            },
        }
        import json
        with open(os.path.join(out, "shmq_ultimate_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        self.log(f"saved config + meta to {out}")
