#!/usr/bin/env python3
"""SHMQ-Ultimate v2 — GPU Run Script for Qwen2.5-7B-Instruct.

Runs the full 11-step pipeline on a real GPU (A100/H100) and produces:
  - Quantized model saved for vLLM
  - WikiText-2 perplexity
  - (Optional) Zero-shot accuracy on HellaSwag/ARC/PIQA/WinoGrande

Expected results (per SHMQ paper Table 2):
  - WikiText-2 PPL: ~7.58 (vs FP16 ~7.55, gap ≤ 0.13%)
  - Zero-shot avg: ~75.58% (vs FP16 75.71%, gap ≤ 0.13%)
  - Inference speedup: 2.86× (per SHMQ Table 3, layer-wise 1.83× to 4.21×)

Usage:
  python scripts/gpu/run_pipeline.py --config configs/qwen7b_paper.json

  # Or override individual params:
  python scripts/gpu/run_pipeline.py \\
      --model Qwen/Qwen2.5-7B-Instruct \\
      --bit-percent 8:10,4:90 \\
      --hp-ratio 0.10 \\
      --autoround-iters 200 \\
      --device cuda \\
      --save-dir ./quantized_models/qwen7b_shmq_v2
"""
import argparse
import json
import logging
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from shmq_v2.config import SHMQv2Config
from shmq_v2.pipeline import SHMQv2Pipeline


def parse_args():
    p = argparse.ArgumentParser(description="SHMQ-Ultimate v2 GPU pipeline")
    p.add_argument("--config", type=str, default=None,
                    help="Path to JSON config file (overrides defaults)")
    p.add_argument("--model", type=str, default=None,
                    help="HuggingFace model name (e.g., Qwen/Qwen2.5-7B-Instruct)")
    p.add_argument("--bit-percent", type=str, default=None,
                    help="Bit allocation, e.g., '8:10,4:90' (10% INT8, 90% INT4)")
    p.add_argument("--hp-ratio", type=float, default=None,
                    help="SHMQ high-precision K-channel ratio (default 0.10)")
    p.add_argument("--autoround-iters", type=int, default=None,
                    help="AutoRound SignSGD iterations (default 200)")
    p.add_argument("--n-samples", type=int, default=None,
                    help="Calibration samples (default 128)")
    p.add_argument("--seqlen", type=int, default=None,
                    help="Calibration sequence length (default 2048)")
    p.add_argument("--device", type=str, default=None,
                    help="Device: 'cuda' or 'cpu'")
    p.add_argument("--dtype", type=str, default=None, choices=["float16", "float32"])
    p.add_argument("--save-dir", type=str, default=None,
                    help="Directory to save quantized model (for vLLM)")
    p.add_argument("--no-permutation", action="store_true",
                    help="Disable SHMQ K-axis permutation")
    p.add_argument("--no-rmsnorm-fusion", action="store_true",
                    help="Disable RMSNorm fusion")
    p.add_argument("--no-smoothquant", action="store_true",
                    help="Disable SmoothQuant pre-processing")
    p.add_argument("--no-autoround", action="store_true",
                    help="Disable AutoRound")
    p.add_argument("--eval-ppl", action="store_true",
                    help="Evaluate WikiText-2 PPL after quantization")
    p.add_argument("--eval-zeroshot", action="store_true",
                    help="Evaluate zero-shot tasks (HellaSwag/ARC/PIQA/WinoGrande)")
    p.add_argument("--skip-steps", type=str, default="",
                    help="Comma-separated step indices to skip (e.g., '1,9')")
    return p.parse_args()


def build_config(args) -> SHMQv2Config:
    """Build SHMQv2Config from args."""
    cfg = SHMQv2Config()

    if args.config:
        with open(args.config) as f:
            data = json.load(f)
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    if args.model:
        cfg.model_name = args.model
    if args.bit_percent:
        parts = args.bit_percent.split(",")
        cfg.bit_percent = {int(k): int(v) for k, v in (p.split(":") for p in parts)}
    if args.hp_ratio is not None:
        cfg.hp_ratio = args.hp_ratio
    if args.autoround_iters is not None:
        cfg.autoround_iters = args.autoround_iters
    if args.n_samples is not None:
        cfg.n_samples = args.n_samples
    if args.seqlen is not None:
        cfg.sequence_length = args.seqlen
    if args.device:
        cfg.device = args.device
    if args.dtype:
        cfg.dtype = args.dtype
    if args.save_dir:
        cfg.save_dir = args.save_dir

    if args.no_permutation:
        cfg.enable_permutation = False
    if args.no_rmsnorm_fusion:
        cfg.enable_rmsnorm_fusion = False
    if args.no_smoothquant:
        cfg.enable_smoothquant = False
    if args.no_autoround:
        cfg.enable_autoround = False

    cfg.eval_ppl = args.eval_ppl
    cfg.eval_zeroshot = args.eval_zeroshot

    if args.skip_steps:
        cfg.skip_steps = set(int(s) for s in args.skip_steps.split(","))

    return cfg


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    cfg = build_config(args)
    cfg.validate()
    print(cfg.summary())

    pipeline = SHMQv2Pipeline(cfg)
    model = pipeline.run()

    print("\nPipeline complete. Model is ready for vLLM inference.")
    if cfg.save_dir:
        print(f"Quantized model saved to: {cfg.save_dir}")
        print(f"\nTo run inference with vLLM:")
        print(f"  # Apply MixLLM vLLM patches first:")
        print(f"  cd external/MixLLM && ./apply_vllm_patche.sh && cd -")
        print(f"  # Then run vLLM:")
        print(f"  python -m vllm.entrypoints.openai.api_server --model {cfg.save_dir}")


if __name__ == "__main__":
    main()
