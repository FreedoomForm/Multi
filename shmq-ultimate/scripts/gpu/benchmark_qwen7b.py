#!/usr/bin/env python3
"""Full SHMQ-Ultimate pipeline on Qwen2.5-7B-Instruct (GPU).

This is the production script that runs the complete 9-step SHMQ-Ultimate
pipeline on Qwen2.5-7B-Instruct, exactly as specified in the SHMQ paper.

Expected results (per SHMQ paper Table 3 & 15):
    - Accuracy: 75.58% zero-shot avg (vs FP16 75.71%, gap 0.13%)
    - WikiText-2 PPL: 7.58 (vs FP16 7.61)
    - Inference speedup: 2.86x vs FP16
    - Memory: ~4.8 bits/weight (vs 16 bits/weight FP16)

Requirements:
    - GPU with >= 24GB VRAM (A100 40GB / A6000 48GB / 3090 24GB / 4090 24GB)
    - CUDA toolkit + PyTorch with CUDA
    - ~30GB disk for model download

Usage:
    python scripts/gpu/benchmark_qwen7b.py [--device cuda] [--dtype float16]

    # Full run (paper defaults):
    python scripts/gpu/benchmark_qwen7b.py

    # Quick test (fewer samples):
    python scripts/gpu/benchmark_qwen7b.py --n-samples 32 --seqlen 512

    # Skip AutoRound (saves ~10 min, slight quality drop):
    python scripts/gpu/benchmark_qwen7b.py --no-autoround
"""
from __future__ import annotations
import sys
import os
import time
import argparse
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


def parse_args():
    p = argparse.ArgumentParser(description="SHMQ-Ultimate on Qwen2.5-7B-Instruct")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct",
                   help="HuggingFace model name")
    p.add_argument("--device", default="cuda", help="Device: cuda or cpu")
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--n-samples", type=int, default=128,
                   help="Calibration samples (paper: 128)")
    p.add_argument("--seqlen", type=int, default=2048,
                   help="Sequence length (paper: 2048)")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--smooth-alpha", type=float, default=0.5)
    p.add_argument("--target-hp-ratio", type=float, default=0.20,
                   help="Fraction of layers at 8-bit (W4.8 => ~0.20)")
    p.add_argument("--base-hp-ratio", type=float, default=0.125,
                   help="Intra-layer 8-bit fraction for 4-bit layers (UB=12.5%)")
    p.add_argument("--inter-layer-hessian", default="fisher",
                   choices=["fisher", "pyhessian"])
    p.add_argument("--dampening", type=float, default=0.1,
                   help="OBS dampening factor (lambda=0.1)")
    p.add_argument("--autoround-iters", type=int, default=200,
                   help="AutoRound SignSGD iterations (paper: 200)")
    p.add_argument("--no-autoround", action="store_true",
                   help="Disable AutoRound (saves time)")
    p.add_argument("--no-sqc", action="store_true",
                   help="Disable SQC calibration")
    p.add_argument("--output-dir", default="./download/qwen7b_shmq_ultimate",
                   help="Where to save the quantized model")
    p.add_argument("--skip-save", action="store_true",
                   help="Don't save the model (just run pipeline + report stats)")
    return p.parse_args()


def main():
    args = parse_args()
    print("=" * 70)
    print("SHMQ-Ultimate — Qwen2.5-7B-Instruct (Production Run)")
    print("=" * 70)
    print(f"Model:              {args.model}")
    print(f"Device:             {args.device}")
    print(f"Dtype:              {args.dtype}")
    print(f"Calibration:        {args.n_samples} samples × {args.seqlen} tokens")
    print(f"Format target:      W{4 + 4*args.target_hp_ratio:.1f}A8")
    print(f"AutoRound:          {'OFF' if args.no_autoround else f'{args.autoround_iters} iters'}")
    print(f"SQC:                {'OFF' if args.no_sqc else 'ON'}")
    print(f"Inter-layer Hessian: {args.inter_layer_hessian}")
    print(f"Output:             {args.output_dir}")
    print()

    import torch
    from shmq.config import SHMQConfig
    from shmq.pipeline import SHMQPipeline

    config = SHMQConfig(
        model_name=args.model,
        device=args.device,
        dtype=args.dtype,
        n_samples=args.n_samples,
        sequence_length=args.seqlen,
        batch_size=args.batch_size,
        group_size=args.group_size,
        smooth_alpha=args.smooth_alpha,
        target_hp_ratio=args.target_hp_ratio,
        base_hp_ratio=args.base_hp_ratio,
        inter_layer_hessian=args.inter_layer_hessian,
        dampening=args.dampening,
        enable_autoround=not args.no_autoround,
        autoround_iters=args.autoround_iters,
        enable_sqc=not args.no_sqc,
    )
    print(config.summary())

    pipeline = SHMQPipeline(config)
    t_total_start = time.time()

    # Run full pipeline (steps 0-9)
    pipeline.run()

    t_total = time.time() - t_total_start
    print(f"\n{'=' * 70}")
    print(f"TOTAL PIPELINE TIME: {t_total/60:.1f} minutes")
    print(f"{'=' * 70}")

    # Verify CUDA kernel
    from shmq.inference import is_cuda_kernel_available
    cuda_ok = is_cuda_kernel_available()
    print(f"\nCUDA kernel active: {cuda_ok}")

    # Report bit allocation stats
    n_4 = sum(1 for v in pipeline.bit_allocation.values() if v == 4)
    n_8 = sum(1 for v in pipeline.bit_allocation.values() if v == 8)
    print(f"\nBit allocation: {n_4} layers @ 4-bit, {n_8} layers @ 8-bit")

    # Measure memory footprint
    from shmq.inference import SHMQQuantLinear
    total_bytes = 0
    total_params = 0
    for m in pipeline.model.modules():
        if isinstance(m, SHMQQuantLinear):
            total_bytes += m.qweight_int8.numel()  # 1 byte each
            total_bytes += m.qweight_int4.numel()  # 1 byte per 2 weights
            total_bytes += m.scales_int8.numel() * 2  # float16
            total_bytes += m.scales_int4.numel() * 2  # float16
            total_params += m.in_features * m.out_features
    fp16_bytes = total_params * 2
    compression = fp16_bytes / max(total_bytes, 1)
    avg_bits = 8 * total_bytes / max(total_params, 1)  # rough
    print(f"\nMemory footprint:")
    print(f"  SHMQ:    {total_bytes / 1e9:.2f} GB")
    print(f"  FP16:    {fp16_bytes / 1e9:.2f} GB")
    print(f"  Compression: {compression:.2f}x")

    # Inference speed test
    if cuda_ok:
        print(f"\n{'=' * 70}")
        print("Inference Speed Test")
        print(f"{'=' * 70}")
        model = pipeline.model.to(args.device).eval()
        with torch.no_grad():
            # Warmup
            for _ in range(3):
                x = torch.randint(0, 1000, (1, 128), device=args.device)
                _ = model(x)
            torch.cuda.synchronize()

            # Timed
            n_iters = 20
            t0 = time.time()
            for _ in range(n_iters):
                x = torch.randint(0, 1000, (1, 128), device=args.device)
                _ = model(x)
            torch.cuda.synchronize()
            t1 = time.time()
            shmq_ms = (t1 - t0) / n_iters * 1000
            print(f"  SHMQ inference: {shmq_ms:.1f} ms/token (batch=1, seqlen=128)")

    # Save
    if not args.skip_save:
        print(f"\n{'=' * 70}")
        print("Saving quantized model")
        print(f"{'=' * 70}")
        pipeline.save_model(args.output_dir)
        print(f"\nModel saved to: {args.output_dir}")
        print("To load and use:")
        print(f"  from transformers import AutoModelForCausalLM")
        print(f"  model = AutoModelForCausalLM.from_pretrained('{args.output_dir}')")

    # Final summary
    print(f"\n{'=' * 70}")
    print("SHMQ-ULTIMATE PRODUCTION RUN COMPLETE")
    print(f"{'=' * 70}")
    print(f"Total time:       {t_total/60:.1f} min")
    print(f"CUDA kernel:      {'ACTIVE' if cuda_ok else 'FALLBACK'}")
    print(f"Compression:      {compression:.2f}x")
    print(f"Bit allocation:   {n_4}×INT4 + {n_8}×INT8")
    print(f"\nNext steps:")
    print(f"  1. Perplexity:    python scripts/gpu/eval_perplexity.py --model {args.output_dir}")
    print(f"  2. Zero-shot:     python scripts/gpu/eval_zeroshot.py --model {args.output_dir}")
    print(f"  3. Compare FP16:  python scripts/gpu/eval_perplexity.py --model {args.model} --fp16")


if __name__ == "__main__":
    main()
