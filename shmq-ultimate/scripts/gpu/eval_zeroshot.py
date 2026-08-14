#!/usr/bin/env python3
"""Zero-shot accuracy evaluation on HellaSwag, ARC-Easy, ARC-Challenge, PIQA.

Matches the SHMQ paper evaluation protocol (§4.1, Table 3).

Usage:
    # SHMQ-quantized model
    python scripts/gpu/eval_zeroshot.py --model ./download/qwen7b_shmq_ultimate

    # FP16 baseline
    python scripts/gpu/eval_zeroshot.py --model Qwen/Qwen2.5-7B-Instruct --fp16

    # Specific tasks only
    python scripts/gpu/eval_zeroshot.py --model ./download/qwen7b_shmq_ultimate --tasks hellaswag piqa

Expected results (SHMQ paper Table 3, Qwen2.5-7B-Instruct):
    HellaSwag: 76.21  (FP16: 76.42)
    PIQA:      78.89  (FP16: 79.05)
    ARC-E:     72.90  (FP16: 72.94)
    ARC-C:     58.11  (FP16: 58.45)
    WinoGrande:75.82  (FP16: 75.93)
    Avg:       75.58  (FP16: 75.71, gap: 0.13%)
"""
from __future__ import annotations
import sys
import os
import argparse
import time
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


def parse_args():
    p = argparse.ArgumentParser(description="Zero-shot accuracy evaluation")
    p.add_argument("--model", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--tasks", nargs="+",
                   default=["hellaswag", "piqa", "arc_easy", "arc_challenge", "winogrande"],
                   help="Tasks to evaluate")
    p.add_argument("--num-fewshot", type=int, default=0,
                   help="Number of few-shot examples (0 for zero-shot)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--limit", type=int, default=None,
                   help="Limit number of examples (for quick testing)")
    return p.parse_args()


def main():
    args = parse_args()
    print("=" * 70)
    print("Zero-Shot Accuracy Evaluation")
    print("=" * 70)
    print(f"Model:    {args.model}")
    print(f"Tasks:    {args.tasks}")
    print(f"Fewshot:  {args.num_fewshot}")
    print(f"Device:   {args.device}")

    # Use lm-eval-harness if available
    try:
        from lm_eval import Evaluator
        from lm_eval.models.huggingface import HFLM
        from lm_eval.tasks import TaskManager
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("\n[ERROR] lm-eval not installed. Install with:")
        print("  pip install lm-eval")
        print("\nOr use the simple built-in evaluator:")
        print("  python scripts/gpu/eval_zeroshot_simple.py --model ...")
        return 1

    print("\nLoading model...")
    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map=args.device,
        trust_remote_code=True,
    )
    print(f"  Parameters: {sum(p.numel() for p in model.parameters())/1e9:.2f}B")

    # Build HFLM wrapper
    print("\nBuilding lm-eval wrapper...")
    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=2048,
        trust_remote_code=True,
    )

    # Evaluate each task
    results = {}
    task_manager = TaskManager()
    for task_name in args.tasks:
        print(f"\n{'=' * 70}")
        print(f"Evaluating: {task_name}")
        print(f"{'=' * 70}")
        t0 = time.time()

        try:
            from lm_eval import simple_evaluate
            result = simple_evaluate(
                model=lm,
                tasks=[task_name],
                num_fewshot=args.num_fewshot,
                limit=args.limit,
            )
            metrics = result["results"][task_name]
            # Extract accuracy
            acc = metrics.get("acc,none") or metrics.get("acc_norm,none") or 0.0
            acc_norm = metrics.get("acc_norm,none", acc)
            results[task_name] = {
                "acc": acc,
                "acc_norm": acc_norm,
                "time": time.time() - t0,
            }
            print(f"  acc={acc:.4f}  acc_norm={acc_norm:.4f}  ({time.time()-t0:.0f}s)")
        except Exception as e:
            print(f"  [ERROR] {e}")
            results[task_name] = {"error": str(e)}

    # Report
    print(f"\n{'=' * 70}")
    print("ZERO-SHOT RESULTS")
    print(f"{'=' * 70}")
    print(f"{'Task':<20} {'acc':>8} {'acc_norm':>10} {'time':>8}")
    print("-" * 50)
    accs = []
    for task, r in results.items():
        if "error" in r:
            print(f"{task:<20} {'ERROR':>8}")
            continue
        print(f"{task:<20} {r['acc']:>8.4f} {r['acc_norm']:>10.4f} {r['time']:>7.0f}s")
        accs.append(r["acc"])
    if accs:
        avg = sum(accs) / len(accs)
        print("-" * 50)
        print(f"{'Average':<20} {avg:>8.4f}")
        print(f"\n  SHMQ paper target: 0.7558 avg (gap 0.13% vs FP16 0.7571)")
        print(f"  Your result:       {avg:.4f} avg")

    # Save results
    out_file = os.path.join(os.path.dirname(args.model), "zeroshot_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_file}")


if __name__ == "__main__":
    main()
