#!/usr/bin/env python3
"""Perplexity evaluation on WikiText-2 (raw).

Computes word-level perplexity of a model on the WikiText-2 test set,
matching the evaluation protocol in the SHMQ paper (§4.1).

Usage:
    # SHMQ-quantized model
    python scripts/gpu/eval_perplexity.py --model ./download/qwen7b_shmq_ultimate

    # FP16 baseline for comparison
    python scripts/gpu/eval_perplexity.py --model Qwen/Qwen2.5-7B-Instruct --fp16

    # Both and compare
    python scripts/gpu/eval_perplexity.py --model ./download/qwen7b_shmq_ultimate --baseline Qwen/Qwen2.5-7B-Instruct

Expected results (SHMQ paper Table 3, Qwen2.5-7B-Instruct):
    FP16:     7.61
    SHMQ:     7.58  (gap: -0.03, SHMQ is actually BETTER due to ILP+AutoRound)
"""
from __future__ import annotations
import sys
import os
import math
import argparse
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


def parse_args():
    p = argparse.ArgumentParser(description="WikiText-2 perplexity evaluation")
    p.add_argument("--model", required=True, help="Model name or path")
    p.add_argument("--baseline", default=None,
                   help="Optional baseline model for comparison")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--seqlen", type=int, default=2048,
                   help="Sequence length for evaluation (paper: 2048)")
    p.add_argument("--dataset", default="wikitext-2-raw-v1",
                   help="Dataset name (default: wikitext-2-raw-v1)")
    return p.parse_args()


@torch.no_grad()
def compute_perplexity(model, tokenizer, dataset_name, seqlen, device):
    """Compute word-level perplexity on WikiText-2 test set."""
    from datasets import load_dataset
    import torch

    print(f"Loading dataset: {dataset_name}")
    test = load_dataset(dataset_name, split="test")
    print(f"  Test examples: {len(test)}")

    # Concatenate all text
    text = "\n\n".join(test["text"])
    print(f"  Total chars: {len(text):,}")

    # Tokenize
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids.to(device)
    print(f"  Total tokens: {input_ids.numel():,}")

    # Split into chunks of seqlen
    n_tokens = input_ids.numel()
    n_chunks = n_tokens // seqlen
    input_ids = input_ids[:, :n_chunks * seqlen].view(n_chunks, seqlen)
    print(f"  Chunks: {n_chunks} × {seqlen} tokens")

    # Compute loss chunk by chunk
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    t0 = time.time()

    for i in range(n_chunks):
        batch = input_ids[i:i+1]  # (1, seqlen)
        out = model(batch)
        # Shift for next-token prediction
        logits = out.logits[:, :-1, :].contiguous()  # (1, seqlen-1, vocab)
        targets = batch[:, 1:].contiguous()           # (1, seqlen-1)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            reduction="sum",
        )
        total_loss += loss.item()
        total_tokens += targets.numel()

        if (i + 1) % 50 == 0:
            avg_loss = total_loss / total_tokens
            ppl = math.exp(avg_loss)
            elapsed = time.time() - t0
            print(f"  [{i+1}/{n_chunks}] loss={avg_loss:.4f} ppl={ppl:.2f} "
                  f"({elapsed:.0f}s, {total_tokens/elapsed:.0f} tok/s)")

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    elapsed = time.time() - t0
    print(f"\nFinal: loss={avg_loss:.4f}  ppl={ppl:.4f}  ({elapsed:.0f}s)")
    return ppl, avg_loss


def main():
    args = parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    def load_model(name):
        print(f"\n{'=' * 70}")
        print(f"Loading model: {name}")
        print(f"{'=' * 70}")
        dtype = getattr(torch, args.dtype)
        tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=dtype, device_map=args.device,
            trust_remote_code=True,
        )
        print(f"  Parameters: {sum(p.numel() for p in model.parameters())/1e9:.2f}B")
        return model, tokenizer

    # Load SHMQ model
    model, tokenizer = load_model(args.model)
    ppl_shmq, loss_shmq = compute_perplexity(
        model, tokenizer, args.dataset, args.seqlen, args.device,
    )

    # Optional baseline
    ppl_base = None
    if args.baseline:
        del model
        torch.cuda.empty_cache()
        model, tokenizer = load_model(args.baseline)
        ppl_base, loss_base = compute_perplexity(
            model, tokenizer, args.dataset, args.seqlen, args.device,
        )

    # Report
    print(f"\n{'=' * 70}")
    print("PERPLEXITY RESULTS (WikiText-2)")
    print(f"{'=' * 70}")
    print(f"  SHMQ ({args.model}): {ppl_shmq:.4f}  (loss={loss_shmq:.4f})")
    if ppl_base is not None:
        print(f"  Base ({args.baseline}): {ppl_base:.4f}  (loss={loss_base:.4f})")
        gap = (ppl_shmq - ppl_base) / ppl_base * 100
        print(f"  Gap: {gap:+.2f}%")
        print(f"\n  SHMQ paper target: 7.58 PPL (gap -0.03 vs FP16 7.61)")
        print(f"  Your result:       {ppl_shmq:.2f} PPL (gap {gap:+.2f}%)")


if __name__ == "__main__":
    import torch  # needed by compute_perplexity
    main()
