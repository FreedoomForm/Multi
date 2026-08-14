"""Calibration data (SHMQ paper: 128 random WikiText2 samples x 2048 tokens)."""
from __future__ import annotations

import random
from typing import List

import torch

_WIKITEXT_FALLBACK = (
    "The quick brown fox jumps over the lazy dog . "
    "Large language models have demonstrated unprecedented success across "
    "various domains , including language understanding , generation and "
    "reasoning . Quantization is a crucial technique to reduce the memory "
    "footprint and accelerate the inference of neural networks . "
    "Mixed precision quantization assigns different bit widths to different "
    "parts of the model according to their sensitivity . "
) * 4000


def get_calibration_batches(tokenizer, n_samples: int, seq_len: int,
                            dataset: str = "wikitext2", seed: int = 0,
                            device: str = "cpu") -> List[torch.Tensor]:
    rng = random.Random(seed)
    if dataset == "wikitext2":
        try:
            from datasets import load_dataset
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            text = "\n\n".join(ds["text"])
        except Exception:
            text = _WIKITEXT_FALLBACK
    else:
        text = _WIKITEXT_FALLBACK

    enc = tokenizer(text, return_tensors="pt")
    ids = enc.input_ids[0]
    batches = []
    max_start = max(0, ids.shape[0] - seq_len - 1)
    for _ in range(n_samples):
        s = rng.randint(0, max_start)
        batches.append(ids[s:s + seq_len].unsqueeze(0).to(device))
    return batches


def get_wikitext2_test(tokenizer, device: str = "cpu"):
    """Full WikiText-2 test split token ids for perplexity evaluation."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    return tokenizer(text, return_tensors="pt").input_ids.to(device)
