"""Calibration data loading for SHMQ-Ultimate.

Loads WikiText-2 (default) and prepares batches of shape (n_samples, seq_len)
for sensitivity computation and AutoRound calibration.

Reference: SHMQ paper Section 4.1 — "128 samples from WikiText2, each with 2048 tokens."
"""
from __future__ import annotations
from typing import Optional, Tuple, Iterator, List
import torch
from torch.utils.data import DataLoader, TensorDataset


def get_wikitext2(tokenizer, nsamples: int = 128, seqlen: int = 2048,
                  device: str = "cpu") -> torch.Tensor:
    """Load WikiText-2 (validation set) and return nsamples of seqlen tokens.

    Returns a tensor of shape (nsamples, seqlen) — input_ids ready for the model.

    This mirrors SliM-LLM's `datautils.py:get_loaders("wikitext2")` exactly.
    """
    from datasets import load_dataset
    print(f"[calibration] Loading WikiText-2 (n_samples={nsamples}, seq_len={seqlen})")

    # Load validation split (test split is also fine — both work for calibration)
    try:
        traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        valdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    except Exception:
        # Fall back to non-raw variant
        traindata = load_dataset("wikitext", "wikitext-2-v1", split="train")
        valdata = load_dataset("wikitext", "wikitext-2-v1", split="test")

    # Tokenize all of validation, concatenate, then split into nsamples chunks
    text = "\n\n".join(valdata["text"])
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids[0]  # shape (total_tokens,)

    # Take nsamples chunks of seqlen each
    samples = []
    for i in range(nsamples):
        start = i * seqlen
        end = start + seqlen
        if end > input_ids.shape[0]:
            # Wrap around (or pad with first tokens)
            chunk = torch.cat([input_ids[start:], input_ids[: end - input_ids.shape[0]]])
        else:
            chunk = input_ids[start:end]
        samples.append(chunk)

    samples_tensor = torch.stack(samples).to(device)
    print(f"[calibration] Calibration data shape: {samples_tensor.shape} "
          f"(tokens: {samples_tensor.numel():,})")
    return samples_tensor


def get_calibration_data(name: str, tokenizer, nsamples: int = 128,
                         seqlen: int = 2048, device: str = "cpu") -> torch.Tensor:
    """Dispatch to the right calibration data loader.

    Currently supports: "wikitext2". Can be extended to "c4", "pile", "alpaca".
    """
    name = name.lower()
    if name in ("wikitext2", "wikitext-2", "wt2"):
        return get_wikitext2(tokenizer, nsamples, seqlen, device)
    elif name in ("c4",):
        return get_c4(tokenizer, nsamples, seqlen, device)
    elif name in ("pile",):
        return get_pile(tokenizer, nsamples, seqlen, device)
    else:
        raise ValueError(f"Unknown calibration dataset: {name}")


def get_c4(tokenizer, nsamples: int = 128, seqlen: int = 2048,
           device: str = "cpu") -> torch.Tensor:
    """Load C4 calibration data."""
    from datasets import load_dataset
    print(f"[calibration] Loading C4 (n_samples={nsamples}, seq_len={seqlen})")
    traindata = load_dataset("allenai/c4", "en", split="train", streaming=True)
    samples = []
    n_collected = 0
    for example in traindata:
        if n_collected >= nsamples:
            break
        enc = tokenizer(example["text"], return_tensors="pt",
                        max_length=seqlen, truncation=True, padding="max_length")
        if enc.input_ids.shape[1] >= seqlen:
            samples.append(enc.input_ids[0, :seqlen])
            n_collected += 1
    return torch.stack(samples).to(device)


def get_pile(tokenizer, nsamples: int = 128, seqlen: int = 2048,
             device: str = "cpu") -> torch.Tensor:
    """Load Pile calibration data."""
    from datasets import load_dataset
    print(f"[calibration] Loading Pile (n_samples={nsamples}, seq_len={seqlen})")
    traindata = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
    samples = []
    n_collected = 0
    for example in traindata:
        if n_collected >= nsamples:
            break
        enc = tokenizer(example["text"], return_tensors="pt",
                        max_length=seqlen, truncation=True, padding="max_length")
        if enc.input_ids.shape[1] >= seqlen:
            samples.append(enc.input_ids[0, :seqlen])
            n_collected += 1
    return torch.stack(samples).to(device)


def iter_batches(input_ids: torch.Tensor, batch_size: int = 1) -> Iterator[torch.Tensor]:
    """Iterate over calibration data in batches.

    Args:
        input_ids: (n_samples, seq_len) tensor
        batch_size: batch size for forward passes
    Yields:
        (batch_size, seq_len) tensor
    """
    n = input_ids.shape[0]
    for i in range(0, n, batch_size):
        yield input_ids[i : i + batch_size]
