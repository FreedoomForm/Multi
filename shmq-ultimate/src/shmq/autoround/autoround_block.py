"""Per-block AutoRound SignSGD optimization (SHMQ Step 6).

For each transformer block:
1. Capture FP16 inputs to the block (via one-time forward pass on calibration data).
2. Wrap all Linear layers in the block with WrapperLinear.
3. For 200 steps:
     - Forward: block(inputs) → outputs
     - Compute MSE loss vs FP16 outputs (captured before quantization)
     - Backward: compute gradients w.r.t. V
     - SignSGD update: V ← V - lr * sign(grad)
4. Bake V into weights and unwrap.
5. Move to next block.

Reference: AutoRound calibration/llm.py (LLMCalibrator), sign_round/quantizer.py.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .wrapper import WrapperLinear, wrap_model_linears, unwrap_model_linears
from .sign_sgd import SignSGD, linear_lr_schedule


def capture_block_inputs(model: nn.Module, block_idx: int,
                          calibration_data: torch.Tensor,
                          batch_size: int = 1,
                          max_samples: int = 8) -> List[Tuple[torch.Tensor, ...]]:
    """Capture the input arguments to transformer block `block_idx`.

    Returns a list of tuples (each tuple = one batch's inputs to the block).
    """
    device = next(model.parameters()).device
    blocks = model.model.layers if hasattr(model, "model") else model.transformer.h

    captured: List[Tuple[torch.Tensor, ...]] = []
    n_total = min(calibration_data.shape[0], max_samples)

    # We capture by replacing the block's forward with a capture function
    original_forwards = []

    def make_capture_forward(block, captures_list):
        def capture_forward(*args, **kwargs):
            captures_list.append(tuple(a.detach().clone() if isinstance(a, torch.Tensor) else a
                                       for a in args))
            # Don't actually run the block; raise to short-circuit
            raise StopIteration
        return capture_forward

    block = blocks[block_idx]
    original_forward = block.forward
    block.forward = make_capture_forward(block, captured)

    try:
        with torch.no_grad():
            for i in range(0, n_total, batch_size):
                batch = calibration_data[i : i + batch_size].to(device)
                try:
                    model(batch)
                except StopIteration:
                    pass
    finally:
        block.forward = original_forward

    return captured


def autoround_block(model: nn.Module, block_idx: int,
                    layer_names_in_block: List[str],
                    calibration_data: torch.Tensor,
                    n_bits: int = 4,
                    group_size: int = 128,
                    iters: int = 200,
                    lr: Optional[float] = None,
                    batch_size: int = 1,
                    max_samples: int = 8,
                    n_bits_per_layer: Optional[Dict[str, int]] = None,
                    verbose: bool = False) -> Dict[str, WrapperLinear]:
    """Apply AutoRound SignSGD to all Linear layers in a single transformer block.

    Args:
        model: HuggingFace LLM
        block_idx: index of the transformer block to optimize
        layer_names_in_block: list of layer names in this block to wrap/optimize
        calibration_data: (n_samples, seq_len) input_ids
        n_bits: default bit-width (4) — overridden by n_bits_per_layer if provided
        group_size: 128
        iters: 200 (AutoRound default)
        lr: if None, use 1.0/iters (= 5e-3 for iters=200)
        batch_size: forward batch size
        max_samples: cap on calibration samples for this block
        n_bits_per_layer: {layer_name: 4 or 8} — if a layer is 8-bit, skip AutoRound
                          (8-bit doesn't benefit much from learnable rounding)
        verbose: print progress

    Returns:
        Dict {layer_name: WrapperLinear} — wrappers after optimization (V is baked)
    """
    if lr is None:
        lr = 1.0 / iters

    if n_bits_per_layer is None:
        n_bits_per_layer = {n: n_bits for n in layer_names_in_block}

    # Only apply AutoRound to 4-bit layers (8-bit doesn't need it as much)
    layers_to_wrap = [n for n in layer_names_in_block if n_bits_per_layer.get(n, n_bits) == 4]
    if not layers_to_wrap:
        if verbose:
            print(f"[autoround] Block {block_idx}: no 4-bit layers, skipping")
        return {}

    print(f"[autoround] Block {block_idx}: wrapping {len(layers_to_wrap)} layers, "
          f"{iters} iters, lr={lr:.4f}")

    # Capture FP16 block inputs (and FP16 reference outputs)
    device = next(model.parameters()).device
    blocks = model.model.layers if hasattr(model, "model") else model.transformer.h
    block = blocks[block_idx]

    # 1. Capture FP16 inputs to this block
    block_inputs = capture_block_inputs(model, block_idx, calibration_data,
                                         batch_size=batch_size, max_samples=max_samples)
    if not block_inputs:
        print(f"[autoround] Block {block_idx}: no inputs captured, skipping")
        return {}

    # 2. Capture FP16 reference outputs
    fp16_outputs = []
    original_forward = block.forward
    try:
        with torch.no_grad():
            for inp in block_inputs:
                out = original_forward(*inp)
                if isinstance(out, tuple):
                    out = out[0]
                fp16_outputs.append(out.detach())
    except Exception as e:
        print(f"[autoround] Block {block_idx}: failed to capture FP16 outputs: {e}")
        return {}

    # 3. Wrap Linear layers
    wrappers = wrap_model_linears(model, layers_to_wrap, n_bits=n_bits,
                                   group_size=group_size, symmetric=True)
    if not wrappers:
        return {}

    # 4. Optimize V via SignSGD
    params = [w.value for w in wrappers.values() if w.value.requires_grad]
    optimizer = SignSGD(params, lr=lr)

    block.train()  # enable grad
    for step in range(iters):
        # Update LR (linear decay)
        cur_lr = linear_lr_schedule(step, iters, start_lr=lr, end_lr=0.0)
        for g in optimizer.param_groups:
            g["lr"] = cur_lr

        optimizer.zero_grad()
        total_loss = 0.0
        for inp, ref_out in zip(block_inputs, fp16_outputs):
            # Forward through the block (with wrapped layers)
            out = block(*inp)
            if isinstance(out, tuple):
                out = out[0]
            # MSE loss vs FP16 reference
            loss = F.mse_loss(out.float(), ref_out.float())
            loss = loss * 1000  # scale up for numerical stability (AutoRound trick)
            loss.backward()
            total_loss += loss.item()
        optimizer.step()
        if verbose and (step + 1) % 50 == 0:
            print(f"[autoround] Block {block_idx} step {step+1}/{iters}: "
                  f"loss={total_loss/len(block_inputs):.4f}, lr={cur_lr:.5f}")

    # 5. Bake V into weights and unwrap
    unwrap_model_linears(model, wrappers)
    block.eval()
    return wrappers
