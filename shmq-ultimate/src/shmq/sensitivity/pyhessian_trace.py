"""PyHessian trace for inter-layer sensitivity (HAWQ-V3 alternative to Fisher).

Implements Hutchinson-trace Hessian computation per layer, used as an alternative
to Fisher (SHMQ paper Appendix A.2 ablation). Default mode in SHMQ-Ultimate is
Fisher; this is provided for ablation comparison.

Reference: HAWQ-V3 ILP.ipynb uses precomputed `Hutchinson_trace` values per layer.
We compute these on-the-fly using the `pyhessian` package.
"""
from __future__ import annotations
from typing import Dict, List, Optional
import torch
import torch.nn as nn
from ..utils import get_module_by_name


def _isolate_layer_for_hessian(model: nn.Module, layer_name: str) -> nn.Module:
    """Return a 'view' of the model where only this layer's parameters require grad.

    PyHessian computes the trace of the Hessian of the loss w.r.t. all parameters
    that have requires_grad=True. To get per-layer Hessian, we set requires_grad=False
    on all params except this layer's weights.
    """
    target_mod = get_module_by_name(model, layer_name)
    # Save original requires_grad state
    saved_state = []
    for n, p in model.named_parameters():
        saved_state.append((n, p.requires_grad))
        p.requires_grad = (target_mod.weight is p)
    return target_mod, saved_state


def _restore_requires_grad(model: nn.Module, saved_state):
    for n, rg in saved_state:
        param = model
        for part in n.split("."):
            param = getattr(param, part)
        param.requires_grad = rg


def compute_inter_layer_pyhessian_trace(
    model: nn.Module,
    layer_names: List[str],
    calibration_data: torch.Tensor,
    batch_size: int = 1,
    max_samples: int = 8,
    hessian_batch_size: int = 1,
    max_iter: int = 100,
    tol: float = 1e-3,
    loss_type: str = "next_token",
) -> Dict[str, float]:
    """Compute per-layer Hessian trace using PyHessian (Hutchinson algorithm).

    PyHessian API:
        from pyhessian import hessian
        H = hessian(model, criterion, data=(inputs, labels), cuda=use_cuda)
        trace_mean, trace_norm = H.trace(maxIter=max_iter, tol=tol)

    We normalize the trace by # parameters (per HAWQ-V3 ILP.ipynb convention).

    Args:
        model: HuggingFace LLM
        layer_names: list of layer names
        calibration_data: (n_samples, seq_len) input_ids
        batch_size: forward batch size for sensitivity computation
        max_samples: cap on number of samples (Hessian trace is expensive,
                     use few samples — e.g. 4-8)
        hessian_batch_size: PyHessian batch size (typically 1)
        max_iter: PyHessian max iterations for Hutchinson
        tol: PyHessian convergence tolerance
        loss_type: "next_token" for cross-entropy (default)

    Returns:
        Dict {layer_name: normalized_hessian_trace (float)}
    """
    try:
        from pyhessian import hessian as pyhessian_hessian
    except ImportError:
        raise ImportError(
            "pyhessian not installed. Install with: pip install pyhessian"
        )

    device = next(model.parameters()).device
    n_total = min(calibration_data.shape[0], max_samples)
    print(f"[pyhessian] Computing trace for {len(layer_names)} layers, "
          f"{n_total} samples each...")

    # Take a few calibration samples
    sample_ids = calibration_data[:n_total].to(device)

    traces: Dict[str, float] = {}
    for name in layer_names:
        print(f"[pyhessian] Processing {name}...")
        target_mod, saved_state = _isolate_layer_for_hessian(model, name)

        # Define criterion: cross-entropy on next-token prediction
        def criterion():
            outputs = model(sample_ids)
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = sample_ids[..., 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)).float(),
                shift_labels.view(-1),
                reduction="mean",
            )
            return loss

        # PyHessian expects: data = (inputs, targets) and criterion(*data)
        # For LM, we wrap criterion to ignore args and use the captured sample_ids.
        try:
            H = pyhessian_hessian(
                model, criterion,
                data=(sample_ids, None), cuda=(device.type == "cuda"),
            )
            trace_mean, _ = H.trace(maxIter=max_iter, tol=tol)
            # Normalize by number of parameters
            n_params = target_mod.weight.numel()
            traces[name] = float(trace_mean) / max(n_params, 1)
        except Exception as e:
            print(f"[pyhessian] WARNING: failed for {name}: {e}")
            traces[name] = 0.0
        finally:
            _restore_requires_grad(model, saved_state)

    return traces
