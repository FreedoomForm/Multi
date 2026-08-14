"""SignSGD optimizer for AutoRound (SHMQ Step 6).

SignSGD: θ ← θ - lr * sign(g)

Reference: AutoRound sign_round/sign_sgd.py (line 389):
    param.add_(torch.sign(d_p), alpha=-lr)
"""
from __future__ import annotations
from typing import List, Optional
import torch
from torch.optim.optimizer import Optimizer


class SignSGD(Optimizer):
    """SignSGD optimizer — updates parameters using only the sign of the gradient.

    Mathematical update rule:
        θ ← θ - lr * sign(∂L/∂θ)

    This is robust to gradient scale (only uses direction), making it well-suited
    for low-precision optimization like AutoRound's V update.

    Reference: AutoRound sign_round/sign_sgd.py.
    """

    def __init__(self, params, lr: float = 5e-3):
        if lr <= 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad
                # SignSGD update: p ← p - lr * sign(grad)
                p.add_(torch.sign(d_p), alpha=-lr)

        return loss


def linear_lr_schedule(step: int, total_steps: int, start_lr: float,
                       end_lr: float = 0.0) -> float:
    """Linear LR decay from start_lr to end_lr over total_steps.

    Matches AutoRound's LinearLR(start_factor=1.0, end_factor=0.0).
    """
    if total_steps <= 0:
        return end_lr
    frac = max(0.0, min(1.0, 1.0 - step / total_steps))
    return end_lr + frac * (start_lr - end_lr)
