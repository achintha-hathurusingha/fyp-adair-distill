"""Reconstruction losses — the ONLY losses permitted in Phase 01.

No distillation losses here (response/feature/attention/relation/frequency).
Those arrive in Phase 02.
"""
from __future__ import annotations

import torch
from torch import nn


class CharbonnierLoss(nn.Module):
    """Charbonnier (smooth-L1-like) loss: ``mean(sqrt((x - y)^2 + eps^2))``."""

    def __init__(self, eps: float = 1e-3) -> None:
        super().__init__()
        self.eps2 = eps * eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(f"shape mismatch: pred {tuple(pred.shape)} "
                             f"vs target {tuple(target.shape)}")
        return torch.sqrt((pred - target) ** 2 + self.eps2).mean()


class L1Loss(nn.Module):
    """Plain mean-absolute-error loss."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(f"shape mismatch: pred {tuple(pred.shape)} "
                             f"vs target {tuple(target.shape)}")
        return torch.abs(pred - target).mean()


def build_loss(cfg: dict) -> nn.Module:
    """Build a reconstruction loss from a loss-config dict."""
    name = cfg.get("name", "charbonnier").lower()
    if name == "charbonnier":
        return CharbonnierLoss(eps=cfg.get("eps", 1e-3))
    if name == "l1":
        return L1Loss()
    raise ValueError(f"Unknown/unsupported loss '{name}'. Phase 01 allows "
                     "only 'charbonnier' or 'l1'.")
