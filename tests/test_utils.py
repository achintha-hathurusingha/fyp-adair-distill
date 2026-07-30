"""Tests for seeding determinism, strict config access, and losses."""
from __future__ import annotations

import pytest
import torch

from src.losses.reconstruction import CharbonnierLoss, L1Loss, build_loss
from src.utils.config import load_yaml, require
from src.utils.seeding import seed_everything


def test_seeding_is_reproducible() -> None:
    seed_everything(123)
    a = torch.randn(64)
    seed_everything(123)
    b = torch.randn(64)
    assert torch.equal(a, b)


def test_require_raises_on_missing_key() -> None:
    with pytest.raises(KeyError):
        require({"a": 1}, "b", context="unit test")


def test_load_yaml_raises_on_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_yaml("configs/does_not_exist.yaml")


def test_charbonnier_zero_at_identity() -> None:
    """Charbonnier of identical inputs equals eps (not exactly zero)."""
    loss = CharbonnierLoss(eps=1e-3)
    x = torch.rand(2, 3, 8, 8)
    assert loss(x, x).item() == pytest.approx(1e-3, abs=1e-6)


def test_l1_matches_manual() -> None:
    pred = torch.tensor([1.0, 2.0])
    target = torch.tensor([0.0, 0.0])
    assert L1Loss()(pred, target).item() == pytest.approx(1.5)


def test_losses_reject_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        L1Loss()(torch.rand(1, 3, 4, 4), torch.rand(1, 3, 5, 5))


def test_build_loss_rejects_distillation_losses() -> None:
    """Phase 01 permits reconstruction losses only."""
    with pytest.raises(ValueError):
        build_loss({"name": "feature_distill"})
