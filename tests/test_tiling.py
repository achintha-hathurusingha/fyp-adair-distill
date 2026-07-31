"""Tests for tiled inference.

Tiling correctness must be established before it is used to cache thousands of
teacher outputs — a seam or a weighting error would be baked into every cached
image and silently corrupt every downstream distillation run.

These use a trivial stand-in model rather than the real teacher, so they run
without a checkpoint and test the blending logic in isolation.
"""
from __future__ import annotations

import pytest
import torch
from torch import nn

from src.models.teacher_wrapper import FrozenTeacher


class _Identity(nn.Module):
    """Stand-in whose output is exactly its input."""

    def forward(self, x):  # noqa: D102
        return x


class _Blur(nn.Module):
    """A genuinely spatial op, so seams would be visible if blending is wrong."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 3, padding=1, bias=False)
        with torch.no_grad():
            self.conv.weight.fill_(1.0 / 27.0)

    def forward(self, x):  # noqa: D102
        return self.conv(x)


def _teacher(net: nn.Module) -> FrozenTeacher:
    """Build a FrozenTeacher around ``net`` without loading a checkpoint."""
    t = FrozenTeacher.__new__(FrozenTeacher)
    nn.Module.__init__(t)
    net.eval()
    net.requires_grad_(False)
    t.net = net
    t.device = torch.device("cpu")
    t.checkpoint = None
    t.n_params = sum(p.numel() for p in net.parameters())
    t.epoch = None
    t.global_step = None
    return t


def test_tiling_of_identity_reconstructs_input_exactly() -> None:
    """Weights must sum to 1 everywhere: an identity model must round-trip."""
    t = _teacher(_Identity())
    x = torch.rand(1, 3, 300, 220)
    out = t.forward_tiled(x, tile=128, overlap=32)
    assert out.shape == x.shape
    assert torch.allclose(out, x, atol=1e-5), \
        f"max deviation {float((out - x).abs().max()):.2e} — weights do not sum to 1"


def test_tiled_matches_untiled_for_a_spatial_model() -> None:
    """The real correctness criterion: tiled output ~= single-pass output."""
    t = _teacher(_Blur())
    x = torch.rand(1, 3, 288, 288)
    full = t(x)
    tiled = t.forward_tiled(x, tile=128, overlap=32)
    diff = float((full - tiled).abs().max())
    assert diff < 2e-2, f"tiled deviates from untiled by {diff:.4f}"


def test_no_seam_at_tile_boundaries() -> None:
    """A weighting bug shows up as a periodic ridge at the stride interval.

    Compare the tiled-vs-untiled error at boundary columns against the interior:
    if blending is wrong, boundaries are markedly worse.
    """
    t = _teacher(_Blur())
    x = torch.rand(1, 3, 256, 384)
    err = (t(x) - t.forward_tiled(x, tile=128, overlap=32)).abs()[0].mean(0)
    stride = 128 - 32
    boundaries = [c for c in range(stride, x.shape[-1] - 1, stride)]
    boundary_err = float(err[:, boundaries].mean())
    interior_err = float(err.mean())
    assert boundary_err < interior_err * 3 + 1e-6, (
        f"seam detected: boundary error {boundary_err:.2e} vs "
        f"interior {interior_err:.2e}")


def test_small_image_falls_back_to_single_pass() -> None:
    t = _teacher(_Identity())
    x = torch.rand(1, 3, 64, 64)
    assert torch.allclose(t.forward_tiled(x, tile=128, overlap=32), x, atol=1e-6)


def test_non_square_and_indivisible_sizes_are_covered() -> None:
    """Sizes that do not divide evenly by the stride must still be fully covered."""
    t = _teacher(_Identity())
    for h, w in ((200, 137), (321, 481), (129, 129)):
        x = torch.rand(1, 3, h, w)
        out = t.forward_tiled(x, tile=128, overlap=32)
        assert out.shape == x.shape
        assert torch.allclose(out, x, atol=1e-5), f"failed at {h}x{w}"


def test_overlap_must_be_smaller_than_tile() -> None:
    t = _teacher(_Identity())
    with pytest.raises(ValueError, match="overlap"):
        t.forward_tiled(torch.rand(1, 3, 300, 300), tile=64, overlap=64)


def test_tiling_refuses_in_training_mode() -> None:
    t = _teacher(_Identity())
    t.net.train()
    with pytest.raises(RuntimeError, match="training mode"):
        t.forward_tiled(torch.rand(1, 3, 300, 300), tile=128, overlap=32)
