"""Tests for checkpoint save/resume.

Resume is the mechanism that makes long unattended runs survivable, and it has
broken twice in this project — most recently because a checkpoint loaded with
``map_location="cuda"`` returned the torch RNG state as a CUDA tensor, which
``torch.set_rng_state`` rejects. Both failures only surfaced when an arm
actually tried to resume, hours into a run. These tests make them surface in CI.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.utils.seeding import (capture_rng_state, restore_rng_state,
                               seed_everything)


def test_rng_state_round_trips() -> None:
    seed_everything(1234)
    state = capture_rng_state()
    before = (torch.rand(4), np.random.rand(4))

    seed_everything(999)          # perturb every generator
    restore_rng_state(state)
    after = (torch.rand(4), np.random.rand(4))

    assert torch.allclose(before[0], after[0])
    assert np.allclose(before[1], after[1])


def test_rng_state_survives_a_cuda_map_location_round_trip(tmp_path) -> None:
    """The exact failure that broke --resume mid-run.

    ``torch.load(..., map_location=device)`` moves the saved RNG state onto that
    device; ``set_rng_state`` then raises
    ``TypeError: RNG state must be a torch.ByteTensor``.
    """
    seed_everything(7)
    state = capture_rng_state()
    expected = torch.rand(4)

    path = tmp_path / "ck.pth"
    torch.save({"rng_python": state.python, "rng_numpy": state.numpy,
                "rng_torch": state.torch, "rng_cuda": state.torch_cuda}, path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(path, map_location=device, weights_only=False)

    from src.utils.seeding import RNGState
    seed_everything(999)
    restore_rng_state(RNGState(ck["rng_python"], ck["rng_numpy"],
                               ck["rng_torch"], ck["rng_cuda"]))
    assert torch.allclose(expected, torch.rand(4))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_rng_state_accepts_an_explicitly_cuda_tensor() -> None:
    """Directly assert the coercion, independent of torch.load behaviour."""
    seed_everything(11)
    state = capture_rng_state()
    expected = torch.rand(4)

    from src.utils.seeding import RNGState
    moved = RNGState(state.python, state.numpy, state.torch.cuda(),
                     state.torch_cuda)
    seed_everything(999)
    restore_rng_state(moved)                     # must not raise
    assert torch.allclose(expected, torch.rand(4))


def test_trainer_checkpoint_round_trip(tmp_path) -> None:
    """Save then load a real Trainer checkpoint and confirm state is restored."""
    from src.models.nafnet import NAFNet
    from src.train.trainer import Trainer

    cfg = {"optim": {"lr": 1e-3}, "schedule": {"total_iters": 10},
           "train": {"val_every": 1000, "ckpt_every": 1000},
           "loss": {"name": "charbonnier"}}
    model = NAFNet(width=8, enc_blk_nums=[1, 1], middle_blk_num=1,
                   dec_blk_nums=[1, 1])
    trainer = Trainer(model, [], cfg, tmp_path, device="cpu")
    trainer.state.iteration = 4242
    trainer.state.best_psnr = 31.019
    trainer.save_checkpoint(tmp_path / "last.pth")

    fresh = Trainer(NAFNet(width=8, enc_blk_nums=[1, 1], middle_blk_num=1,
                           dec_blk_nums=[1, 1]), [], cfg, tmp_path, device="cpu")
    fresh.load_checkpoint(tmp_path / "last.pth")

    assert fresh.state.iteration == 4242
    assert fresh.state.best_psnr == pytest.approx(31.019)
    for a, b in zip(model.parameters(), fresh.model.parameters()):
        assert torch.allclose(a, b)
