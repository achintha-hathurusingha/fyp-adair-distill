"""Tests for gradient accumulation.

F8 locks "effective batch 32 everywhere, via accumulation when the micro-batch
does not fit", and justifies it by claiming accumulation is equivalent to a true
large batch for this architecture (no batch statistics anywhere). That claim is
the reason the batch-32 S-arm numbers stay valid, so it is tested rather than
asserted.
"""
from __future__ import annotations

import pytest
import torch
from torch import nn


def _model(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(),
                         nn.Conv2d(8, 3, 3, padding=1))


def _grads(model: nn.Module) -> list[torch.Tensor]:
    return [p.grad.detach().clone() for p in model.parameters() if p.grad is not None]


def test_accumulated_gradient_matches_single_large_batch() -> None:
    """The core F8 claim: 2 micro-steps of 8 == 1 step of 16.

    Requires dividing each micro-batch loss by accum_steps, so the accumulated
    gradient AVERAGES rather than sums. Without that division the effective
    learning rate silently scales with accum_steps.
    """
    torch.manual_seed(42)
    x = torch.randn(16, 3, 16, 16)
    y = torch.randn(16, 3, 16, 16)
    crit = nn.L1Loss()

    # --- one full batch of 16 ---
    big = _model()
    big.zero_grad(set_to_none=True)
    crit(big(x), y).backward()
    big_grads = _grads(big)

    # --- two micro-batches of 8, accumulated ---
    small = _model()
    small.zero_grad(set_to_none=True)
    accum = 2
    for i in range(accum):
        sl = slice(i * 8, (i + 1) * 8)
        (crit(small(x[sl]), y[sl]) / accum).backward()
    small_grads = _grads(small)

    assert len(big_grads) == len(small_grads)
    for b, s in zip(big_grads, small_grads):
        assert torch.allclose(b, s, atol=1e-6), \
            f"max deviation {float((b - s).abs().max()):.2e}"


def test_summing_without_division_inflates_gradients() -> None:
    """Guard the failure mode: forgetting /accum_steps scales the gradient."""
    torch.manual_seed(42)
    x, y = torch.randn(16, 3, 8, 8), torch.randn(16, 3, 8, 8)
    crit = nn.L1Loss()

    big = _model()
    big.zero_grad(set_to_none=True)
    crit(big(x), y).backward()

    wrong = _model()
    wrong.zero_grad(set_to_none=True)
    for i in range(2):                       # NO division — the bug
        sl = slice(i * 8, (i + 1) * 8)
        crit(wrong(x[sl]), y[sl]).backward()

    b, w = _grads(big)[0], _grads(wrong)[0]
    assert not torch.allclose(b, w, atol=1e-6)
    # Summing two averaged half-batches gives ~2x the correct gradient.
    assert torch.allclose(w, b * 2, atol=1e-5)


@pytest.mark.parametrize("accum", [1, 2, 4])
def test_iteration_counts_optimizer_steps_not_micro_batches(accum: int, tmp_path) -> None:
    """`total_iters` must mean optimizer steps regardless of accumulation.

    Otherwise the LR schedule and the run length change meaning when a config
    switches to accumulation, and runs stop being comparable.
    """
    from src.models.nafnet import NAFNet
    from src.train.trainer import Trainer

    steps = 6
    batches = [(torch.rand(2, 3, 32, 32), torch.rand(2, 3, 32, 32), 15)
               for _ in range(steps * accum)]

    cfg = {"optim": {"lr": 1e-3}, "schedule": {"total_iters": steps,
                                               "warmup_iters": 1},
           "train": {"accum_steps": accum, "amp": False,
                     "val_every": 10**9, "ckpt_every": 10**9},
           "loss": {"name": "charbonnier"}}
    model = NAFNet(width=4, enc_blk_nums=[1], middle_blk_num=1, dec_blk_nums=[1])
    trainer = Trainer(model, batches, cfg, tmp_path, device="cpu")
    state = trainer.train()
    assert state.iteration == steps, (
        f"accum={accum}: got {state.iteration} iterations, expected {steps} "
        "optimizer steps")


def test_accum_steps_defaults_to_one(tmp_path) -> None:
    """Absent config, behaviour is unchanged from a plain training loop."""
    from src.models.nafnet import NAFNet
    from src.train.trainer import Trainer

    model = NAFNet(width=4, enc_blk_nums=[1], middle_blk_num=1, dec_blk_nums=[1])
    t = Trainer(model, [], {"optim": {"lr": 1e-3},
                            "schedule": {"total_iters": 1},
                            "loss": {"name": "charbonnier"}}, tmp_path,
                device="cpu")
    assert t.accum_steps == 1


def test_clip_rate_is_reported_and_counted_per_interval(tmp_path) -> None:
    """The clip-hit rate is the B0 watch signal, so it is tested not assumed.

    A near-zero clip threshold forces every optimizer step to clip, which pins
    both the count and the denominator (optimizer steps in the interval, NOT
    micro-batches). It must be positive, not 0.0 -- the trainer treats a falsy
    threshold as "clipping disabled", same as None.
    """
    from src.models.nafnet import NAFNet
    from src.train.trainer import Trainer

    steps, accum, val_every = 6, 2, 3
    batches = [(torch.rand(2, 3, 32, 32), torch.rand(2, 3, 32, 32), 15)
               for _ in range(steps * accum)]
    cfg = {"optim": {"lr": 1e-3, "grad_clip": 1e-8},
           "schedule": {"total_iters": steps, "warmup_iters": 1},
           "train": {"accum_steps": accum, "amp": False,
                     "val_every": val_every, "ckpt_every": 10**9},
           "loss": {"name": "charbonnier"}}
    model = NAFNet(width=4, enc_blk_nums=[1], middle_blk_num=1, dec_blk_nums=[1])
    state = Trainer(model, batches, cfg, tmp_path, device="cpu").train()

    rows = [r for r in state.history if "clip_rate" in r]
    assert rows, "clip_rate must be recorded in history"
    for row in rows:
        assert row["clip_hits"] == val_every, (
            f"every step should clip: {row['clip_hits']} of {val_every}")
        assert row["clip_rate"] == 1.0, row["clip_rate"]


def test_clip_rate_is_zero_when_clipping_never_fires(tmp_path) -> None:
    """A loose threshold (B0 uses 8.0) should leave the rate at zero."""
    from src.models.nafnet import NAFNet
    from src.train.trainer import Trainer

    batches = [(torch.rand(2, 3, 32, 32), torch.rand(2, 3, 32, 32), 15)
               for _ in range(8)]
    cfg = {"optim": {"lr": 1e-4, "grad_clip": 1e9},
           "schedule": {"total_iters": 4, "warmup_iters": 1},
           "train": {"accum_steps": 2, "amp": False,
                     "val_every": 2, "ckpt_every": 10**9},
           "loss": {"name": "charbonnier"}}
    model = NAFNet(width=4, enc_blk_nums=[1], middle_blk_num=1, dec_blk_nums=[1])
    state = Trainer(model, batches, cfg, tmp_path, device="cpu").train()
    for row in state.history:
        assert row["clip_hits"] == 0 and row["clip_rate"] == 0.0
