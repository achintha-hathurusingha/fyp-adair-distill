"""Non-finite gradients must never reach the weights.

B0 diverged with a gradient norm of 6.5e7 and then NaN. Gradient CLIPPING is not
a defence against this: `clip_grad_norm_` computes
``clip_coef = max_norm / (total_norm + eps)``, which for an Inf norm is ~0, and
``inf * 0 = nan`` — so clipping actively converts an Inf gradient into NaN
weights. The trainer must skip the step instead.
"""
from __future__ import annotations

import torch
from torch import nn

from src.models.nafnet import NAFNet
from src.train.trainer import Trainer


def _cfg(**over):
    cfg = {"optim": {"lr": 1e-3, "grad_clip": 1.0},
           "schedule": {"total_iters": 4, "warmup_iters": 1},
           "train": {"accum_steps": 1, "amp": False,
                     "val_every": 4, "ckpt_every": 10 ** 9},
           "loss": {"name": "charbonnier"}}
    for k, v in over.items():
        cfg.setdefault(k, {}).update(v)
    return cfg


def test_clipping_an_inf_gradient_produces_nan() -> None:
    """The failure mode this guard exists for — demonstrated, not assumed."""
    p = nn.Parameter(torch.ones(4))
    p.grad = torch.full((4,), float("inf"))
    torch.nn.utils.clip_grad_norm_([p], 1.0)
    assert torch.isnan(p.grad).all(), (
        "expected clip_grad_norm_ to turn Inf into NaN; if this ever stops "
        "being true the guard's rationale should be revisited")


def test_nonfinite_gradient_step_is_skipped_not_applied(tmp_path) -> None:
    """Weights must be untouched by a step with a non-finite gradient."""
    torch.manual_seed(0)
    model = NAFNet(width=4, enc_blk_nums=[1], middle_blk_num=1, dec_blk_nums=[1])
    batches = [(torch.rand(2, 3, 32, 32), torch.rand(2, 3, 32, 32), 15)
               for _ in range(4)]
    trainer = Trainer(model, batches, _cfg(), tmp_path, device="cpu")

    # Force every gradient norm to be non-finite.
    import src.train.trainer as tr_mod
    monkey = tr_mod.grad_norm
    tr_mod.grad_norm = lambda m: float("inf")
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    try:
        state = trainer.train()
    finally:
        tr_mod.grad_norm = monkey

    for k, v in model.state_dict().items():
        assert torch.equal(v, before[k]), f"{k} changed despite skipped steps"
        assert torch.isfinite(v).all(), f"{k} contains non-finite values"
    assert state.history[-1]["nonfinite_skips"] > 0, "skips were not counted"


def test_skips_are_counted_and_logged(tmp_path) -> None:
    """A run surviving only by skipping steps must not look healthy."""
    torch.manual_seed(0)
    model = NAFNet(width=4, enc_blk_nums=[1], middle_blk_num=1, dec_blk_nums=[1])
    batches = [(torch.rand(2, 3, 32, 32), torch.rand(2, 3, 32, 32), 15)
               for _ in range(4)]
    trainer = Trainer(model, batches, _cfg(), tmp_path, device="cpu")
    import src.train.trainer as tr_mod
    monkey = tr_mod.grad_norm
    tr_mod.grad_norm = lambda m: float("nan")
    try:
        state = trainer.train()
    finally:
        tr_mod.grad_norm = monkey
    assert state.history[-1]["nonfinite_skips"] == 4


def test_finite_gradients_still_step(tmp_path) -> None:
    """The guard must not block normal training."""
    torch.manual_seed(0)
    model = NAFNet(width=4, enc_blk_nums=[1], middle_blk_num=1, dec_blk_nums=[1])
    batches = [(torch.rand(2, 3, 32, 32), torch.rand(2, 3, 32, 32), 15)
               for _ in range(4)]
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    state = Trainer(model, batches, _cfg(), tmp_path, device="cpu").train()
    assert state.history[-1]["nonfinite_skips"] == 0
    changed = any(not torch.equal(v, before[k])
                  for k, v in model.state_dict().items())
    assert changed, "no parameter moved — training did not happen"


def test_residual_weight_decay_creates_separate_group(tmp_path) -> None:
    """beta/gamma go in their own group when residual_weight_decay is set."""
    model = NAFNet(width=4, enc_blk_nums=[1], middle_blk_num=1, dec_blk_nums=[1])
    cfg = _cfg()
    cfg["optim"]["residual_weight_decay"] = 0.01
    t = Trainer(model, [], cfg, tmp_path, device="cpu")
    groups = t.optimizer.param_groups
    assert len(groups) == 2, f"expected 2 param groups, got {len(groups)}"
    assert groups[1]["weight_decay"] == 0.01
    n_res = sum(1 for n, _ in model.named_parameters()
                if n.endswith((".beta", ".gamma")))
    assert len(groups[1]["params"]) == n_res


def test_residual_weight_decay_absent_keeps_single_group(tmp_path) -> None:
    """Opt-in: absent, behaviour is unchanged."""
    model = NAFNet(width=4, enc_blk_nums=[1], middle_blk_num=1, dec_blk_nums=[1])
    t = Trainer(model, [], _cfg(), tmp_path, device="cpu")
    assert len(t.optimizer.param_groups) == 1


def test_trace_writes_every_step_and_flushes(tmp_path) -> None:
    """The trace must survive the crash it exists to diagnose.

    B0 died between two validation points having logged only "clip 1/5000" —
    enough to know one step spiked to 6.5e7, not enough to know when, or what
    the loss was doing around it. So the trace is per-step and flushed per row.
    """
    import csv

    torch.manual_seed(0)
    model = NAFNet(width=4, enc_blk_nums=[1], middle_blk_num=1, dec_blk_nums=[1])
    batches = [(torch.rand(2, 3, 32, 32), torch.rand(2, 3, 32, 32), 15)
               for _ in range(6)]
    cfg = _cfg()
    cfg["schedule"]["total_iters"] = 6
    cfg["train"]["val_every"] = 10 ** 9
    cfg["train"]["trace_every"] = 1
    Trainer(model, batches, cfg, tmp_path, device="cpu").train()

    path = tmp_path / "trace.csv"
    assert path.exists(), "no trace.csv written"
    rows = list(csv.DictReader(path.open()))
    assert len(rows) == 6, f"expected one row per step, got {len(rows)}"
    assert list(rows[0]) == ["iteration", "lr", "loss", "grad_norm"]
    for r in rows:
        assert float(r["grad_norm"]) >= 0.0
        assert float(r["loss"]) == float(r["loss"])      # not NaN


def test_trace_disabled_by_default(tmp_path) -> None:
    torch.manual_seed(0)
    model = NAFNet(width=4, enc_blk_nums=[1], middle_blk_num=1, dec_blk_nums=[1])
    batches = [(torch.rand(2, 3, 32, 32), torch.rand(2, 3, 32, 32), 15)
               for _ in range(4)]
    Trainer(model, batches, _cfg(), tmp_path, device="cpu").train()
    assert not (tmp_path / "trace.csv").exists()


def test_spike_dump_captures_the_offending_batch(tmp_path) -> None:
    """The batch that caused an anomalous gradient must be recoverable.

    Reconstructing it from the seed offline is fragile — model init and RNG
    restore both consume the global stream — so the trainer saves it in the act.
    """
    torch.manual_seed(0)
    model = NAFNet(width=4, enc_blk_nums=[1], middle_blk_num=1, dec_blk_nums=[1])
    batches = [(torch.rand(2, 3, 32, 32), torch.rand(2, 3, 32, 32), 15)
               for _ in range(4)]
    cfg = _cfg()
    cfg["schedule"]["total_iters"] = 4
    cfg["train"]["val_every"] = 10 ** 9
    cfg["train"]["spike_dump_threshold"] = 1e-12   # everything counts as a spike
    Trainer(model, batches, cfg, tmp_path, device="cpu").train()

    dumps = sorted((tmp_path / "spikes").glob("step_*.pt"))
    assert dumps, "no spike dump written"
    payload = torch.load(dumps[0], weights_only=False)
    assert "micro_batches" in payload and payload["micro_batches"]
    d, c = payload["micro_batches"][0]
    assert d.shape == (2, 3, 32, 32) and c.shape == (2, 3, 32, 32)
    assert payload["grad_norm"] > 0


def test_spike_dump_holds_only_the_current_step(tmp_path) -> None:
    """The buffer must not accumulate across steps — one step's worth only."""
    torch.manual_seed(0)
    model = NAFNet(width=4, enc_blk_nums=[1], middle_blk_num=1, dec_blk_nums=[1])
    accum = 2
    batches = [(torch.rand(2, 3, 32, 32), torch.rand(2, 3, 32, 32), 15)
               for _ in range(6)]
    cfg = _cfg()
    cfg["schedule"]["total_iters"] = 3
    cfg["train"]["accum_steps"] = accum
    cfg["train"]["val_every"] = 10 ** 9
    cfg["train"]["spike_dump_threshold"] = 1e-12
    Trainer(model, batches, cfg, tmp_path, device="cpu").train()

    for dump in sorted((tmp_path / "spikes").glob("step_*.pt")):
        payload = torch.load(dump, weights_only=False)
        assert len(payload["micro_batches"]) == accum, (
            f"expected {accum} micro-batches, got "
            f"{len(payload['micro_batches'])} — buffer is leaking across steps")


def test_spike_dump_disabled_by_default(tmp_path) -> None:
    torch.manual_seed(0)
    model = NAFNet(width=4, enc_blk_nums=[1], middle_blk_num=1, dec_blk_nums=[1])
    batches = [(torch.rand(2, 3, 32, 32), torch.rand(2, 3, 32, 32), 15)
               for _ in range(4)]
    Trainer(model, batches, _cfg(), tmp_path, device="cpu").train()
    assert not (tmp_path / "spikes").exists()
