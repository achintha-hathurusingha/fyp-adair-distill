"""Tests for the single evaluation harness."""
from __future__ import annotations

import json

import pytest
import torch

from src.eval.evaluate import EvalResult, evaluate, format_table, write_results
from src.eval.metrics import ADAIR_DEFAULT, MetricConfig


def _samples(n: int = 4, shape=(3, 32, 32), noise: float = 0.0):
    g = torch.Generator().manual_seed(0)
    for i in range(n):
        clean = torch.rand(shape, generator=g)
        degraded = (clean + noise * torch.randn(shape, generator=g)).clamp(0, 1)
        yield f"img{i}", degraded, clean


def test_perfect_model_gives_infinite_psnr() -> None:
    """An identity model on noiseless pairs is the analytic upper bound."""
    res = evaluate(lambda x: x, _samples(noise=0.0), name="identity")
    assert res.n_images == 4
    assert res.psnr == float("inf")
    assert res.ssim == pytest.approx(1.0, abs=1e-9)


def test_counts_every_image() -> None:
    res = evaluate(lambda x: x, _samples(n=7), name="count")
    assert res.n_images == 7
    assert len(res.per_image) == 7


def test_aggregation_is_mean_of_per_image_metrics() -> None:
    """Must match AdaIR's AverageMeter: per-image metrics, then mean.

    Not a global MSE converted once — the two differ on heterogeneous sets.
    """
    res = evaluate(lambda x: x * 0.9, _samples(n=5), name="agg")
    assert res.psnr == pytest.approx(
        sum(r["psnr"] for r in res.per_image) / len(res.per_image), abs=1e-9)
    assert res.ssim == pytest.approx(
        sum(r["ssim"] for r in res.per_image) / len(res.per_image), abs=1e-9)


def test_shape_mismatch_raises_rather_than_resizing() -> None:
    """A model that changes resolution is a bug, not something to paper over."""
    with pytest.raises(ValueError, match="does not match ground truth"):
        evaluate(lambda x: x[:, :, :16, :16], _samples(), name="bad")


def test_empty_dataset_raises() -> None:
    with pytest.raises(ValueError, match="no samples evaluated"):
        evaluate(lambda x: x, iter([]), name="empty")


def test_accepts_chw_and_nchw_samples() -> None:
    chw = evaluate(lambda x: x * 0.9, _samples(n=3), name="chw")
    nchw = evaluate(lambda x: x * 0.9,
                    ((n, d.unsqueeze(0), c.unsqueeze(0)) for n, d, c in _samples(n=3)),
                    name="nchw")
    assert chw.psnr == pytest.approx(nchw.psnr, abs=1e-9)


def test_config_is_recorded_with_the_numbers() -> None:
    """Every number must carry the conventions that produced it."""
    res = evaluate(lambda x: x * 0.9, _samples(), name="cfg",
                   config=MetricConfig(crop_border=2))
    assert res.config["crop_border"] == 2
    assert res.config["ssim_win_size"] == ADAIR_DEFAULT.ssim_win_size


def test_config_changes_the_result() -> None:
    a = evaluate(lambda x: x * 0.9, _samples(), name="a")
    b = evaluate(lambda x: x * 0.9, _samples(), name="b",
                 config=MetricConfig(crop_border=4))
    assert a.psnr != pytest.approx(b.psnr, abs=1e-6)


def test_gradients_are_not_tracked() -> None:
    """The harness must never build a graph, even if handed a live module."""
    lin = torch.nn.Conv2d(3, 3, 1)
    res = evaluate(lin, _samples(n=2), name="nograd")
    assert res.n_images == 2
    assert all(p.grad is None for p in lin.parameters())


def test_write_results_refuses_to_overwrite(tmp_path) -> None:
    res = evaluate(lambda x: x * 0.9, _samples(n=2), name="w")
    out = tmp_path / "metrics.json"
    write_results([res], out)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["summary"]["w"]["n_images"] == 2
    with pytest.raises(FileExistsError):
        write_results([res], out)


def test_format_table_flags_gate_failures() -> None:
    """Deltas beyond the +/-0.10 dB gate tolerance must be visibly marked."""
    good = EvalResult(name="ok", n_images=1, psnr=31.05, ssim=0.980)
    bad = EvalResult(name="off", n_images=1, psnr=30.00, ssim=0.900)
    table = format_table([good, bad],
                         reference={"ok": (31.06, 0.980), "off": (31.06, 0.980)})
    lines = {ln.split("|")[1].strip(): ln for ln in table.splitlines()}
    assert "⚠" not in lines["ok"]
    assert "⚠" in lines["off"]
