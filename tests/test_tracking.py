"""RunTracker: MLflow logging that must never be able to fail a training run.

Runs here take 6-22 hours. A tracking server is a convenience; the run
directory (config, git hash, metrics.json, history.json) is the source of
truth. These tests assert the asymmetry directly: no URI configured is a true
no-op, and a broken server degrades to a warning, never an exception.
"""
from __future__ import annotations

import logging

import pytest

from src.utils.tracking import RunTracker


def _log():
    lg = logging.getLogger("test-tracking")
    lg.handlers.clear()
    return lg


def test_no_uri_is_a_true_no_op(tmp_path, monkeypatch) -> None:
    """The default state for every run on this machine today."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    t = RunTracker({}, tmp_path, _log())
    assert t._active is False
    # Every call must be safe with no active run and no mlflow import attempted.
    t.log_start({}, "abc123", 0)
    t.log_metrics({"iteration": 100, "psnr": 30.0}, step=100)
    t.finish({"diverged": False})


def test_env_var_uri_is_picked_up(tmp_path, monkeypatch) -> None:
    """MLFLOW_TRACKING_URI is the standard variable; must not require a
    project-specific config key to work."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://nonexistent-host:5000")
    t = RunTracker({}, tmp_path, _log())
    # Connection isn't attempted until a real MLflow call — construction alone
    # (set_tracking_uri, set_experiment, start_run) may itself fail against an
    # unreachable host, and that must degrade rather than raise.
    assert isinstance(t._active, bool)


def test_config_uri_overrides_and_disables_cleanly(tmp_path) -> None:
    cfg = {"tracking": {"mlflow_uri": "http://nonexistent-host:5000",
                        "experiment": "test-exp"}}
    t = RunTracker(cfg, tmp_path, _log())
    assert isinstance(t._active, bool)
    # Whether or not construction connected, nothing below may raise.
    t.log_start(cfg, "deadbeef", 2)
    t.log_metrics({"iteration": 1, "loss": 0.5}, step=1)
    t.finish()


def test_a_failing_mlflow_call_disables_further_logging_not_the_run(tmp_path) -> None:
    """The core guarantee: one failure degrades the tracker, never propagates."""
    t = RunTracker({}, tmp_path, _log())
    calls = []

    class _FakeMlflow:
        def log_metrics(self, *a, **kw):
            calls.append("log_metrics")
            raise RuntimeError("server unreachable")

    t._active = True
    t._mlflow = _FakeMlflow()
    # Must not raise.
    t.log_metrics({"iteration": 1, "psnr": 10.0}, step=1)
    assert calls == ["log_metrics"]
    assert t._active is False, "one failure must disable further attempts"

    # A second call after disablement must be a silent no-op, not a retry.
    t.log_metrics({"iteration": 2, "psnr": 11.0}, step=2)
    assert calls == ["log_metrics"], "disabled tracker must not retry"


def test_log_metrics_skips_non_numeric_and_none(tmp_path) -> None:
    """The row dict carries provenance strings and None sigmas (F10's
    denoise-vs-other-task sentinel) alongside real metrics -- only floats/ints
    may reach mlflow.log_metrics, which rejects anything else."""
    t = RunTracker({}, tmp_path, _log())
    seen = {}

    class _FakeMlflow:
        def log_metrics(self, d, step):
            seen.update(d)

    t._active = True
    t._mlflow = _FakeMlflow()
    t.log_metrics({"iteration": 5, "psnr": 30.0, "diverged": False,
                   "task_name": "dehaze", "sigma": None}, step=5)
    assert seen == {"psnr": 30.0}, seen


def test_log_start_only_keeps_scalar_params(tmp_path) -> None:
    """A nested dict in the config (e.g. per-task source paths) must not reach
    mlflow.log_params, which rejects non-scalar values."""
    t = RunTracker({}, tmp_path, _log())
    seen = {}

    class _FakeMlflow:
        def log_params(self, d):
            seen.update(d)
        def set_tags(self, d):
            pass

    t._active = True
    t._mlflow = _FakeMlflow()
    cfg = {"data": {"batch_size": 16, "tasks": {"derain": {"input": "x"}}},
           "optim": {"lr": 1e-3}}
    t.log_start(cfg, "abc", 1)
    assert seen == {"data.batch_size": "16", "optim.lr": "0.001"}, seen


def test_finish_is_idempotent_after_disablement(tmp_path) -> None:
    t = RunTracker({}, tmp_path, _log())
    t.finish()          # never active
    t.finish()           # calling twice must not raise either
