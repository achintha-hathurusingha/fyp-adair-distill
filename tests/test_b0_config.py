"""B0's resolved config must equal its reviewed YAML.

B0 is a multi-day, three-seed run that a supervisor signs off by reading
`configs/train/b0_baseline.yaml`. If the code that actually trains carries its
own copy of those constants, the two drift and the run directory records
something nobody approved. These tests pin the YAML as the authority.
"""
from __future__ import annotations

import pytest

from src.train.train import ARMS, build_config
from src.utils.config import load_yaml

YAML = load_yaml("configs/train/b0_baseline.yaml")


def _cfg() -> dict:
    # Deliberately absurd CLI values: none of them may reach the resolved config.
    return build_config("B0", iters=7, batch_size=999, lr=9.9, patch_size=999)


@pytest.mark.parametrize("section", ["data", "optim", "schedule", "train", "loss"])
def test_yaml_wins_over_cli_and_defaults(section: str) -> None:
    cfg = _cfg()
    for key, want in YAML.get(section, {}).items():
        assert cfg[section][key] == want, (
            f"{section}.{key}: resolved {cfg[section][key]!r}, YAML says {want!r}")


def test_cli_cannot_shorten_or_reshape_the_run() -> None:
    cfg = _cfg()
    assert cfg["schedule"]["total_iters"] == 300_000
    assert cfg["data"]["batch_size"] == 16
    assert cfg["data"]["patch_size"] == 128
    assert cfg["optim"]["lr"] == 1.0e-3


def test_effective_batch_is_32() -> None:
    """The whole point of accumulation here (findings F8)."""
    cfg = _cfg()
    assert cfg["data"]["batch_size"] * cfg["train"]["accum_steps"] == 32


def test_architecture_matches_the_locked_m_arm() -> None:
    cfg = _cfg()
    assert cfg["model"] == {
        "width": 16, "enc_blk_nums": [2, 2, 4, 8], "middle_blk_num": 12,
        "dec_blk_nums": [2, 2, 2, 2], "norm_type": "layernorm2d",
        # affine_clamp, not plain affine: findings F9. Plain affine at full
        # resolution is the configuration that diverged at iteration 24356.
        "full_res_norm_type": "affine_clamp", "clamp_bound": 8.0}


def test_b0_carries_no_distillation_term() -> None:
    """Phase 01 admits reconstruction only. A teacher here voids the baseline."""
    assert YAML["loss"]["name"] == "charbonnier"
    assert set(YAML["loss"]) <= {"name", "eps"}, YAML["loss"]
    flat = repr(YAML).lower()
    for banned in ("teacher", "distill", "kd_", "adair_ckpt"):
        assert banned not in flat, f"B0 config mentions {banned!r}"


def test_architecture_drift_raises(tmp_path, monkeypatch) -> None:
    """A YAML/code disagreement must be an error, never a silent merge."""
    import yaml as _yaml

    bad = dict(YAML)
    bad["arch"] = {**YAML["arch"], "width": 24}      # code says 16
    p = tmp_path / "drift.yaml"
    p.write_text(_yaml.safe_dump(bad), encoding="utf-8")

    monkeypatch.setitem(ARMS["B0"], "config", str(p))
    with pytest.raises(ValueError, match="architecture drift"):
        _cfg()


def test_num_workers_override_is_recorded_not_just_applied(monkeypatch, tmp_path) -> None:
    """The run directory must record the workers actually used.

    A CLI override applied AFTER create_run_dir writes config.yaml leaves the
    directory claiming the YAML value while the run uses another — the same
    provenance failure as an un-recorded resume. Order matters, so it is pinned.
    """
    import sys
    from src.train import train as train_mod

    captured = {}

    def fake_create_run_dir(runs_root, experiment, *, config, seed):
        captured["config"] = config
        d = tmp_path / "run"
        d.mkdir(exist_ok=True)
        return d

    def stop(*a, **k):
        raise SystemExit(0)          # abort before any real work

    monkeypatch.setattr(train_mod, "create_run_dir", fake_create_run_dir)
    monkeypatch.setattr(train_mod, "build_train_loader", stop)
    monkeypatch.setattr(sys, "argv",
                        ["train", "--arm", "B0", "--num-workers", "12",
                         "--device", "cpu"])
    with pytest.raises(SystemExit):
        train_mod.main()

    assert captured["config"]["data"]["num_workers"] == 12, (
        "run directory recorded "
        f"{captured['config']['data']['num_workers']}, CLI asked for 12")
