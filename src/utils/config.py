"""Config loading with strict access — no silent fallbacks (rule 9).

Paths resolve from ``configs/paths.yaml`` (or ``paths.local.yaml`` if present),
with ``DATA_ROOT`` / ``RUNS_ROOT`` / ``ADAIR_CKPT`` env vars overriding.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file, raising a clear error if it is missing."""
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        raise ValueError(f"Config file is empty: {p}")
    return data


def require(cfg: dict[str, Any], key: str, *, context: str = "") -> Any:
    """Fetch ``key`` from ``cfg`` or raise — never default-and-continue."""
    if key not in cfg:
        where = f" in {context}" if context else ""
        raise KeyError(f"Required config key '{key}' missing{where}. Present keys: {sorted(cfg)}")
    return cfg[key]


def load_paths() -> dict[str, Any]:
    """Load paths.yaml (preferring paths.local.yaml), applying env overrides."""
    local = REPO_ROOT / "configs" / "paths.local.yaml"
    cfg = load_yaml(local if local.exists() else REPO_ROOT / "configs" / "paths.yaml")
    cfg["data_root"] = os.environ.get("DATA_ROOT", cfg.get("data_root", "./data"))
    cfg["runs_root"] = os.environ.get("RUNS_ROOT", cfg.get("runs_root", "./runs"))
    if os.environ.get("ADAIR_CKPT"):
        cfg["adair_checkpoint"] = os.environ["ADAIR_CKPT"]
    return cfg
