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
    if os.environ.get("ADAIR_WEIGHTS"):
        cfg["adair_weights_root"] = os.environ["ADAIR_WEIGHTS"]
    return cfg


def teacher_checkpoint(task: str) -> Path:
    """Absolute path to the released AdaIR single-degradation teacher for ``task``.

    Resolves ``adair_weights_root`` (machine-specific, set in
    ``configs/paths.local.yaml`` or ``$ADAIR_WEIGHTS``) against the portable
    filenames in ``teachers``. Exists so no tracked file carries one machine's
    absolute path — the rule stated at the top of ``configs/paths.yaml``, which
    eight files had quietly broken.

    Raises:
        ValueError: for an unknown task.
        FileNotFoundError: if the root is unset or the checkpoint is missing,
            with the instruction needed to fix it. A silent fallback here would
            surface as a mysterious load failure much later.
    """
    cfg = load_paths()
    names = cfg.get("teachers") or {}
    if task not in names:
        raise ValueError(
            f"no teacher configured for task {task!r}; "
            f"paths.yaml lists {sorted(names) or 'none'}")
    root = cfg.get("adair_weights_root")
    if not root:
        raise FileNotFoundError(
            "adair_weights_root is not set. Point it at the directory holding "
            "the released AdaIR checkpoints, either in "
            "configs/paths.local.yaml (gitignored) or via the ADAIR_WEIGHTS "
            "environment variable.")
    path = Path(root).expanduser() / names[task]
    if not path.exists():
        raise FileNotFoundError(
            f"teacher checkpoint for {task!r} not found at {path}. Check "
            "adair_weights_root and the `teachers` filenames in paths.yaml.")
    return path
