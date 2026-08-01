"""Run-directory creation (rule 6).

Every experiment writes a directory containing: the resolved config, the git
commit hash, an environment dump (``pip freeze``), logs, checkpoints, and
``metrics.json``.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def _git_commit() -> str:
    """Return the current git commit hash, or 'unknown' outside a repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _pip_freeze(timeout: int = 60) -> str:
    """Return ``pip freeze`` output for the active environment.

    Bounded by a timeout: during an unattended run this call blocked for ~3.5
    hours while contending with concurrent network jobs, silently delaying
    training start. An environment dump is useful provenance but is never worth
    stalling a run for, so a timeout degrades to a recorded note instead.
    """
    try:
        out = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True, text=True, check=True, timeout=timeout,
        )
        return out.stdout
    except subprocess.TimeoutExpired:
        return (f"pip freeze timed out after {timeout}s; environment not "
                "captured. See requirements.txt for the intended pins.\n")
    except (subprocess.CalledProcessError, OSError) as exc:  # pragma: no cover
        return f"pip freeze failed: {exc}\n"


def create_run_dir(runs_root: str | Path, experiment: str, *,
                   config: dict[str, Any], seed: int) -> Path:
    """Create ``runs_root/<experiment>_<seed>_<timestamp>/`` and dump metadata.

    Writes: ``config.yaml``, ``git_commit.txt``, ``env.txt`` (pip freeze),
    ``seed.txt``. Raises if the target already exists (no silent overwrite).
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(runs_root) / f"{experiment}_seed{seed}_{ts}"
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)

    with (run_dir / "config.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, sort_keys=False)
    (run_dir / "git_commit.txt").write_text(_git_commit() + "\n", encoding="utf-8")
    (run_dir / "env.txt").write_text(_pip_freeze(), encoding="utf-8")
    (run_dir / "seed.txt").write_text(f"{seed}\n", encoding="utf-8")
    return run_dir


def write_metrics(run_dir: str | Path, metrics: dict[str, Any]) -> None:
    """Write ``metrics.json`` into the run directory."""
    with (Path(run_dir) / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, sort_keys=True)


def record_resume(run_dir: str | Path, config: dict[str, Any], *,
                  iteration: int, reason: str = "") -> Path:
    """Append a provenance record for a resumed run.

    A resume reuses the existing run directory, so ``config.yaml`` and
    ``git_commit.txt`` keep describing the ORIGINAL launch. When a run is
    resumed after a code or config change — which is exactly why B0 seed 0 was
    resumed at iteration 5000, to pick up the bounded image cache — the
    directory would otherwise misrepresent what actually produced the weights.

    Each resume appends one JSON line to ``resumes.jsonl`` recording the config
    and commit *in force from that iteration onward*. The original files are
    never rewritten: they are the record of the first launch.
    """
    run_dir = Path(run_dir)
    record = {
        "resumed_at": datetime.now().isoformat(timespec="seconds"),
        "from_iteration": iteration,
        "git_commit": _git_commit(),
        "reason": reason,
        "config": config,
    }
    path = run_dir / "resumes.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True, default=str) + "\n")
    return path
