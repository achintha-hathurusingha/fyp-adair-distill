"""Per-host status gatherer for the unified dashboard.

Runs ON a training host (devon or qbits), reads that host's OWN runs/ and
log file, prints one JSON line to stdout. The local aggregator (see
local_server.py) SSHes in, executes this, and merges both hosts' output —
this script never talks over the network itself.

Usage: python3 remote_status.py <repo_root> <log_path> <arm1> [arm2 ...]
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
import time
from pathlib import Path

TEACHER_PSNR = 34.5056


def _latest_run_dir(runs_root: Path, arm_exact: str, seed: int) -> Path | None:
    pattern = str(runs_root / "*" / arm_exact / f"{arm_exact}_seed{seed}_*")
    candidates = [Path(p) for p in glob.glob(pattern) if Path(p).is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _read_history(run_dir: Path) -> list[dict]:
    hf = run_dir / "history.json"
    if not hf.exists():
        return []
    try:
        return json.loads(hf.read_text())
    except (json.JSONDecodeError, OSError):
        return []


def _read_total_iters(run_dir: Path, default: int = 60000) -> int:
    cfg = run_dir / "config.yaml"
    if not cfg.exists():
        return default
    try:
        for line in cfg.read_text().splitlines():
            m = re.match(r"\s*total_iters:\s*(\d+)", line)
            if m:
                return int(m.group(1))
    except OSError:
        pass
    return default


def _seed_snapshot(runs_root: Path, arm_exact: str, seed: int) -> dict:
    run_dir = _latest_run_dir(runs_root, arm_exact, seed)
    if run_dir is None:
        return {"seed": seed, "status": "pending", "iteration": 0,
                "target_iters": 60000, "history": []}

    history = _read_history(run_dir)
    total_iters = _read_total_iters(run_dir)
    last = history[-1] if history else None
    iteration = last["iteration"] if last else 0
    completed_marker = (run_dir / "last.pth").exists() and iteration >= total_iters

    hf = run_dir / "history.json"
    fresh = hf.exists() and (time.time() - hf.stat().st_mtime) < 1800
    just_started = (not history and not hf.exists()
                     and (run_dir / "config.yaml").exists()
                     and (time.time() - (run_dir / "config.yaml").stat().st_mtime) < 1800)

    if completed_marker:
        status = "done"
    elif fresh or just_started:
        status = "running"
    elif history:
        status = "stalled"
    else:
        status = "pending"

    return {
        "seed": seed,
        "run_dir": run_dir.name,
        "status": status,
        "iteration": iteration,
        "target_iters": total_iters,
        "progress": round(iteration / total_iters, 4) if total_iters else 0,
        "latest": last,
        "history": history,
    }


def _arm_summary(runs_root: Path, arm_exact: str, seeds=(0, 1, 2)) -> dict:
    seed_snaps = [_seed_snapshot(runs_root, arm_exact, s) for s in seeds]
    bests = [max((h["psnr"] for h in s["history"]), default=None)
             for s in seed_snaps]
    bests = [b for b in bests if b is not None]
    mean_best_psnr = sum(bests) / len(bests) if bests else None
    return {
        "arm": arm_exact,
        "seeds": seed_snaps,
        "mean_best_psnr": mean_best_psnr,
        "gap_to_teacher": (TEACHER_PSNR - mean_best_psnr) if mean_best_psnr else None,
        "n_seeds_reporting": len(bests),
    }


def _tail_log(path: Path, n: int = 40) -> list[str]:
    if not path.exists():
        return []
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            read_size = min(size, 65536)
            f.seek(size - read_size, os.SEEK_SET)
            chunk = f.read(read_size)
        lines = chunk.decode("utf-8", errors="replace").splitlines()
        return lines[-n:]
    except OSError:
        return []


def main():
    repo_root = Path(sys.argv[1])
    log_path = Path(sys.argv[2])
    arm_names = sys.argv[3:]
    runs_root = repo_root / "runs"

    log_tail = _tail_log(log_path)
    crash = any(re.search(r"Traceback|CUDA out of memory|nonfinite|NaN\b|Error", ln)
                for ln in log_tail)

    print(json.dumps({
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "crash_detected": crash,
        "log_tail": log_tail,
        "arms": {arm: _arm_summary(runs_root, arm) for arm in arm_names},
    }))


if __name__ == "__main__":
    main()
