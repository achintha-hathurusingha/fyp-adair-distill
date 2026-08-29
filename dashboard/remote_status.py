#!/usr/bin/env python3
"""Dashboard v2 -- remote-side status reader, deployed to /tmp/remote_status.py
on each training host (devon, qbits). Stdlib only.

Reads run state straight off disk, never touches the training process:
  RUNS_ROOT   env var, default ~/fyp-adair-distill/runs
  ARMS        env var, comma-separated "out_root/ARM_NAME" pairs to track on
              THIS host, e.g. "b0v2_kd_feat/B0V2-KD-FEAT,b0v2_kd_feat_cond/B0V2-KD-FEAT-COND"

For each configured arm: finds the most-recently-modified
<out_root>/<ARM>/<ARM>_seed*_* run directory, reads history.json's last
entry (the structured per-checkpoint metrics Trainer._dump_history already
writes -- psnr_denoise/psnr_derain/psnr_dehaze, clamp/grad diagnostics,
feat_last/aux_last), and tails that run's own train.log for the live text
stream between checkpoints. Also returns a trimmed copy of the FULL history
(iteration/elapsed_s/psnr* only, dropping the clamp/grad-norm/vram debug
fields) so the frontend can chart PSNR-over-iterations, not just show the
latest point.

Prints one JSON line to stdout and exits -- local_server.py runs this over
SSH once per poll.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import time

RUNS_ROOT = os.environ.get("RUNS_ROOT", os.path.expanduser("~/fyp-adair-distill/runs"))
ARMS = [a for a in os.environ.get("ARMS", "").split(",") if a]
STALE_AFTER_S = 3600
LOG_TAIL_LINES = 12


def _gpu_status() -> list[dict]:
    """Per-GPU VRAM/utilization on THIS host, independent of whether any arm
    is configured here -- this is what lets qbits' free capacity be seen even
    while it's idle, so a new workload can actually be placed on it instead
    of guessing.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.used,memory.total,memory.free,"
             "utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        if out.returncode != 0:
            return [{"error": out.stderr.strip()[:300]}]
        gpus = []
        for line in out.stdout.strip().splitlines():
            idx, name, used, total, free, util, temp = [p.strip() for p in line.split(",")]
            gpus.append({
                "index": int(idx), "name": name,
                "mem_used_mb": int(used), "mem_total_mb": int(total),
                "mem_free_mb": int(free), "util_pct": int(util),
                "temp_c": int(temp),
            })
        return gpus
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return [{"error": f"{type(e).__name__}: {e}"}]


def _latest_run_dir(out_root: str, arm: str) -> str | None:
    pattern = os.path.join(RUNS_ROOT, out_root, arm, f"{arm}_seed*_*")
    candidates = glob.glob(pattern)
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def _tail(path: str, n: int) -> list[str]:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return [l.rstrip("\n") for l in lines[-n:]]
    except FileNotFoundError:
        return []


HISTORY_KEYS = ("iteration", "elapsed_s", "psnr", "psnr_denoise", "psnr_derain", "psnr_dehaze")


def _trim(entry: dict) -> dict:
    return {k: entry[k] for k in HISTORY_KEYS if k in entry}


def _arm_status(out_root: str, arm: str) -> dict:
    run_dir = _latest_run_dir(out_root, arm)
    if run_dir is None:
        return {"arm": arm, "error": f"no run directory found under {out_root}/{arm}"}

    hist_path = os.path.join(run_dir, "history.json")
    last = {}
    history_trimmed = []
    try:
        with open(hist_path, "r", encoding="utf-8") as f:
            history = json.load(f)
        if history:
            last = history[-1]
            history_trimmed = [_trim(h) for h in history]
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    log_path = os.path.join(run_dir, "train.log")
    log_mtime = os.path.getmtime(log_path) if os.path.exists(log_path) else 0
    stale = (time.time() - log_mtime) > STALE_AFTER_S if log_mtime else True

    return {
        "arm": arm,
        "run_dir": os.path.basename(run_dir),
        "latest": last,
        "history": history_trimmed,
        "log_tail": _tail(log_path, LOG_TAIL_LINES),
        "stale": stale,
        "last_update_s_ago": (time.time() - log_mtime) if log_mtime else None,
    }


def main() -> None:
    arms_out = []
    for spec in ARMS:
        if "/" not in spec:
            continue
        out_root, arm = spec.split("/", 1)
        arms_out.append(_arm_status(out_root, arm))

    print(json.dumps({
        "hostname": os.uname().nodename if hasattr(os, "uname") else "unknown",
        "timestamp": time.time(),
        "gpus": _gpu_status(),
        "arms": arms_out,
    }))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa -- must always emit valid JSON, even on error
        print(json.dumps({"error": f"{type(e).__name__}: {e}"}))
        sys.exit(0)
