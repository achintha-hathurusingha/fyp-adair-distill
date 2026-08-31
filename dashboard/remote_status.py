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


def _cache_jobs() -> list[dict]:
    """Teacher-cache builds running on THIS host.

    Reported because such a build occupies the GPU for ~an hour while producing
    no training log, so on an arms-only board it looks like nothing is happening
    -- or worse, like an arm has stalled. Progress comes from bytes written vs
    the expected total (the builder is near-silent until it exits).
    """
    out = []
    try:
        ps = subprocess.run(
            ["ps", "-eo", "pid=,args="], capture_output=True, text=True, timeout=8)
    except Exception:
        return out
    for line in ps.stdout.splitlines():
        line = line.strip()
        if "build_teacher_cache.py" not in line or "ps -eo" in line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        pid, args = parts[0], parts[1]
        out_dir = None
        toks = args.split()
        for i, t in enumerate(toks):
            if t == "--out-dir" and i + 1 < len(toks):
                out_dir = toks[i + 1]
        if any(j.get("out_dir") == out_dir for j in out):
            continue                      # dataloader workers share the cmdline
        # The builder is launched with a repo-relative --out-dir, but this
        # script runs from the SSH login directory, so resolve against the repo
        # root (RUNS_ROOT's parent) or the size reads 0 and the bar never moves.
        resolved = out_dir
        if out_dir and not os.path.isabs(out_dir):
            resolved = os.path.join(os.path.dirname(RUNS_ROOT), out_dir)
        used = 0
        if resolved and os.path.isdir(resolved):
            for f in os.listdir(resolved):
                try:
                    # st_blocks*512, NOT getsize: the builder pre-allocates
                    # sparse memmaps at full size, so apparent size reads ~100%
                    # from the first second. Allocated blocks track real progress.
                    st = os.stat(os.path.join(resolved, f))
                    used += st.st_blocks * 512
                except OSError:
                    pass
        cfg = None
        for i, t in enumerate(toks):
            if t == "--config" and i + 1 < len(toks):
                cfg = toks[i + 1]
        # 180k samples x ~336 KB; the builder's own sizing, see
        # reports/kd_feature_multitask/plan_cached_teacher.md
        expected = 60.5 * 1024 ** 3
        out.append({
            "kind": "teacher_cache",
            "pid": int(pid),
            "out_dir": out_dir,
            "config": cfg,
            "bytes": used,
            "expected_bytes": int(expected),
            "pct": round(min(100.0, used / expected * 100.0), 1),
            "log_tail": _tail("/tmp/cache_build.log", 6),
        })
    return out


#: A run whose train.log moved this recently is shown regardless of age.
ACTIVE_WINDOW_S = 6 * 3600
#: Beyond the active ones, show at most this many recent runs.
RECENT_LIMIT = 8


def _discover_arms() -> list[str]:
    """Find "out_root/ARM" pairs under RUNS_ROOT, newest first.

    Exists so a newly launched sequence shows up without anyone editing the
    server. Bounded by ACTIVE_WINDOW_S / RECENT_LIMIT so old history does not
    crowd out what is actually running.
    """
    found = []
    if not os.path.isdir(RUNS_ROOT):
        return []
    for out_root in os.listdir(RUNS_ROOT):
        p_out = os.path.join(RUNS_ROOT, out_root)
        if not os.path.isdir(p_out):
            continue
        for arm in os.listdir(p_out):
            p_arm = os.path.join(p_out, arm)
            if not os.path.isdir(p_arm):
                continue
            runs = glob.glob(os.path.join(p_arm, f"{arm}_seed*_*"))
            if not runs:
                continue
            newest = max(runs, key=lambda d: os.path.getmtime(d))
            log = os.path.join(newest, "train.log")
            mt = os.path.getmtime(log) if os.path.exists(log) else os.path.getmtime(newest)
            found.append((mt, f"{out_root}/{arm}"))
    found.sort(reverse=True)
    now = time.time()
    active = [name for mt, name in found if now - mt < ACTIVE_WINDOW_S]
    recent = [name for _, name in found[:RECENT_LIMIT]]
    out, seen = [], set()
    for name in active + recent:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def _is_training(out_root: str, arm: str) -> bool:
    """Is a trainer process actually running for this run right now?

    Log-mtime staleness alone marks a killed run as "training" until the window
    expires -- an hour of the board saying something is running when it is not.
    """
    try:
        ps = subprocess.run(["ps", "-eo", "args="],
                            capture_output=True, text=True, timeout=8)
    except Exception:
        return False
    needle_arm, needle_root = f"--arm {arm}", f"--out-root {os.path.join('runs', out_root)}"
    for line in ps.stdout.splitlines():
        if "src.train.train" in line and needle_arm in line:
            if needle_root in line or f"--out-root runs/{out_root}" in line:
                return True
    return False


def _run_regime(run_dir: str) -> dict:
    """Which evaluation split this run was scored on, read from ITS OWN config.

    Hard-coding this per arm on the page meant a new arm was unlabelled and a
    wrong entry would mislabel real data. The run stores the config it ran with,
    so the label cannot drift from the truth.
    """
    cfg_p = os.path.join(run_dir, "config.yaml")
    if not os.path.exists(cfg_p):
        return {"regime": None, "val_tasks": None}
    try:
        import yaml
        cfg = yaml.safe_load(open(cfg_p)) or {}
    except Exception:
        return {"regime": None, "val_tasks": None}
    vt = ((cfg.get("eval") or {}).get("val_tasks")) or {}
    blob = " ".join(str(v) for v in vt.values()).lower()
    if not blob:
        regime = None
    elif "demo" in blob:
        regime = "leaked"
    else:
        regime = "clean"
    return {"regime": regime, "val_tasks": vt or None}


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
        "out_root": out_root,
        "run_dir": os.path.basename(run_dir),
        "alive": _is_training(out_root, arm),
        **_run_regime(run_dir),
        "latest": last,
        "history": history_trimmed,
        "log_tail": _tail(log_path, LOG_TAIL_LINES),
        "stale": stale,
        "last_update_s_ago": (time.time() - log_mtime) if log_mtime else None,
    }


def main() -> None:
    _names, _seen = [], set()
    for _a in list(ARMS) + _discover_arms():   # pinned first, then found
        if _a not in _seen:
            _seen.add(_a)
            _names.append(_a)
    arms_out = [_arm_status(*a.split("/", 1)) for a in _names if "/" in a]
    for spec in ARMS:
        if "/" not in spec:
            continue
        out_root, arm = spec.split("/", 1)
        arms_out.append(_arm_status(out_root, arm))

    print(json.dumps({
        "hostname": os.uname().nodename if hasattr(os, "uname") else "unknown",
        "timestamp": time.time(),
        "gpus": _gpu_status(),
        "jobs": _cache_jobs(),
        "arms": arms_out,
    }))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa -- must always emit valid JSON, even on error
        print(json.dumps({"error": f"{type(e).__name__}: {e}"}))
        sys.exit(0)
