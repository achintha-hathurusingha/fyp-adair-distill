"""M-DEHAZE-KD-FREQ live dashboard — stdlib-only HTTP server.

No pip dependencies on purpose: this runs identically inside the Docker
image and directly on the host while Docker is being set up. Reads run
state straight off disk (history.json per run dir + the combined launch
log) — never touches the training process itself.

Data sources (all read-only):
  RUNS_ROOT   ~/fyp-adair-distill/runs           — per-arm/per-seed history.json
  LAUNCH_LOG  /tmp/kd_freq_3seed.log             — SEED START/DONE + crash signatures

Served:
  GET /            -> index.html (single page)
  GET /api/status  -> JSON snapshot, polled by the page every few seconds
"""
from __future__ import annotations

import glob
import json
import os
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

RUNS_ROOT = Path(os.environ.get("RUNS_ROOT", "/data/runs"))
LAUNCH_LOG = Path(os.environ.get("LAUNCH_LOG", "/data/logs/kd_freq_3seed.log"))
LAUNCH_LOG_FEAT = Path(os.environ.get("LAUNCH_LOG_FEAT", "/data/logs/kd_feat_3seed.log"))
TEACHER_PSNR = 34.5056  # AdaIR teacher, dehaze, frozen — reports/report_demo_dehaze.md
TARGET_ITERS = 60000
PORT = int(os.environ.get("PORT", "8080"))

STATIC_DIR = Path(__file__).parent


def _latest_run_dir(arm_exact: str, seed: int) -> Path | None:
    """Most recently modified run dir for an exact arm name + seed, across any
    runs/<group>/<arm>/ prefix (the repo's run-root naming isn't consistent
    across experiments, so this globs rather than assuming one prefix)."""
    pattern = str(RUNS_ROOT / "*" / arm_exact / f"{arm_exact}_seed{seed}_*")
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


def _read_total_iters(run_dir: Path) -> int:
    cfg = run_dir / "config.yaml"
    if not cfg.exists():
        return TARGET_ITERS
    try:
        for line in cfg.read_text().splitlines():
            m = re.match(r"\s*total_iters:\s*(\d+)", line)
            if m:
                return int(m.group(1))
    except OSError:
        pass
    return TARGET_ITERS


def _seed_snapshot(arm_exact: str, seed: int) -> dict:
    run_dir = _latest_run_dir(arm_exact, seed)
    if run_dir is None:
        return {"seed": seed, "status": "pending", "iteration": 0,
                "target_iters": TARGET_ITERS, "history": []}

    history = _read_history(run_dir)
    total_iters = _read_total_iters(run_dir)
    last = history[-1] if history else None
    iteration = last["iteration"] if last else 0
    completed_marker = (run_dir / "last.pth").exists() and iteration >= total_iters

    # "running" vs "stalled": history file mtime within the last checkpoint
    # interval, with margin. Checkpoints land every val_every=2000 iters,
    # which has run ~16 min/2000 iters in practice (slower with two
    # concurrent experiments sharing the GPU/CPU) — a threshold shorter than
    # that reports "stalled" every cycle right before the next checkpoint
    # lands, which is exactly what happened at 600s (caught live: kd_freq
    # showed "stalled" mid-training with nothing actually wrong). 1800s
    # covers the observed interval with real margin.
    hf = run_dir / "history.json"
    fresh = hf.exists() and (time.time() - hf.stat().st_mtime) < 1800

    # Before the FIRST checkpoint, history.json doesn't exist yet at all --
    # without this, a run that has genuinely started shows "pending" (=
    # "not started") for its first ~16 minutes. config.yaml is written at
    # process start, before any training iterations, so its recency is a
    # real "the process is alive" signal in that window.
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


def _arm_summary(arm_exact: str, seeds=(0, 1, 2)) -> dict:
    seed_snaps = [_seed_snapshot(arm_exact, s) for s in seeds]
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


def build_status() -> dict:
    baseline = _arm_summary("M-DEHAZE")
    response_kd = _arm_summary("M-DEHAZE-KD")
    freq_kd = _arm_summary("M-DEHAZE-KD-FREQ")
    feat_kd = _arm_summary("M-DEHAZE-KD-FEAT")

    log_tail = _tail_log(LAUNCH_LOG)
    log_tail_feat = _tail_log(LAUNCH_LOG_FEAT)
    crash_pattern = r"Traceback|CUDA out of memory|nonfinite|NaN\b|Error"
    crash = any(re.search(crash_pattern, ln) for ln in log_tail)
    crash_feat = any(re.search(crash_pattern, ln) for ln in log_tail_feat)

    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "teacher_psnr": TEACHER_PSNR,
        "crash_detected": crash,
        "crash_detected_feat": crash_feat,
        "log_tail": log_tail,
        "log_tail_feat": log_tail_feat,
        "arms": {
            "baseline": baseline,
            "response_kd": response_kd,
            "freq_kd": freq_kd,
            "feat_kd": feat_kd,
        },
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quiet the default stderr access log
        pass

    def do_GET(self):
        if self.path.startswith("/api/status"):
            body = json.dumps(build_status()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path in ("/", "/index.html"):
            fp = STATIC_DIR / "index.html"
            body = fp.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            # No caching — the page's own JS changes between edits with no
            # versioned filename, so a cached copy silently keeps running old
            # logic after a fix is deployed (this bit us: the axis-stability
            # fix was live server-side but the browser kept the old page).
            self.send_header("Cache-Control", "no-store, must-revalidate")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_response(404)
        self.end_headers()


def main():
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"kd_freq dashboard on :{PORT}  runs_root={RUNS_ROOT}  launch_log={LAUNCH_LOG}")
    server.serve_forever()


if __name__ == "__main__":
    main()
