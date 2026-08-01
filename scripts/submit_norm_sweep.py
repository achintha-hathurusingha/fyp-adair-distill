"""Task 0 — submit the 24 norm-variant AI Hub jobs (12 configs x N-F, N-E).

The existing 12-config sweep already IS the N-A table, so it is not re-run.

Job IDs are persisted to ``reports/aihub_jobs.json`` **as each is submitted**,
not at the end, so an interrupted process loses nothing. Submission is batched
with exponential back-off on throttling.

    python -m scripts.submit_norm_sweep
    python -m scripts.submit_norm_sweep --collect     # harvest finished jobs
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.export.to_onnx import export_onnx
from src.models.nafnet import NAFNet
from src.models.student_sweep import build_grid
from src.utils.config import REPO_ROOT, load_yaml

#: Normalisation variants to sweep. N-A is already measured.
VARIANTS = {
    "NF": {"norm_type": "layernorm2d", "full_res_norm_type": "affine"},
    "NE": {"norm_type": "affine"},
    # FC — the LOCKED variant after findings F9. Swept across the whole grid
    # rather than extrapolated from the M arm's +0.3%: the clamp adds Clip nodes
    # in proportion to each config's full-resolution block count, so the delta
    # is not uniform and the family ranking cannot be assumed unchanged.
    "FC": {"norm_type": "layernorm2d", "full_res_norm_type": "affine_clamp",
           "clamp_bound": 8.0},
}
DEVICE = "Samsung Galaxy S24 (Family)"
COMPILE_OPTS = "--target_runtime qnn_context_binary"
SHAPE = (1, 3, 256, 256)


def _load(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"jobs": {}}


def _save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def submit_all(jobs_path: Path, onnx_dir: Path, *, max_retries: int = 5) -> dict:
    """Export and submit every variant, persisting each job ID immediately."""
    import qai_hub as hub

    from src.export.aihub import make_calibration_data

    cfg = load_yaml("configs/sweep/student_sweep.yaml")
    grid = build_grid(cfg)
    state = _load(jobs_path)

    for spec in grid:
        for tag, norm_kw in VARIANTS.items():
            name = f"{spec['name']}_{tag}"
            if name in state["jobs"] and state["jobs"][name].get("quantize"):
                print(f"[submit] skip {name} (already submitted)")
                continue

            onnx_path = onnx_dir / f"{name}.onnx"
            if not onnx_path.exists():
                model = NAFNet(width=spec["width"],
                               enc_blk_nums=spec["enc_blk_nums"],
                               middle_blk_num=spec["middle_blk_num"],
                               dec_blk_nums=spec["dec_blk_nums"], **norm_kw)
                export_onnx(model.eval(), onnx_path, SHAPE)

            import onnx as onnx_mod
            input_name = onnx_mod.load(str(onnx_path)).graph.input[0].name

            for attempt in range(max_retries):
                try:
                    job = hub.submit_quantize_job(
                        model=str(onnx_path),
                        calibration_data=make_calibration_data(
                            input_name, SHAPE, 8, 0),
                        weights_dtype=hub.QuantizeDtype.INT8,
                        activations_dtype=hub.QuantizeDtype.INT8,
                        name=f"{name}-quantize")
                    state["jobs"].setdefault(name, {})["quantize"] = job.job_id
                    state["jobs"][name]["config"] = spec["name"]
                    state["jobs"][name]["variant"] = tag
                    state["jobs"][name]["onnx"] = str(onnx_path)
                    _save(jobs_path, state)          # persist IMMEDIATELY
                    print(f"[submit] {name:20s} quantize={job.job_id}")
                    break
                except Exception as exc:  # noqa: BLE001
                    wait = 2 ** attempt * 15
                    print(f"[submit] {name} attempt {attempt+1} failed "
                          f"({str(exc)[:90]}); backing off {wait}s")
                    if attempt == max_retries - 1:
                        state["jobs"].setdefault(name, {})["error"] = str(exc)[:300]
                        _save(jobs_path, state)
                    else:
                        time.sleep(wait)
    return state


def advance_all(jobs_path: Path) -> dict:
    """Drive submitted jobs through compile and profile; harvest results."""
    import qai_hub as hub

    from src.export.aihub import (_extract_compute_units, _extract_latency_ms,
                                 _extract_peak_memory_mb)

    state = _load(jobs_path)
    for name, entry in sorted(state["jobs"].items()):
        if entry.get("results") or entry.get("error"):
            continue
        try:
            qid = entry.get("quantize")
            if not qid:
                continue
            if hub.get_job(qid).get_status().code != "SUCCESS":
                continue

            if not entry.get("compile"):
                job = hub.submit_compile_job(
                    model=hub.get_job(qid).get_target_model(),
                    device=hub.Device(DEVICE), options=COMPILE_OPTS,
                    name=f"{name}-compile")
                entry["compile"] = job.job_id
                _save(jobs_path, state)
                continue
            if hub.get_job(entry["compile"]).get_status().code != "SUCCESS":
                continue

            if not entry.get("profile"):
                job = hub.submit_profile_job(
                    model=hub.get_job(entry["compile"]).get_target_model(),
                    device=hub.Device(DEVICE), name=f"{name}-profile")
                entry["profile"] = job.job_id
                _save(jobs_path, state)
                continue
            if hub.get_job(entry["profile"]).get_status().code != "SUCCESS":
                continue

            prof = hub.get_job(entry["profile"]).download_profile()
            entry["results"] = {
                "latency_ms": _extract_latency_ms(prof),
                "peak_memory_mb": _extract_peak_memory_mb(prof),
                "compute_units": _extract_compute_units(prof),
            }
            _save(jobs_path, state)
            print(f"[collect] {name:20s} {entry['results']['latency_ms']:.3f} ms")
        except Exception as exc:  # noqa: BLE001
            print(f"[collect] {name}: {str(exc)[:120]}")
    return state


def main() -> None:
    ap = argparse.ArgumentParser(description="Norm-variant AI Hub sweep.")
    ap.add_argument("--jobs", default="reports/aihub_jobs.json")
    ap.add_argument("--onnx-dir", default="runs/norm_sweep")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--poll-minutes", type=int, default=0,
                    help="if >0, keep advancing jobs for this many minutes")
    args = ap.parse_args()

    jobs_path = REPO_ROOT / args.jobs
    onnx_dir = REPO_ROOT / args.onnx_dir
    onnx_dir.mkdir(parents=True, exist_ok=True)

    if not args.collect:
        submit_all(jobs_path, onnx_dir)

    if args.collect or args.poll_minutes:
        deadline = time.time() + max(args.poll_minutes, 1) * 60
        while True:
            state = advance_all(jobs_path)
            done = sum(1 for e in state["jobs"].values()
                       if e.get("results") or e.get("error"))
            print(f"[collect] {done}/{len(state['jobs'])} settled")
            if done >= len(state["jobs"]) or time.time() > deadline:
                break
            time.sleep(60)


if __name__ == "__main__":
    main()
