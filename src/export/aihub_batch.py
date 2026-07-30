"""Resumable, parallel Qualcomm AI Hub batch pipeline.

Submitting 12 architectures serially (quantize -> compile -> profile, blocking on
each) takes hours. This module submits every model at each stage, then polls,
so the whole sweep runs concurrently on AI Hub.

State lives in a JSON manifest keyed by model name, so an interrupted run
resumes without resubmitting completed jobs (rule 5: assume any run can be
killed at any time).

    python -m src.export.aihub_batch --submit    # export + submit/advance all
    python -m src.export.aihub_batch --status    # print manifest state
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

#: Only these two AI Hub states are terminal. Everything else (CREATED,
#: OPTIMIZING_MODEL, PROVISIONING_DEVICE, MEASURING_PERFORMANCE,
#: RUNNING_INFERENCE, QUANTIZING_MODEL, LINKING_MODELS, ...) means "still
#: working". Enumerating the pending states instead caused an in-progress job
#: (QUANTIZING_MODEL) to be misreported as a hard failure, so this is expressed
#: as a denylist: unknown/new states are treated as pending, never as errors.
_TERMINAL_OK = "SUCCESS"
_TERMINAL_FAIL = "FAILED"
_STAGES = ("quantize", "compile", "profile")


def _is_pending(code: str) -> bool:
    """True while a job is still working (any non-terminal state)."""
    return code not in (_TERMINAL_OK, _TERMINAL_FAIL)


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Load the batch manifest, or return an empty one."""
    p = Path(path)
    if not p.exists():
        return {"models": {}}
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    """Persist the manifest atomically enough for interactive use."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)


def _entry(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    return manifest["models"].setdefault(
        name, {"jobs": {}, "status": {}, "results": {}, "error": None})


def job_state(job_id: str) -> str:
    """Return the current status code for a job id."""
    import qai_hub as hub

    return hub.get_job(job_id).get_status().code


def advance(manifest: dict[str, Any], name: str, onnx_path: Path, device: str,
            input_shape: tuple[int, ...], *, calib_samples: int,
            compile_options: str, profile_options: str, seed: int = 0) -> str:
    """Move one model to its next AI Hub stage; return a short state string.

    Idempotent: already-submitted stages are polled rather than resubmitted, so
    calling this repeatedly drives the pipeline forward safely.
    """
    import onnx
    import qai_hub as hub

    from src.export.aihub import make_calibration_data

    e = _entry(manifest, name)
    target = hub.Device(device)

    for stage in _STAGES:
        jid = e["jobs"].get(stage)
        if jid:
            code = job_state(jid)
            e["status"][stage] = code
            if _is_pending(code):
                return f"{stage}:{code}"
            if code == _TERMINAL_FAIL:
                st = hub.get_job(jid).get_status()
                e["error"] = f"{stage} {code}: {(st.message or '')[:400]}"
                return f"{stage}:{code}"
            continue  # SUCCESS -> fall through to next stage

        # stage not yet submitted
        if stage == "quantize":
            input_name = onnx.load(str(onnx_path)).graph.input[0].name
            job = hub.submit_quantize_job(
                model=str(onnx_path),
                calibration_data=make_calibration_data(
                    input_name, tuple(input_shape), calib_samples, seed),
                weights_dtype=hub.QuantizeDtype.INT8,
                activations_dtype=hub.QuantizeDtype.INT8,
                name=f"{name}-quantize")
        elif stage == "compile":
            src = hub.get_job(e["jobs"]["quantize"]).get_target_model()
            job = hub.submit_compile_job(
                model=src, device=target, options=compile_options,
                name=f"{name}-compile")
        else:
            src = hub.get_job(e["jobs"]["compile"]).get_target_model()
            job = hub.submit_profile_job(
                model=src, device=target, options=profile_options,
                name=f"{name}-profile")

        e["jobs"][stage] = job.job_id
        e["status"][stage] = "CREATED"
        e["urls"] = e.get("urls", {})
        e["urls"][stage] = getattr(job, "url", "") or ""
        return f"{stage}:SUBMITTED"

    # all three stages succeeded -> harvest results once
    if not e["results"]:
        from src.export.aihub import (_extract_compute_units, _extract_latency_ms,
                                     _extract_peak_memory_mb)

        prof = hub.get_job(e["jobs"]["profile"]).download_profile()
        e["results"] = {
            "latency_ms": _extract_latency_ms(prof),
            "peak_memory_mb": _extract_peak_memory_mb(prof),
            "compute_units": _extract_compute_units(prof),
        }
    return "done"


def run_batch(specs: list[dict[str, Any]], manifest_path: str | Path, device: str,
              input_shape: tuple[int, ...], *, calib_samples: int,
              compile_options: str, profile_options: str,
              poll_seconds: int = 30, max_minutes: int = 180) -> dict[str, Any]:
    """Drive every spec through all stages concurrently until done or timeout.

    Args:
        specs: dicts with ``name`` and ``onnx`` (path) keys.
        manifest_path: JSON file used for resumability.
    """
    manifest = load_manifest(manifest_path)
    deadline = time.time() + max_minutes * 60

    while time.time() < deadline:
        states = {}
        for spec in specs:
            name = spec["name"]
            e = _entry(manifest, name)
            if e.get("error") or (e.get("results") and e["results"]):
                states[name] = "done" if not e.get("error") else "error"
                continue
            try:
                states[name] = advance(
                    manifest, name, Path(spec["onnx"]), device, input_shape,
                    calib_samples=calib_samples,
                    compile_options=compile_options,
                    profile_options=profile_options)
            except Exception as exc:  # noqa: BLE001
                e["error"] = f"submit failed: {exc}"
                states[name] = "error"
        save_manifest(manifest_path, manifest)

        finished = sum(1 for s in states.values() if s in ("done", "error"))
        print(f"[batch] {finished}/{len(specs)} settled | "
              + " ".join(f"{n}={s}" for n, s in sorted(states.items())))
        if finished == len(specs):
            break
        time.sleep(poll_seconds)

    save_manifest(manifest_path, manifest)
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="AI Hub batch pipeline.")
    ap.add_argument("--manifest", default="runs/sweep/aihub_manifest.json")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    if args.status:
        for name, e in sorted(manifest.get("models", {}).items()):
            res = e.get("results") or {}
            print(f"{name:14s} {e.get('status')} "
                  f"latency={res.get('latency_ms')} err={e.get('error')}")
        return
    ap.error("pass --status; submission is driven by src.models.student_sweep")


if __name__ == "__main__":
    main()
