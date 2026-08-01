"""Measure the on-device latency cost of the F9 fix candidates.

Three variants of the LOCKED M arm (`w16_sidd`), Samsung Galaxy S24, INT8:

    N-F        affine at full resolution            — current lock, 1.59x
    Fix-B      LayerNorm2d at full resolution       — restores the reductions
    Fix-C(8)   affine + Clip(+-8) at full resolution — one elementwise op

The question is which fix keeps the most of N-F's speedup while passing the
adversarial stress suite (`scripts/stress_test_norm.py`).

The ONNX op counts already say what to expect — Fix-C adds 8 `Clip` nodes and
leaves `Div`/`Sqrt`/`ReduceMean` at exactly N-F's counts, while Fix-B reinstates
the reductions N-F removed — but F3 and F4 are this project's standing warning
that op counts do not predict fused-kernel latency. `Clip` is expected to fuse
into the preceding op and cost ~nothing; that is a hypothesis to measure.

    python -m scripts.submit_fix_latency            # submit
    python -m scripts.submit_fix_latency --collect  # harvest
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.export.to_onnx import export_onnx
from src.models.nafnet import NAFNet
from src.utils.config import REPO_ROOT

#: The locked M arm — the config B0 trains and the one that diverged.
GEOMETRY = dict(width=16, enc_blk_nums=[2, 2, 4, 8], middle_blk_num=12,
                dec_blk_nums=[2, 2, 2, 2])

VARIANTS = {
    "M_NF": dict(norm_type="layernorm2d", full_res_norm_type="affine"),
    "M_FixB": dict(norm_type="layernorm2d", full_res_norm_type=None),
    "M_FixC8": dict(norm_type="layernorm2d", full_res_norm_type="affine_clamp",
                    clamp_bound=8.0),
}

DEVICE = "Samsung Galaxy S24 (Family)"
COMPILE_OPTS = "--target_runtime qnn_context_binary"
SHAPE = (1, 3, 256, 256)
JOBS = REPO_ROOT / "reports" / "aihub_fix_jobs.json"


def _load() -> dict:
    return json.loads(JOBS.read_text(encoding="utf-8")) if JOBS.exists() else {"jobs": {}}


def _save(d: dict) -> None:
    JOBS.parent.mkdir(parents=True, exist_ok=True)
    JOBS.write_text(json.dumps(d, indent=2, sort_keys=True), encoding="utf-8")


def submit() -> None:
    import onnx as onnx_mod
    import qai_hub as hub

    from src.export.aihub import make_calibration_data

    onnx_dir = REPO_ROOT / "runs" / "export" / "fix_latency"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    state = _load()

    for name, norm_kw in VARIANTS.items():
        if state["jobs"].get(name, {}).get("quantize"):
            print(f"[submit] skip {name} (already submitted)")
            continue
        path = onnx_dir / f"{name}.onnx"
        if not path.exists():
            export_onnx(NAFNet(**GEOMETRY, **norm_kw).eval(), path, SHAPE)
        input_name = onnx_mod.load(str(path)).graph.input[0].name
        for attempt in range(4):
            try:
                job = hub.submit_quantize_job(
                    model=str(path),
                    calibration_data=make_calibration_data(input_name, SHAPE, 8, 0),
                    weights_dtype=hub.QuantizeDtype.INT8,
                    activations_dtype=hub.QuantizeDtype.INT8,
                    name=f"{name}-quantize")
                state["jobs"].setdefault(name, {})["quantize"] = job.job_id
                state["jobs"][name]["onnx"] = str(path)
                _save(state)                       # persist immediately
                print(f"[submit] {name:10s} quantize={job.job_id}")
                break
            except Exception as exc:               # noqa: BLE001
                wait = 2 ** attempt * 15
                print(f"[submit] {name} attempt {attempt+1}: {str(exc)[:100]}")
                if attempt == 3:
                    state["jobs"].setdefault(name, {})["error"] = str(exc)[:300]
                    _save(state)
                else:
                    time.sleep(wait)


def collect() -> None:
    import qai_hub as hub

    from src.export.aihub import _extract_latency_ms, _extract_peak_memory_mb

    state = _load()
    for name, e in sorted(state["jobs"].items()):
        if e.get("results") or e.get("error"):
            continue
        try:
            q = hub.get_job(e["quantize"])
            if not q.wait().success:
                e["error"] = "quantize failed"
                _save(state)
                continue
            if not e.get("compile"):
                c = hub.submit_compile_job(model=q.get_target_model(),
                                           device=hub.Device(DEVICE),
                                           options=COMPILE_OPTS,
                                           name=f"{name}-compile")
                e["compile"] = c.job_id
                _save(state)
            c = hub.get_job(e["compile"])
            if not c.wait().success:
                e["error"] = "compile failed"
                _save(state)
                continue
            if not e.get("profile"):
                p = hub.submit_profile_job(model=c.get_target_model(),
                                           device=hub.Device(DEVICE),
                                           name=f"{name}-profile")
                e["profile"] = p.job_id
                _save(state)
            p = hub.get_job(e["profile"])
            if not p.wait().success:
                e["error"] = "profile failed"
                _save(state)
                continue
            prof = p.download_profile()
            e["results"] = {"latency_ms": _extract_latency_ms(prof),
                            "peak_memory_mb": _extract_peak_memory_mb(prof)}
            _save(state)
            print(f"[collect] {name:10s} {e['results']['latency_ms']:.3f} ms")
        except Exception as exc:                   # noqa: BLE001
            print(f"[collect] {name}: {str(exc)[:140]}")

    done = {n: e["results"]["latency_ms"] for n, e in state["jobs"].items()
            if e.get("results")}
    if len(done) >= 2 and "M_NF" in done:
        print(f"\n{'variant':<12}{'ms':>10}{'vs N-F':>12}")
        for n, ms in sorted(done.items(), key=lambda kv: kv[1]):
            print(f"{n:<12}{ms:>10.3f}{ms / done['M_NF']:>11.2f}x")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--collect", action="store_true")
    a = ap.parse_args()
    collect() if a.collect else submit()
