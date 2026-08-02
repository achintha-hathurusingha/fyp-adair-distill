"""Assemble every metric this project has produced into one bundle.

The numbers live in many places — AI Hub job files, per-run metrics.json,
training histories on two machines, findings tables. This gathers them into
runs/demo_nb/all_metrics.json so the notebook has a single source and nothing
has to be retyped (retyping is how a report drifts from its evidence).

    python scripts/collect_all_metrics.py
"""
from __future__ import annotations

import json
from pathlib import Path

from src.utils.config import REPO_ROOT

OUT = REPO_ROOT / "runs" / "demo_nb"
DEVON = REPO_ROOT / "runs_devon" / "runs"


def _j(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def norm_ablation() -> list[dict]:
    """Task 1.5b: the normalization quality ladder (S arm, w16_b8)."""
    out = []
    for d in sorted((REPO_ROOT / "runs" / "1p5b").glob("*/")):
        m = next(d.rglob("metrics.json"), None)
        if not m:
            continue
        j = _j(m) or {}
        out.append({"arm": d.name, "best_psnr": j.get("best_psnr"),
                    "iterations": j.get("iterations"),
                    "diverged": bool(j.get("diverged")),
                    "description": j.get("description", "")})
    return out


def aihub_latency() -> dict:
    """Every on-device latency measured, by normalization variant."""
    jobs = (_j(REPO_ROOT / "reports" / "aihub_jobs.json") or {}).get("jobs", {})
    by_variant: dict[str, dict[str, float]] = {}
    for e in jobs.values():
        v, cfg = e.get("variant"), e.get("config")
        ms = (e.get("results") or {}).get("latency_ms")
        if v and cfg and ms:
            by_variant.setdefault(v, {})[cfg] = ms
    na = (_j(REPO_ROOT / "runs" / "sweep" / "aihub_manifest.json") or {}).get("models", {})
    by_variant["NA"] = {k: (v.get("results") or {}).get("latency_ms")
                        for k, v in na.items()
                        if (v.get("results") or {}).get("latency_ms")}
    fix = (_j(REPO_ROOT / "reports" / "aihub_fix_jobs.json") or {}).get("jobs", {})
    by_variant["fix_candidates"] = {k: (v.get("results") or {}).get("latency_ms")
                                    for k, v in fix.items() if v.get("results")}
    return by_variant


def training_curves() -> dict:
    """Validation histories for every long run, both machines."""
    runs = {
        "B0_seed0_final": DEVON / "b0_final/B0/B0_seed0_20260802_021452/history.json",
        "B0_diverged_NF": DEVON / "b0/B0/B0_seed0_20260801_150903/history.json",
        "QA_control_300k": next((DEVON / "qa_control").rglob("history.json"), None),
        "FixC_validation": next((DEVON / "fixc").rglob("history.json"), None),
    }
    out = {}
    for name, p in runs.items():
        if p and Path(p).exists():
            h = _j(Path(p)) or []
            out[name] = [{k: r.get(k) for k in
                          ("iteration", "loss", "psnr", "ssim", "max_grad_norm",
                           "clip_rate", "nonfinite_skips", "clamp_engage_rate",
                           "clamp_max_preclamp")} for r in h]
    for arm in ("Q-A", "Q-F", "Q-E", "Q-E1", "Q-E2", "Q-E3", "M-A", "M-F"):
        p = next((REPO_ROOT / "runs" / "1p5b" / arm).rglob("history.json"), None)
        if p:
            h = _j(p) or []
            out[f"1p5b_{arm}"] = [{k: r.get(k) for k in
                                   ("iteration", "loss", "psnr", "max_grad_norm")}
                                  for r in h]
    return out


#: Measured results that live only in findings.md / commit history, transcribed
#: here ONCE so the notebook never retypes them. Each carries its source.
LITERALS = {
    "f9_stage_activations": {
        "source": "findings F9, captured spike step 24356, pathological sample 12",
        "stages": ["enc0", "enc1", "enc2", "enc3", "middle",
                   "dec0", "dec1", "dec2", "dec3"],
        "mean_abs": [0.0595, 0.0731, 0.1545, 0.3591, 23.4667,
                     16.4296, 8.1680, 4.8449, 2821.8494],
        "max_abs": [0.50, 0.76, 1.69, 7.59, 970.42,
                    249.87, 68.11, 28.03, 5588974.50],
    },
    "f9_fix_containment": {
        "source": "findings F9, same spike-state weights loaded into each variant",
        "variant": ["N-F (affine)", "Fix-B (LayerNorm)", "Fix-C clamp 64",
                    "Fix-C clamp 16", "Fix-C clamp 8", "Fix-C clamp 4",
                    "Fix-C clamp 2"],
        "sample12_max_out": [705100.0, 15.71, 667.5, 55.67, 17.49, 6.502, 2.572],
        "healthy_max_out": [1.048, 11.69, 1.048, 1.048, 1.048, 1.048, 1.048],
    },
    "f9_agc_gradient_concentration": {
        "source": "findings F9, per-stage grad-norm/param-norm ratio, UNCLAMPED model",
        "stage": ["intro", "encoders.0", "encoders.1", "encoders.2", "encoders.3",
                  "middle_blks", "decoders.0", "decoders.1", "decoders.2",
                  "decoders.3", "ending"],
        "healthy": [5.086e-3, 1.254e-4, 2.045e-4, 5.199e-5, 1.483e-5, 3.376e-6,
                    2.487e-5, 4.799e-5, 1.553e-4, 2.937e-4, 3.69e-2],
        "pathological": [9.118e6, 2.414e5, 5.871e5, 1.583e5, 3.969e4, 1.416e4,
                         117.5, 521.4, 2863.0, 2.301e4, 1.319e4],
    },
    "f9_clamp_blocks_backward": {
        "source": "findings F9, same weights and sample, clamp off vs on",
        "stage": ["intro", "encoders.1", "middle_blks", "decoders.3", "ending", "TOTAL"],
        "unclamped": [2.712e7, 1.315e7, 3.794e6, 4.307e5, 1.476e4, 3.851e7],
        "clamped": [1216.0, 590.7, 164.8, 21.0, 6.482, 1728.0],
    },
    "f9_mann_kendall": {
        "source": "pre-committed trend test, 8 clamp intervals, alpha=0.05",
        "series": ["log(premax)", "clamp_engage_rate"],
        "n": [8, 8], "S": [4, 10], "tau": [0.143, 0.357],
        "Z": [0.371, 1.113], "p": [0.7105, 0.2655],
        "verdict": ["no significant trend", "no significant trend"],
    },
    "f7_teacher_export": {
        "source": "findings F7 — AdaIR export attempts, all failed",
        "attempt": ["unpatched / TorchScript / 17", "patched / TorchScript / 17",
                    "patched / TorchScript / 20", "unpatched / dynamo / 18",
                    "unpatched / dynamo / 20"],
        "result": ["SymbolicValueError", "UnsupportedOperatorError: aten::fft_fft2",
                   "UnsupportedOperatorError: aten::fft_fft2",
                   "TorchExportError", "TorchExportError"],
    },
    "f1_norm_cycle_share": {
        "source": "findings F1 — Snapdragon 8 Gen 3 Hexagon v75 INT8 cycle attribution",
        "op": ["LayerNorm2d", "Conv"], "npu_cycle_pct": [62.0, 3.4],
    },
    "student_grid": {
        "source": "reports/student_sweep.md — the 12 candidate students profiled",
        "config":   ["w16_b8", "w16_b14", "w16_b28", "w16_sidd",
                     "w24_b8", "w24_b14", "w24_b28", "w24_sidd",
                     "w32_b8", "w32_b14", "w32_b28", "w32_sidd"],
        "params_m": [2.44, 3.15, 4.35, 7.37,
                     5.44, 7.02, 9.68, 16.46,
                     9.62, 12.42, 17.11, 29.16],
        "gmacs":    [2.13, 2.74, 4.09, 4.13,
                     4.67, 6.05, 9.05, 9.11,
                     8.21, 10.66, 15.96, 16.05],
        "selected": {"w16_b8": "S", "w16_sidd": "M", "w24_b28": "L"},
        "ceiling_m": 10.0,
    },
    "family_locked": {
        "source": "reports/student_sweep.md, re-selected on measured Fix-C latency",
        "arm": ["S", "M", "L"],
        "config": ["w16_b8", "w16_sidd", "w24_b28"],
        "params_m": [2.44, 7.37, 9.68], "gmacs": [2.13, 4.13, 9.05],
        "ms_fixc": [1.577, 2.885, 3.328],
    },
    "b0_divergence_events": {
        "source": "findings F9 — where plain N-F died",
        "run": ["original (continuous)", "rA (resume, clip 8.0)",
                "rB (resume, clip 8.0)", "retry (resume, clip 1.0)"],
        "diverged_at_step": [21967, 25582, 25582, 28654],
        "maxgn_at_25k": [None, 65240022.433, 65240022.433, 65240022.433],
    },
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    bundle = {
        "norm_ablation_1p5b": norm_ablation(),
        "aihub_latency": aihub_latency(),
        "training_curves": training_curves(),
        "int8_demo_b0": _j(REPO_ROOT / "runs/int8_demo/int8_results.json"),
        "int8_demo_adair": _j(REPO_ROOT / "runs/int8_demo/adair.json"),
        "fp32_timing": _j(REPO_ROOT / "runs/int8_demo/timing.json"),
        "aihub_jobs_int8_demo": _j(REPO_ROOT / "runs/int8_demo/jobs.json"),
        "literals": LITERALS,
    }
    p = OUT / "all_metrics.json"
    p.write_text(json.dumps(bundle, indent=1, default=str), encoding="utf-8")

    print(f"wrote {p}  ({p.stat().st_size/1024:.0f} KB)")
    print(f"  norm ablation arms : {len(bundle['norm_ablation_1p5b'])}")
    for v, d in bundle["aihub_latency"].items():
        print(f"  aihub {v:<15}: {len(d)} configs")
    for k, v in bundle["training_curves"].items():
        print(f"  curve {k:<18}: {len(v)} validation points")
    print(f"  literal tables     : {len(LITERALS)}")


if __name__ == "__main__":
    main()
