"""Run the real-world demo images through the Qualcomm INT8 path on a Galaxy S24.

The earlier INT8 measurement covered 12 BSD68 crops at 256x256. This extends it
to the 10 real-world photographs at their native 512x512, which needs a SEPARATE
export and compile: a QNN context binary is built for one fixed input shape, so
the 256x256 binary cannot serve 512x512.

    python scripts/int8_real_world.py submit
    python scripts/int8_real_world.py collect

Scope note: only the SYNTHETIC regime is measured on-device. The native-JPEG
images have no ground truth, so an INT8 PSNR for them would be meaningless —
they are shown visually in the notebook instead.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.eval.metrics import ADAIR_DEFAULT, psnr_ssim
from src.models.nafnet import NAFNet
from src.utils.config import REPO_ROOT

OUT = REPO_ROOT / "runs" / "demo_nb"
REAL = OUT / "real"
SHAPE = (1, 3, 512, 512)
SIGMAS = (15, 25, 50)
DEVICE_NAME = "Samsung Galaxy S24 (Family)"
JOBS = OUT / "int8_real_jobs.json"


def _pairs() -> list[tuple[str, int]]:
    stems = sorted({p.name.split("_s")[0] for p in REAL.glob("*_s*_noisy.npy")})
    return [(s, sg) for sg in SIGMAS for s in stems]


def _inputs() -> np.ndarray:
    arrs = []
    for stem, sg in _pairs():
        a = np.load(REAL / f"{stem}_s{sg}_noisy.npy").astype(np.float32) / 255.0
        arrs.append(a.transpose(2, 0, 1)[None])
    return np.concatenate(arrs, 0)


def submit() -> None:
    import onnx
    import qai_hub as hub
    from src.export.to_onnx import export_onnx

    state = json.loads(JOBS.read_text()) if JOBS.exists() else {}
    onnx_path = OUT / "b0_seed0_512.onnx"
    if not onnx_path.exists():
        ck = torch.load(REPO_ROOT / "runs/int8_demo/b0_seed0.pth",
                        map_location="cpu", weights_only=False)
        m = NAFNet(**ck["config"]["model"])
        m.load_state_dict(ck["model"])
        export_onnx(m.eval(), onnx_path, SHAPE)
        print(f"exported {onnx_path.name} at {SHAPE}")

    x = _inputs()
    print(f"{len(x)} inputs of {x.shape[1:]}")
    name = onnx.load(str(onnx_path)).graph.input[0].name

    if "quantize" not in state:
        j = hub.submit_quantize_job(
            model=str(onnx_path),
            calibration_data={name: [x[i:i + 1] for i in range(0, len(x), 3)]},
            weights_dtype=hub.QuantizeDtype.INT8,
            activations_dtype=hub.QuantizeDtype.INT8,
            name="b0-real-512-quantize")
        state["quantize"] = j.job_id
        JOBS.write_text(json.dumps(state, indent=2))
        print(f"quantize {j.job_id}")
    q = hub.get_job(state["quantize"])
    if not q.wait().success:
        raise SystemExit("quantize failed")

    if "compile" not in state:
        c = hub.submit_compile_job(model=q.get_target_model(),
                                   device=hub.Device(DEVICE_NAME),
                                   options="--target_runtime qnn_context_binary",
                                   name="b0-real-512-compile")
        state["compile"] = c.job_id
        JOBS.write_text(json.dumps(state, indent=2))
        print(f"compile {c.job_id}")
    c = hub.get_job(state["compile"])
    if not c.wait().success:
        raise SystemExit("compile failed")

    if "profile" not in state:
        p = hub.submit_profile_job(model=c.get_target_model(),
                                   device=hub.Device(DEVICE_NAME),
                                   name="b0-real-512-profile")
        state["profile"] = p.job_id
        JOBS.write_text(json.dumps(state, indent=2))
        print(f"profile {p.job_id}")

    if "inference" not in state:
        i = hub.submit_inference_job(model=c.get_target_model(),
                                     device=hub.Device(DEVICE_NAME),
                                     inputs={name: [x[k:k + 1] for k in range(len(x))]},
                                     name="b0-real-512-inference")
        state["inference"] = i.job_id
        JOBS.write_text(json.dumps(state, indent=2))
        print(f"inference {i.job_id}")
    print(json.dumps(state, indent=2))


def collect() -> None:
    import qai_hub as hub
    from src.export.aihub import _extract_latency_ms

    state = json.loads(JOBS.read_text())
    job = hub.get_job(state["inference"])
    if not job.wait().success:
        raise SystemExit("inference failed")
    out = job.download_output_data()
    preds = out[next(iter(out))]

    rows = []
    for k, (stem, sg) in enumerate(_pairs()):
        clean = np.load(REAL / f"{stem}_clean.npy").astype(np.float32) / 255.0
        arr = np.asarray(preds[k])
        arr = arr[0] if arr.ndim == 4 else arr
        pred = np.clip(arr.transpose(1, 2, 0) if arr.shape[0] == 3 else arr, 0, 1)
        p, s = psnr_ssim(pred, clean, ADAIR_DEFAULT)
        np.save(REAL / f"{stem}_s{sg}_int8.npy", (pred * 255).round().astype(np.uint8))
        rows.append({"image": stem, "sigma": sg, "int8_psnr": p, "int8_ssim": s})

    if "profile" in state:
        pj = hub.get_job(state["profile"])
        if pj.wait().success:
            ms = _extract_latency_ms(pj.download_profile())
            rows.append({"_latency_ms_512": ms})
            print(f"on-device latency at 512x512: {ms:.3f} ms")

    (OUT / "int8_real.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    rw = json.loads((OUT / "real_world.json").read_text())
    print(f"\n{'sigma':>6}{'B0 FP32':>10}{'B0 INT8':>10}{'delta':>9}")
    for sg in SIGMAS:
        f = [r for r in rw if r["regime"] == "synthetic" and r["sigma"] == sg]
        q = [r for r in rows if r.get("sigma") == sg]
        mf = np.mean([r["b0_psnr"] for r in f]); mq = np.mean([r["int8_psnr"] for r in q])
        print(f"{sg:>6}{mf:>10.3f}{mq:>10.3f}{mq-mf:>+9.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["submit", "collect"])
    a = ap.parse_args()
    (submit if a.stage == "submit" else collect)()
