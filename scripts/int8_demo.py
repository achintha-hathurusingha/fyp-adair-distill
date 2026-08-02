"""INT8-on-device verification of a trained B0 checkpoint, against FP32.

Answers the question left open since G1: does post-training INT8 quantization
preserve B0's measured FP32 quality on REAL trained weights? Every AI Hub number
so far has been latency on untrained weights, because latency does not depend on
them — quality does.

    python scripts/int8_demo.py prepare --ckpt <pth>   # demo set + FP32 reference
    python scripts/int8_demo.py submit                 # quantize -> compile -> inference
    python scripts/int8_demo.py collect                # download, score, compare

SCOPE. B0 is a DENOISE-ONLY model — `train.py` builds its loader from
`Train/Denoise` alone and the dataset class is `DenoiseTrainDataset`. It has
never seen rain or haze, so derain/dehaze are deliberately excluded; numbers for
them would describe nothing.

METHOD. The QNN binary has a fixed input shape, so BSD68 images are centre
cropped to 256x256. The FP32 reference is computed on the **identical cropped,
noised inputs**, so the delta is attributable to quantization alone and not to
preprocessing. Metrics go through the locked harness (`ADAIR_DEFAULT`), not an
ad-hoc calculation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.data.datasets import load_rgb_uint8
from src.eval.metrics import ADAIR_DEFAULT, psnr_ssim
from src.models.nafnet import NAFNet
from src.utils.config import REPO_ROOT, load_paths

SHAPE = (1, 3, 256, 256)
SIGMAS = (15, 25, 50)
N_PER_SIGMA = 4
OUT = REPO_ROOT / "runs" / "int8_demo"
DEVICE_NAME = "Samsung Galaxy S24 (Family)"


def _bsd68_paths() -> list[Path]:
    paths = load_paths()
    root = Path(paths["data_root"])
    if not root.is_absolute():
        root = REPO_ROOT / root
    root = root / "test" / "denoise" / "bsd68"
    files = sorted(p for p in root.rglob("*")
                   if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})
    if not files:
        raise FileNotFoundError(f"no BSD68 images under {root}")
    return files


def _centre_crop(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h < size or w < size:
        raise ValueError(f"image {h}x{w} smaller than crop {size}")
    top, left = (h - size) // 2, (w - size) // 2
    return img[top:top + size, left:left + size]


def prepare(ckpt_path: Path) -> None:
    """Build the demo set and compute the FP32 reference on it."""
    OUT.mkdir(parents=True, exist_ok=True)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["config"]["model"]
    model = NAFNet(**cfg)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"checkpoint iteration {ck['iteration']}  best_psnr {ck['best_psnr']:.4f}")
    print(f"arch {cfg['norm_type']}/{cfg.get('full_res_norm_type')} "
          f"bound {cfg.get('clamp_bound')}")

    files = _bsd68_paths()
    # Widest-spread selection: sort by std so the demo set spans easy to hard
    # rather than accidentally sampling one kind of image.
    scored = []
    for f in files:
        img = load_rgb_uint8(f, base=1)
        if min(img.shape[:2]) < SHAPE[-1]:
            continue
        c = _centre_crop(img, SHAPE[-1])
        scored.append((float(c.std()), f, c))
    scored.sort(key=lambda t: t[0])
    picks = [scored[int(i * (len(scored) - 1) / (N_PER_SIGMA - 1))]
             for i in range(N_PER_SIGMA)]
    print(f"selected {len(picks)} images of {len(scored)} eligible "
          f"(std {picks[0][0]:.1f} to {picks[-1][0]:.1f})")

    inputs, meta, fp32_rows = [], [], []
    rng = np.random.RandomState(0)         # fixed: the demo must be reproducible
    for sigma in SIGMAS:
        for std, f, clean in picks:
            noisy = np.clip(clean.astype(np.float32)
                            + rng.normal(0, sigma, clean.shape), 0, 255)
            x = torch.from_numpy(noisy.transpose(2, 0, 1))[None] / 255.0
            with torch.no_grad():
                y = model(x.float())
            pred = (y[0].clamp(0, 1).numpy().transpose(1, 2, 0))
            tgt = clean.astype(np.float32) / 255.0
            p, s = psnr_ssim(pred, tgt, ADAIR_DEFAULT)
            fp32_rows.append({"image": f.name, "sigma": sigma,
                              "psnr": p, "ssim": s})
            inputs.append(x.numpy().astype(np.float32))
            meta.append({"image": f.name, "sigma": sigma})
            np.save(OUT / f"clean_{f.stem}.npy", clean)
            np.save(OUT / f"noisy_{f.stem}_s{sigma}.npy", noisy.astype(np.uint8))
            np.save(OUT / f"fp32_{f.stem}_s{sigma}.npy",
                    (pred * 255).clip(0, 255).astype(np.uint8))
            print(f"  {f.name:<16} sigma {sigma:<3} FP32  psnr {p:6.3f}  ssim {s:.4f}")

    np.save(OUT / "inputs.npy", np.concatenate(inputs, 0))
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (OUT / "fp32.json").write_text(json.dumps(fp32_rows, indent=2), encoding="utf-8")

    for sigma in SIGMAS:
        rows = [r for r in fp32_rows if r["sigma"] == sigma]
        print(f"FP32 sigma {sigma}: psnr {np.mean([r['psnr'] for r in rows]):.4f}  "
              f"ssim {np.mean([r['ssim'] for r in rows]):.4f}")

    from src.export.to_onnx import export_onnx
    onnx_path = OUT / "b0_seed0.onnx"
    export_onnx(model, onnx_path, SHAPE)
    print(f"exported {onnx_path.name}")


def submit() -> None:
    import onnx
    import qai_hub as hub

    state_path = OUT / "jobs.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    onnx_path = OUT / "b0_seed0.onnx"
    inputs = np.load(OUT / "inputs.npy")
    name = onnx.load(str(onnx_path)).graph.input[0].name

    if "quantize" not in state:
        # Calibrate on the REAL demo inputs, not random noise: PTQ ranges taken
        # from the wrong distribution is a classic way to lose quality and then
        # blame quantization itself.
        j = hub.submit_quantize_job(
            model=str(onnx_path),
            calibration_data={name: [inputs[i:i + 1] for i in range(len(inputs))]},
            weights_dtype=hub.QuantizeDtype.INT8,
            activations_dtype=hub.QuantizeDtype.INT8,
            name="b0-int8-demo-quantize")
        state["quantize"] = j.job_id
        state_path.write_text(json.dumps(state, indent=2))
        print(f"quantize {j.job_id}")

    q = hub.get_job(state["quantize"])
    if not q.wait().success:
        raise SystemExit("quantize failed")

    if "compile" not in state:
        c = hub.submit_compile_job(model=q.get_target_model(),
                                   device=hub.Device(DEVICE_NAME),
                                   options="--target_runtime qnn_context_binary",
                                   name="b0-int8-demo-compile")
        state["compile"] = c.job_id
        state_path.write_text(json.dumps(state, indent=2))
        print(f"compile {c.job_id}")

    c = hub.get_job(state["compile"])
    if not c.wait().success:
        raise SystemExit("compile failed")

    if "inference" not in state:
        i = hub.submit_inference_job(
            model=c.get_target_model(), device=hub.Device(DEVICE_NAME),
            inputs={name: [inputs[k:k + 1] for k in range(len(inputs))]},
            name="b0-int8-demo-inference")
        state["inference"] = i.job_id
        state_path.write_text(json.dumps(state, indent=2))
        print(f"inference {i.job_id}")
    print(json.dumps(state, indent=2))


def collect() -> None:
    import qai_hub as hub

    state = json.loads((OUT / "jobs.json").read_text())
    job = hub.get_job(state["inference"])
    if not job.wait().success:
        raise SystemExit("inference job failed")
    out = job.download_output_data()
    key = next(iter(out))
    preds = out[key]
    meta = json.loads((OUT / "meta.json").read_text())
    fp32 = json.loads((OUT / "fp32.json").read_text())

    rows = []
    for k, m in enumerate(meta):
        stem = Path(m["image"]).stem
        clean = np.load(OUT / f"clean_{stem}.npy").astype(np.float32) / 255.0
        arr = np.asarray(preds[k])
        arr = arr[0] if arr.ndim == 4 else arr
        pred = arr.transpose(1, 2, 0) if arr.shape[0] == 3 else arr
        pred = np.clip(pred, 0, 1)
        p, s = psnr_ssim(pred, clean, ADAIR_DEFAULT)
        ref = next(r for r in fp32
                   if r["image"] == m["image"] and r["sigma"] == m["sigma"])
        rows.append({**m, "int8_psnr": p, "int8_ssim": s,
                     "fp32_psnr": ref["psnr"], "fp32_ssim": ref["ssim"],
                     "d_psnr": p - ref["psnr"], "d_ssim": s - ref["ssim"]})
        np.save(OUT / f"int8_{stem}_s{m['sigma']}.npy",
                (pred * 255).clip(0, 255).astype(np.uint8))

    (OUT / "int8_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"{'sigma':>6}{'FP32 psnr':>12}{'INT8 psnr':>12}{'delta':>10}"
          f"{'FP32 ssim':>12}{'INT8 ssim':>12}{'delta':>10}")
    for sigma in SIGMAS:
        r = [x for x in rows if x["sigma"] == sigma]
        print(f"{sigma:>6}{np.mean([x['fp32_psnr'] for x in r]):>12.4f}"
              f"{np.mean([x['int8_psnr'] for x in r]):>12.4f}"
              f"{np.mean([x['d_psnr'] for x in r]):>+10.4f}"
              f"{np.mean([x['fp32_ssim'] for x in r]):>12.4f}"
              f"{np.mean([x['int8_ssim'] for x in r]):>12.4f}"
              f"{np.mean([x['d_ssim'] for x in r]):>+10.4f}")
    print(f"\noverall PSNR delta: {np.mean([x['d_psnr'] for x in rows]):+.4f} dB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["prepare", "submit", "collect"])
    ap.add_argument("--ckpt", default=str(OUT / "b0_seed0.pth"))
    a = ap.parse_args()
    {"prepare": lambda: prepare(Path(a.ckpt)),
     "submit": submit, "collect": collect}[a.stage]()
