"""Compute everything the demo notebook displays, and cache it to disk.

Separated from the notebook on purpose: the heavy work (204 BSD68 evaluations
plus real-world inference on two models) runs once here, and the notebook stays
fast to re-run and easy to share.

    python scripts/build_demo_data.py            # everything
    python scripts/build_demo_data.py --skip-bsd68

Outputs land in runs/demo_nb/.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.eval.metrics import ADAIR_DEFAULT, psnr_ssim
from src.models.nafnet import NAFNet
from src.models.teacher_wrapper import load_teacher
from src.utils.config import REPO_ROOT, load_paths

OUT = REPO_ROOT / "runs" / "demo_nb"
B0_CKPT = REPO_ROOT / "runs" / "int8_demo" / "b0_seed0.pth"
TEACHER = REPO_ROOT / "data" / "ckpt" / "adair-single-denoise.ckpt"
SIGMAS = (15, 25, 50)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _load_b0():
    ck = torch.load(B0_CKPT, map_location="cpu", weights_only=False)
    m = NAFNet(**ck["config"]["model"])
    m.load_state_dict(ck["model"])
    return m.eval().to(DEV), ck


def _infer(model, img_u8: np.ndarray) -> np.ndarray:
    """Run a uint8 HWC image through a model, return uint8 HWC."""
    x = torch.from_numpy(img_u8.astype(np.float32).transpose(2, 0, 1))[None] / 255.0
    with torch.no_grad():
        y = model(x.to(DEV))
    out = np.clip(y[0].float().cpu().numpy().transpose(1, 2, 0), 0, 1)
    return (out * 255).round().astype(np.uint8)


def _crop16(a: np.ndarray) -> np.ndarray:
    """AdaIR requires dimensions divisible by 16 (pixel_unshuffle)."""
    h, w = a.shape[:2]
    return a[: h - h % 16, : w - w % 16]


def bsd68(b0, teacher) -> None:
    """Full BSD68: 68 images x 3 sigmas x 2 models = 408 inferences."""
    from src.data.datasets import load_rgb_uint8

    paths = load_paths()
    root = Path(paths["data_root"])
    if not root.is_absolute():
        root = REPO_ROOT / root
    files = sorted(p for p in (root / "test" / "denoise" / "bsd68").rglob("*")
                   if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})
    print(f"BSD68: {len(files)} images x {len(SIGMAS)} sigmas")

    rows = []
    for sigma in SIGMAS:
        rng = np.random.RandomState(0)          # fixed per sigma: reproducible
        t0 = time.perf_counter()
        for f in files:
            clean = _crop16(load_rgb_uint8(f, base=1))
            noisy = np.clip(clean.astype(np.float32)
                            + rng.normal(0, sigma, clean.shape), 0, 255).astype(np.uint8)
            tgt = clean.astype(np.float32) / 255.0
            ps, ss = psnr_ssim(_infer(b0, noisy).astype(np.float32) / 255.0, tgt, ADAIR_DEFAULT)
            pt, st = psnr_ssim(_infer(teacher, noisy).astype(np.float32) / 255.0, tgt, ADAIR_DEFAULT)
            pn, sn = psnr_ssim(noisy.astype(np.float32) / 255.0, tgt, ADAIR_DEFAULT)
            rows.append({"image": f.name, "sigma": sigma, "h": clean.shape[0],
                         "w": clean.shape[1], "noisy_psnr": pn, "noisy_ssim": sn,
                         "b0_psnr": ps, "b0_ssim": ss,
                         "adair_psnr": pt, "adair_ssim": st})
        dt = time.perf_counter() - t0
        s = [r for r in rows if r["sigma"] == sigma]
        print(f"  sigma {sigma}: noisy {np.mean([r['noisy_psnr'] for r in s]):6.3f}  "
              f"B0 {np.mean([r['b0_psnr'] for r in s]):6.3f}  "
              f"AdaIR {np.mean([r['adair_psnr'] for r in s]):6.3f}  ({dt:.0f}s)")
    (OUT / "bsd68.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")


def real_world(b0, teacher) -> None:
    """Two regimes on the same real photos.

    A) SYNTHETIC: our own pipeline noise added -> ground truth exists, so PSNR
       is meaningful. Tests generalisation to modern content.
    B) NATIVE: the downloaded JPEG as-is, carrying real capture noise and JPEG
       artefacts. NO ground truth exists, so no PSNR is possible and none is
       reported -- visual evidence only. This is the honest way to show
       behaviour on genuinely real degradation.
    """
    from PIL import Image

    src = REPO_ROOT / "data" / "real_world" / "originals"
    files = sorted(src.glob("*.jpg"))
    print(f"real-world: {len(files)} images, synthetic + native regimes")
    vis = OUT / "real"
    vis.mkdir(parents=True, exist_ok=True)

    rows = []
    for f in files:
        clean = _crop16(np.asarray(Image.open(f).convert("RGB")))
        np.save(vis / f"{f.stem}_clean.npy", clean)

        # --- A) synthetic, PSNR meaningful ---
        for sigma in SIGMAS:
            rng = np.random.RandomState(abs(hash(f.stem)) % 2**31)
            noisy = np.clip(clean.astype(np.float32)
                            + rng.normal(0, sigma, clean.shape), 0, 255).astype(np.uint8)
            b = _infer(b0, noisy)
            a = _infer(teacher, noisy)
            tgt = clean.astype(np.float32) / 255.0
            pn, sn = psnr_ssim(noisy.astype(np.float32) / 255.0, tgt, ADAIR_DEFAULT)
            pb, sb = psnr_ssim(b.astype(np.float32) / 255.0, tgt, ADAIR_DEFAULT)
            pa, sa = psnr_ssim(a.astype(np.float32) / 255.0, tgt, ADAIR_DEFAULT)
            np.save(vis / f"{f.stem}_s{sigma}_noisy.npy", noisy)
            np.save(vis / f"{f.stem}_s{sigma}_b0.npy", b)
            np.save(vis / f"{f.stem}_s{sigma}_adair.npy", a)
            rows.append({"image": f.name, "regime": "synthetic", "sigma": sigma,
                         "noisy_psnr": pn, "noisy_ssim": sn,
                         "b0_psnr": pb, "b0_ssim": sb,
                         "adair_psnr": pa, "adair_ssim": sa})

        # --- B) native degradation, NO ground truth ---
        b = _infer(b0, clean)
        a = _infer(teacher, clean)
        np.save(vis / f"{f.stem}_native_b0.npy", b)
        np.save(vis / f"{f.stem}_native_adair.npy", a)
        # Only a self-consistency figure is defensible here: how much the model
        # changed the input. It is NOT a quality score.
        rows.append({"image": f.name, "regime": "native", "sigma": None,
                     "b0_mae_vs_input": float(np.abs(b.astype(np.float32)
                                                     - clean.astype(np.float32)).mean()),
                     "adair_mae_vs_input": float(np.abs(a.astype(np.float32)
                                                        - clean.astype(np.float32)).mean())})
    (OUT / "real_world.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    for sigma in SIGMAS:
        s = [r for r in rows if r["regime"] == "synthetic" and r["sigma"] == sigma]
        print(f"  sigma {sigma}: noisy {np.mean([r['noisy_psnr'] for r in s]):6.3f}  "
              f"B0 {np.mean([r['b0_psnr'] for r in s]):6.3f}  "
              f"AdaIR {np.mean([r['adair_psnr'] for r in s]):6.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-bsd68", action="store_true")
    ap.add_argument("--skip-real", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    b0, ck = _load_b0()
    print(f"B0    iteration {ck['iteration']}  "
          f"{sum(p.numel() for p in b0.parameters()):,} params  on {DEV}")
    teacher = load_teacher(TEACHER, device=DEV)
    print(f"AdaIR {sum(p.numel() for p in teacher.parameters()):,} params  on {DEV}")

    (OUT / "models.json").write_text(json.dumps({
        "b0_iteration": ck["iteration"], "b0_best_psnr": ck["best_psnr"],
        "b0_params": sum(p.numel() for p in b0.parameters()),
        "b0_config": ck["config"]["model"],
        "adair_params": sum(p.numel() for p in teacher.parameters()),
        "device": DEV}, indent=2, default=str), encoding="utf-8")

    if not a.skip_bsd68:
        bsd68(b0, teacher)
    if not a.skip_real:
        real_world(b0, teacher)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
