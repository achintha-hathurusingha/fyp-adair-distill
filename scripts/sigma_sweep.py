"""The F10 gate: does a checkpoint still collapse on low-noise input?

F10 was found by eye — running B0-denoise on real photographs and seeing a
rainbow checkerboard — and the table in findings.md was measured ad hoc. That is
fine for a discovery and not fine for a gate, because the number that clears
B0-v2 has to be reproducible against the number that condemned B0-denoise. This
script is that measurement, fixed in code.

    python scripts/sigma_sweep.py <checkpoint.pth> [--config configs/train/b0v2_multitask.yaml]

Two tests, both from F10:

**Sigma sweep** over BSD68 at sigma 0, 5, 8, 10, 15, 25, 50. B0-denoise scored
4.25 / 4.20 / 4.29 dB at sigma 0/5/8 and 38.82 at sigma 10 — a cliff, not a
slope, since sigma 10 is equally unseen and works. Output std and mean change
are reported alongside PSNR because they are what distinguish "slightly wrong"
from "replaced the image with noise": at the cliff B0-denoise's output std was
127 (a saturated checkerboard) against ~92 in the healthy regime.

**Clean-input identity** on real photographs, no synthetic noise. The
restoration a model applies to an already-clean image should be near zero.
AdaIR changes it by 1.88/255; B0-denoise by 125.37/255.

PASS CRITERIA, stated before the run so they cannot be adjusted after it:

* no sigma in the sweep scores below ``--min-psnr`` (default 25 dB)
* PSNR is monotone non-increasing in sigma to within ``--tol`` (default 1.0 dB) —
  a cliff shows up as a large violation, a normal curve does not
* clean-input mean absolute change is below ``--max-clean-mae`` (default 10/255)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.data.datasets import load_rgb_uint8, to_tensor
from src.data.degradations import add_gaussian_noise
from src.eval.metrics import psnr as psnr_fn
from src.train.train import build_model
from src.utils.config import REPO_ROOT, load_paths, load_yaml

#: The exact sigmas F10 was characterised on. 0/5/8 failed, 10 worked.
SWEEP_SIGMAS = (0, 5, 8, 10, 15, 25, 50)


def _load(checkpoint: Path, cfg: dict, device: str) -> torch.nn.Module:
    model = build_model(cfg)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    # Prefer EMA weights when present: they are what evaluation and export use,
    # so gating on raw weights would clear a model nobody deploys.
    for key in ("ema", "ema_state_dict", "model", "state_dict"):
        if isinstance(state, dict) and key in state and isinstance(state[key], dict):
            state = state[key]
            break
    model.load_state_dict(state)
    return model.to(device).eval()


@torch.no_grad()
def sweep(model, files: list[Path], device: str) -> list[dict]:
    rows = []
    for sigma in SWEEP_SIGMAS:
        psnrs, stds, changes, in_psnrs = [], [], [], []
        for path in files:
            clean = load_rgb_uint8(path)
            noisy = (clean if sigma == 0
                     else add_gaussian_noise(clean, sigma, filename=path.name))
            out = model(to_tensor(noisy)[None].to(device))[0].clamp(0, 1).cpu()
            # METRICS IN [0,1]. MetricConfig.data_range is 1.0 and clip is True,
            # so handing it uint8 clips everything above 1 to 1, makes both
            # images identical and returns inf -- which would have scored a
            # destroyed output as a perfect one. Reporting stays in 0-255,
            # which is the scale F10's table used.
            out01 = out.numpy().transpose(1, 2, 0)
            clean01 = clean.astype(np.float32) / 255.0
            psnrs.append(psnr_fn(out01, clean01))
            in_psnrs.append(psnr_fn(noisy.astype(np.float32) / 255.0, clean01))
            out_u8 = (out01 * 255).round()
            stds.append(float(out_u8.std()))
            changes.append(float(np.abs(out_u8 - noisy.astype(float)).mean()))
        rows.append({"sigma": sigma,
                     "input_psnr": float(np.mean(in_psnrs)),
                     "psnr": float(np.mean(psnrs)),
                     "output_std": float(np.mean(stds)),
                     "mean_change": float(np.mean(changes))})
    return rows


@torch.no_grad()
def clean_identity(model, files: list[Path], device: str) -> float:
    """Mean absolute change applied to an already-clean image, in 0-255."""
    deltas = []
    for path in files:
        img = load_rgb_uint8(path)
        out = model(to_tensor(img)[None].to(device))[0].clamp(0, 1).cpu()
        out_u8 = (out.numpy().transpose(1, 2, 0) * 255).round()
        deltas.append(float(np.abs(out_u8 - img.astype(float)).mean()))
    return float(np.mean(deltas))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--config", type=Path, default=None,
                    help="resolved config.yaml; defaults to the one beside the checkpoint")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=0, help="use only N test images")
    ap.add_argument("--min-psnr", type=float, default=25.0)
    ap.add_argument("--tol", type=float, default=1.0,
                    help="allowed non-monotonicity in dB before it counts as a cliff")
    ap.add_argument("--max-clean-mae", type=float, default=10.0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    # The run directory's OWN resolved config, so the architecture gated is
    # the architecture trained -- not whatever the template says today.
    cfg_path = args.config or args.checkpoint.parent / "config.yaml"
    if not cfg_path.exists():
        raise SystemExit(f"no config at {cfg_path}; pass --config")
    cfg = load_yaml(cfg_path)
    paths = load_paths()
    data_root = Path(paths["data_root"])
    if not data_root.is_absolute():
        data_root = REPO_ROOT / data_root

    bsd = sorted((data_root / "test" / "denoise" / "bsd68").glob("*"))
    bsd = [p for p in bsd if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp")]
    if not bsd:
        raise SystemExit(f"no BSD68 images under {data_root}")
    if args.limit:
        bsd = bsd[:args.limit]

    real_dir = REPO_ROOT / "data" / "real_world"
    # Recursive: the ten photographs F10 used live in real_world/originals/,
    # with derived strips beside them in other subdirectories.
    real = sorted(p for p in real_dir.rglob("*")
                  if p.suffix.lower() in (".png", ".jpg", ".jpeg")
                  and p.parent.name == "originals") if real_dir.exists() else []

    model = _load(args.checkpoint, cfg, args.device)
    print(f"checkpoint : {args.checkpoint}")
    print(f"BSD68      : {len(bsd)} images   real-world: {len(real)}")
    print()

    rows = sweep(model, bsd, args.device)
    print(f"{'sigma':>6} {'input':>8} {'PSNR':>8} {'out std':>8} {'change':>8}")
    for r in rows:
        ip = "inf" if not np.isfinite(r["input_psnr"]) else f"{r['input_psnr']:.2f}"
        print(f"{r['sigma']:>6} {ip:>8} {r['psnr']:>8.2f} "
              f"{r['output_std']:>8.1f} {r['mean_change']:>8.1f}")

    # A non-finite PSNR means the comparison degenerated, not that the model is
    # perfect. Refuse to grade rather than let inf clear the threshold.
    if not all(np.isfinite(r["psnr"]) for r in rows):
        bad = [r["sigma"] for r in rows if not np.isfinite(r["psnr"])]
        raise SystemExit(
            f"non-finite PSNR at sigma {bad} — the metric degenerated "
            "(identical inputs after clipping?). Not a pass; fix the harness.")

    worst = min(r["psnr"] for r in rows)
    # A cliff is a LOW sigma scoring far worse than a HIGHER one. Normal
    # degradation is monotone decreasing, so only increases beyond tol count.
    violations = [(a["sigma"], b["sigma"], b["psnr"] - a["psnr"])
                  for a, b in zip(rows, rows[1:]) if b["psnr"] - a["psnr"] > args.tol]

    clean_mae = clean_identity(model, real, args.device) if real else float("nan")
    print(f"\nclean-input mean abs change : {clean_mae:.2f}/255 "
          f"(AdaIR 1.88, B0-denoise 125.37)")

    ok = worst >= args.min_psnr and not violations
    if real:
        ok = ok and clean_mae <= args.max_clean_mae
    print(f"\nworst sigma PSNR : {worst:.2f} dB (need >= {args.min_psnr})")
    if violations:
        print("CLIFF DETECTED — a lower sigma scores worse than a higher one:")
        for lo, hi, d in violations:
            print(f"  sigma {lo} is {d:.2f} dB WORSE than sigma {hi}")
    print(f"\n{'PASS' if ok else 'FAIL'}  F10 gate")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"checkpoint": str(args.checkpoint), "sweep": rows,
             "clean_mae": clean_mae, "pass": ok,
             "criteria": {"min_psnr": args.min_psnr, "tol": args.tol,
                          "max_clean_mae": args.max_clean_mae}},
            indent=1), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
