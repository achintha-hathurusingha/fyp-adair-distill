"""Generate a cross-implementation oracle for the metric conventions.

Run this **inside the legacy environment** (Python 3.8 / scikit-image 0.19.3 /
scikit-video 1.1.11) where AdaIR's own ``compute_psnr_ssim`` executes. It feeds
that function fixed, seeded arrays and dumps inputs plus outputs to JSON.

``tests/test_golden_metrics.py`` then asserts our modern implementation matches
that JSON. This is a genuine cross-implementation oracle: it settles metric
correctness independently of datasets, teacher and dataloaders, so a G3 failure
cannot be ambiguous between four subsystems.

    # in the legacy env:
    python scripts/make_golden_metrics.py --out tests/golden/adair_metrics.json

Deliberately dependency-light — it imports AdaIR's function and numpy, nothing
from ``src/``, so it runs in an environment where this project is not installed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

#: Shapes exercised, including BSD68's post-crop geometry (480x320) and a
#: non-square case. Small shapes keep the JSON compact.
SHAPES = [
    (32, 32, 3),
    (64, 48, 3),
    (17, 23, 3),      # odd dimensions, smaller than the 481x321 originals
    (320, 480, 3),    # BSD68 after crop_img(base=16)
]
#: Noise levels applied to build degraded/clean pairs, in [0, 1] space.
NOISE_LEVELS = [0.0, 0.01, 0.05, 0.2]


def build_cases(n_per_shape: int = 2) -> list[dict]:
    """Deterministic (clean, degraded) pairs spanning shapes and noise levels."""
    cases = []
    case_id = 0
    for shape in SHAPES:
        for i in range(n_per_shape):
            noise = NOISE_LEVELS[(case_id) % len(NOISE_LEVELS)]
            rng = np.random.RandomState(1000 + case_id)
            clean = rng.rand(*shape)
            degraded = np.clip(clean + rng.randn(*shape) * noise, 0, 1)
            cases.append({
                "id": case_id,
                "shape": list(shape),
                "noise": noise,
                "seed": 1000 + case_id,
                "clean": clean,
                "degraded": degraded,
            })
            case_id += 1
    return cases


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate golden metric values.")
    ap.add_argument("--adair-root", default="third_party/AdaIR")
    ap.add_argument("--out", default="tests/golden/adair_metrics.json")
    ap.add_argument("--n-per-shape", type=int, default=2)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.adair_root).resolve()))
    import torch  # noqa: E402  (legacy env import order)
    from utils.val_utils import compute_psnr_ssim  # noqa: E402

    import skimage  # noqa: E402
    print(f"[golden] scikit-image {skimage.__version__}, torch {torch.__version__}")
    if skimage.__version__ >= "0.23":
        print("[golden] WARNING: multichannel= was removed in scikit-image 0.23; "
              "this must run in the legacy env (0.19.3) to be meaningful.")

    records = []
    for case in build_cases(args.n_per_shape):
        # AdaIR's function takes NCHW torch tensors.
        clean_t = torch.from_numpy(case["clean"].transpose(2, 0, 1)).unsqueeze(0)
        deg_t = torch.from_numpy(case["degraded"].transpose(2, 0, 1)).unsqueeze(0)
        psnr, ssim, n = compute_psnr_ssim(deg_t, clean_t)
        records.append({
            "id": case["id"],
            "shape": case["shape"],
            "noise": case["noise"],
            "seed": case["seed"],
            "n": int(n),
            "psnr": float(psnr),
            "ssim": float(ssim),
        })
        print(f"[golden] case {case['id']:2d} shape={case['shape']} "
              f"noise={case['noise']:.2f}  PSNR={psnr:.6f}  SSIM={ssim:.6f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "generator": "AdaIR utils/val_utils.py compute_psnr_ssim",
        "skimage_version": skimage.__version__,
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "construction": (
            "clean = RandomState(seed).rand(*shape); "
            "degraded = clip(clean + RandomState(seed).randn(*shape) * noise, 0, 1) "
            "with the SAME RandomState instance, drawn in that order"
        ),
        "cases": records,
    }, indent=2), encoding="utf-8")
    print(f"[golden] wrote {len(records)} cases -> {out}")


if __name__ == "__main__":
    main()
