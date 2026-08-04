"""Measure the AdaIR-to-NAFNet dehazing gap on the held-out demo set.

    python scripts/dehaze_gap.py --student <ckpt> [--student <ckpt2> ...]

Every model sees **byte-identical inputs** — the same ``PairedTestDataset``
instance feeds each one, so nothing about loading, cropping or ordering can
differ between them. Metrics go through the locked harness
(``src/eval/evaluate.py`` with ``ADAIR_DEFAULT``); there are no metrics computed
in this file.

Multiple ``--student`` checkpoints are compared side by side, which is how the
GT-only and GT+KD runs are put on the same table: same images, same harness,
same teacher column, one row per model.

Writes ``reports/dehaze_gap.json`` and a comparison strip
(hazy / teacher / student(s) / ground truth).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.data.datasets import build_dataset
from src.eval.evaluate import evaluate
from src.eval.metrics import ADAIR_DEFAULT
from src.models.teacher_wrapper import load_teacher
from src.train.train import build_model
from src.utils.config import REPO_ROOT, load_paths, load_yaml

TEACHER = Path("/home/minura/FYP/Workspace/Himeth/weights/adair-single-dehaze.ckpt")


def _student(ckpt: Path, device: str):
    cfg = load_yaml(ckpt.parent / "config.yaml")
    model = build_model(cfg)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    for key in ("ema", "model", "state_dict"):
        if isinstance(state, dict) and key in state and isinstance(state[key], dict):
            state = state[key]
            break
    model.load_state_dict(state)
    return model.to(device).eval()


@torch.no_grad()
def _strip(models: dict, samples: list, out: Path, n: int = 4) -> None:
    """hazy / each model / ground truth, one row per image."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = ["hazy input"] + list(models) + ["ground truth"]
    fig, axes = plt.subplots(n, len(cols), figsize=(3.1 * len(cols), 3.1 * n))
    for r, (name, deg, clean) in enumerate(samples[:n]):
        imgs = [deg]
        for m in models.values():
            imgs.append(m(deg[None].to(next(m.parameters()).device)
                          if hasattr(m, "parameters") else deg[None])[0]
                        .clamp(0, 1).cpu())
        imgs.append(clean)
        for c, img in enumerate(imgs):
            ax = axes[r, c]
            ax.imshow(img.numpy().transpose(1, 2, 0))
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(cols[c], fontsize=11)
            if c == 0:
                ax.set_ylabel(name, fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"wrote {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--student", type=Path, action="append", required=True,
                    help="student checkpoint; repeat to compare several")
    ap.add_argument("--label", action="append", default=None,
                    help="display name per --student, in the same order")
    ap.add_argument("--teacher", type=Path, default=TEACHER)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--val-root", default="test/dehaze/demo")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "reports" / "dehaze_gap.json")
    ap.add_argument("--strip", type=Path,
                    default=REPO_ROOT / "reports" / "dehaze_gap_strip.png")
    args = ap.parse_args()

    data_root = Path(load_paths()["data_root"])
    if not data_root.is_absolute():
        data_root = REPO_ROOT / data_root
    root = data_root / args.val_root

    labels = args.label or [f"student{i}" if i else "student"
                            for i in range(len(args.student))]
    if len(labels) != len(args.student):
        raise SystemExit(f"{len(labels)} labels for {len(args.student)} students")

    models: dict[str, object] = {}
    teacher = load_teacher(args.teacher, device=args.device)
    models["AdaIR (teacher)"] = teacher
    for lbl, ck in zip(labels, args.student):
        models[lbl] = _student(ck, args.device)

    # Identical inputs: one dataset, materialised once, replayed per model.
    samples = list(build_dataset("dehaze", root))
    print(f"held-out set : {len(samples)} pairs from {root}")
    print(f"teacher      : {args.teacher.name}\n")

    results = {}
    for name, model in models.items():
        res = evaluate(model, iter(samples), name=name, config=ADAIR_DEFAULT,
                       device=args.device, keep_per_image=False)
        results[name] = {"psnr": res.psnr, "ssim": res.ssim, "images": res.count}
        print(f"{name:<22} psnr {res.psnr:7.4f}  ssim {res.ssim:.4f}  "
              f"n={res.count}")

    t = results["AdaIR (teacher)"]
    print()
    for name, r in results.items():
        if name == "AdaIR (teacher)":
            continue
        print(f"gap  teacher - {name:<18} {t['psnr'] - r['psnr']:+7.4f} dB  "
              f"{t['ssim'] - r['ssim']:+.4f} ssim")

    _strip(models, samples, args.strip)
    args.out.write_text(json.dumps(
        {"val_root": str(root), "n_images": len(samples),
         "teacher_ckpt": str(args.teacher), "results": results,
         "students": {l: str(c) for l, c in zip(labels, args.student)}},
        indent=1), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
