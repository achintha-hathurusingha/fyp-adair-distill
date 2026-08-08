"""One shareable comparison figure per degradation: input / teacher / student / GT.

    python scripts/task_one_figure.py --task derain  --student <ckpt>
    python scripts/task_one_figure.py --task dehaze  --student <ckpt>
    python scripts/task_one_figure.py --task denoise --student <ckpt> --sigma 25
    python scripts/task_one_figure.py --task all --combined ...   # one 3-row figure

PSNR and SSIM come from the locked harness (``src/eval/metrics.py``,
``ADAIR_DEFAULT``); this script computes no metric of its own.

**The image is the MEDIAN case, selected automatically.** Every task ranks its
held-out set by student PSNR and takes the middle one. Picking the best image is
how a figure stops representing the result it illustrates, and picking by eye is
the same failure with extra steps. Override with ``--image`` when a specific
case is wanted.

**Every panel has a ground truth.** dB is a distance to a reference, so a
photograph off the web cannot carry one — those figures exist elsewhere in this
repo and deliberately report mean absolute change instead.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.data.datasets import build_dataset
from src.eval.metrics import ADAIR_DEFAULT, psnr, ssim
from src.models.teacher_wrapper import load_teacher
from src.train.train import build_model
from src.utils.config import (REPO_ROOT, load_paths, load_yaml,
                              teacher_checkpoint)

VAL_ROOT = {"derain":  "test/derain/demo",
            "dehaze":  "test/dehaze/demo",
            "denoise": "test/denoise/bsd68"}


def _student(ckpt: Path, device: str):
    model = build_model(load_yaml(ckpt.parent / "config.yaml"))
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    for k in ("ema", "model", "state_dict"):
        if isinstance(state, dict) and k in state and isinstance(state[k], dict):
            state = state[k]
            break
    model.load_state_dict(state)
    return model.to(device).eval()


def _samples(task: str, sigma: int, data_root: Path):
    root = data_root / VAL_ROOT[task]
    if task == "denoise":
        return {n: (d, c) for n, d, c
                in build_dataset("denoise", root, sigma=sigma, seed_mode="filename")}
    return {n: (d, c) for n, d, c in build_dataset(task, root)}


@torch.no_grad()
def panels_for(task: str, student_ckpt: Path, sigma: int, image: str | None,
               device: str, data_root: Path):
    """(title, panels) for one task row. Panels are (label, image, psnr, ssim)."""
    samples = _samples(task, sigma, data_root)
    teacher = load_teacher(teacher_checkpoint(task), device=device)
    student = _student(student_ckpt, device)

    def run(m, d):
        return m(d[None].to(device))[0].clamp(0, 1).cpu().numpy().transpose(1, 2, 0)

    if image is None:
        scored = sorted((psnr(run(student, d), c.numpy().transpose(1, 2, 0),
                              ADAIR_DEFAULT), n)
                        for n, (d, c) in samples.items())
        image = scored[len(scored) // 2][1]
        print(f"  {task}: median case {image} ({scored[len(scored)//2][0]:.2f} dB)")

    deg, clean = samples[image]
    gt = clean.numpy().transpose(1, 2, 0)
    din = deg.numpy().transpose(1, 2, 0)
    labels = {"derain": "rainy input", "dehaze": "hazy input",
              "denoise": f"noisy input ($\\sigma$={sigma})"}
    out = [(labels[task], din, psnr(din, gt, ADAIR_DEFAULT), ssim(din, gt, ADAIR_DEFAULT))]
    for lbl, m in (("AdaIR (teacher)\n28.8M params", teacher),
                   ("NAFNet student (ours)\n7.37M params", student)):
        o = run(m, deg)
        out.append((lbl, o, psnr(o, gt, ADAIR_DEFAULT), ssim(o, gt, ADAIR_DEFAULT)))
    out.append(("ground truth", gt, float("inf"), 1.0))
    return f"{task} — {image}", out


def draw(rows, out_path: Path, suptitle: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncol = len(rows[0][1])
    fig, axes = plt.subplots(len(rows), ncol,
                             figsize=(4.0 * ncol, 4.05 * len(rows)), squeeze=False)
    for r, (title, panels) in enumerate(rows):
        for c, (lbl, img, p, s) in enumerate(panels):
            ax = axes[r][c]
            ax.imshow(np.clip(img, 0, 1))
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(lbl, fontsize=13)
            ax.set_xlabel("reference" if not np.isfinite(p)
                          else f"{p:.2f} dB   SSIM {s:.4f}", fontsize=12)
            if c == 0:
                ax.set_ylabel(title, fontsize=12)
    fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"wrote {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True,
                    choices=["derain", "dehaze", "denoise", "all"])
    ap.add_argument("--student", type=Path, action="append", required=True,
                    help="checkpoint; for --task all give three, in the order "
                         "denoise derain dehaze")
    ap.add_argument("--sigma", type=int, default=25)
    ap.add_argument("--image", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    data_root = Path(load_paths()["data_root"])
    if not data_root.is_absolute():
        data_root = REPO_ROOT / data_root

    if args.task == "all":
        order = ["denoise", "derain", "dehaze"]
        if len(args.student) != 3:
            raise SystemExit("--task all needs three --student, in the order "
                             "denoise derain dehaze")
        rows = [panels_for(t, ck, args.sigma, None, args.device, data_root)
                for t, ck in zip(order, args.student)]
        out = args.out or REPO_ROOT / "reports" / "all_tasks_one.png"
        draw(rows, out, "AdaIR teacher vs NAFNet student — held-out images, "
                        "median case per task")
    else:
        rows = [panels_for(args.task, args.student[0], args.sigma, args.image,
                           args.device, data_root)]
        out = args.out or REPO_ROOT / "reports" / f"{args.task}_one.png"
        draw(rows, out, f"{args.task.capitalize()} — held-out image, never seen "
                        "in training")

    for title, panels in rows:
        print(f"\n{title}")
        for lbl, _, p, s in panels:
            print(f"  {lbl.splitlines()[0]:<26} "
                  + ("reference" if not np.isfinite(p) else f"{p:7.2f} dB  ssim {s:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
