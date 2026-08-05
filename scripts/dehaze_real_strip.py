"""Run the teacher and both dehaze students on real photographs.

    python scripts/dehaze_real_strip.py img1.jpg [img2.jpg ...] --out strip.png

**There is no ground truth for a real hazy photo**, so this reports NO PSNR or
SSIM. Quoting a distortion metric against a target that does not exist is how a
qualitative demo turns into a fabricated number. What is reported instead is the
mean absolute change each model applies to the input, which is a description of
what the model did, not a score for how well it did it.

The models see byte-identical input: one decode, cropped to a multiple of 16 the
way the evaluation harness does, shared across all three.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.data.datasets import load_rgb_uint8, to_tensor
from src.models.teacher_wrapper import load_teacher
from src.train.train import build_model
from src.utils.config import REPO_ROOT, load_yaml

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
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("images", nargs="+", type=Path)
    ap.add_argument("--student", type=Path, action="append", required=True)
    ap.add_argument("--label", action="append", required=True)
    ap.add_argument("--teacher", type=Path, default=TEACHER)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "reports" / "dehaze_real_strip.png")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = {"AdaIR (teacher)\n28.8M params": load_teacher(args.teacher,
                                                            device=args.device)}
    for lbl, ck in zip(args.label, args.student):
        models[lbl] = _student(ck, args.device)

    cols = ["hazy input"] + list(models)
    rows = len(args.images)
    fig, axes = plt.subplots(rows, len(cols),
                             figsize=(4.0 * len(cols), 3.0 * rows), squeeze=False)

    for r, path in enumerate(args.images):
        img = load_rgb_uint8(path)               # crop to multiple of 16, as the
        x = to_tensor(img)[None].to(args.device)  # harness does
        panels = [(img.astype(np.float32) / 255.0, None)]
        for m in models.values():
            out = m(x)[0].clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
            delta = float(np.abs(out * 255 - img.astype(np.float32)).mean())
            panels.append((out, delta))
        for c, (arr, delta) in enumerate(panels):
            ax = axes[r][c]
            ax.imshow(arr)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(cols[c], fontsize=12)
            if delta is not None:
                # A description of what the model changed -- NOT a score. There
                # is no ground truth for a real photograph.
                ax.set_xlabel(f"mean change {delta:.1f}/255", fontsize=9)
            if c == 0:
                ax.set_ylabel(path.name, fontsize=9)
        print(f"{path.name}: {img.shape[1]}x{img.shape[0]}  " +
              "  ".join(f"{n.splitlines()[0]} {d:.1f}/255"
                        for n, (_, d) in zip(cols[1:], panels[1:])))

    fig.suptitle("Real hazy photograph — no ground truth exists, so no PSNR is "
                 "reported", fontsize=11, y=1.0)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
