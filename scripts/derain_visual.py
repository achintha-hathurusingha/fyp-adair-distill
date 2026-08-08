"""Visual comparison for deraining, at a scale where the difference is visible.

    python scripts/derain_visual.py --student <gt.pth> --student <kd.pth>

**Why not just a side-by-side strip.** At 37 dB the three restorations are
perceptually identical at page scale — a strip of full images shows that all
three remove the rain and nothing more. What separates them is *residual*: what
is left over after subtraction, and where.

So this draws two things a strip cannot:

* **amplified error maps**, ``|output - ground truth|`` scaled by a stated
  factor. Residual rain streaks are structured and show up as diagonal texture;
  ordinary reconstruction error is unstructured. The amplification factor is
  printed on the figure because an error map without one is unreadable.
* **zoomed crops** at native resolution on the region of largest teacher-student
  disagreement, chosen automatically rather than by eye — picking the crop by
  hand is how a figure ends up showing whatever the author hoped for.

Images are selected by GT-only PSNR: the worst case, the median, and the best.
Reporting only the best is the failure mode this ordering exists to prevent.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.data.datasets import build_dataset
from src.eval.metrics import ADAIR_DEFAULT, psnr
from src.models.teacher_wrapper import load_teacher
from src.train.train import build_model
from src.utils.config import REPO_ROOT, load_paths, teacher_checkpoint

AMP = 6           # error-map amplification; stated on the figure
CROP = 128


def _student(ckpt: Path, device: str):
    cfg_p = ckpt.parent / "config.yaml"
    from src.utils.config import load_yaml
    model = build_model(load_yaml(cfg_p))
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    for k in ("ema", "model", "state_dict"):
        if isinstance(state, dict) and k in state and isinstance(state[k], dict):
            state = state[k]
            break
    model.load_state_dict(state)
    return model.to(device).eval()


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--student", type=Path, action="append", required=True)
    ap.add_argument("--label", action="append", required=True)
    ap.add_argument("--teacher", default=None,
                    help="teacher checkpoint; default resolves from paths.yaml")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "reports" / "derain_visual.png")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data_root = Path(load_paths()["data_root"])
    if not data_root.is_absolute():
        data_root = REPO_ROOT / data_root
    samples = list(build_dataset("derain", data_root / "test" / "derain" / "demo"))

    models = {"AdaIR (teacher)": load_teacher(
        args.teacher or teacher_checkpoint("derain"), device=args.device)}
    for lbl, ck in zip(args.label, args.student):
        models[lbl] = _student(ck, args.device)

    # Run everything once, keep outputs and per-image PSNR.
    outs, scores = {}, {}
    for name, deg, clean in samples:
        c = clean.numpy().transpose(1, 2, 0)
        row = {}
        for mname, m in models.items():
            o = m(deg[None].to(args.device))[0].clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
            row[mname] = o
        outs[name] = (deg.numpy().transpose(1, 2, 0), row, c)
        scores[name] = psnr(row[args.label[0]], c, ADAIR_DEFAULT)

    order = sorted(scores, key=scores.get)
    picks = [(order[0], "worst"), (order[len(order) // 2], "median"), (order[-1], "best")]
    print(f"selected by {args.label[0]} PSNR: " +
          ", ".join(f"{n} ({t}, {scores[n]:.2f} dB)" for n, t in picks))

    names = list(models)
    ncols = 2 + len(names) * 2          # input, gt, and per-model (output, error)
    fig, axes = plt.subplots(len(picks), ncols,
                             figsize=(2.5 * ncols, 2.7 * len(picks)), squeeze=False)

    for r, (nm, tag) in enumerate(picks):
        deg, row, clean = outs[nm]
        # Crop where the two students disagree most -- chosen, not hand-picked.
        a, b = row[args.label[0]], row[args.label[1]] if len(args.label) > 1 else clean
        diff = np.abs(a - b).mean(axis=2)
        H, W = diff.shape
        ch, cw = min(CROP, H), min(CROP, W)
        best, by, bx = -1, 0, 0
        for y in range(0, H - ch + 1, 32):
            for x in range(0, W - cw + 1, 32):
                v = diff[y:y + ch, x:x + cw].mean()
                if v > best:
                    best, by, bx = v, y, x
        sl = (slice(by, by + ch), slice(bx, bx + cw))

        col = 0
        axes[r][col].imshow(deg[sl]); axes[r][col].set_ylabel(f"{nm}\n({tag})", fontsize=8)
        if r == 0: axes[r][col].set_title("rainy input", fontsize=10)
        col += 1
        for mname in names:
            axes[r][col].imshow(row[mname][sl])
            if r == 0: axes[r][col].set_title(mname, fontsize=10)
            axes[r][col].set_xlabel(f"{psnr(row[mname], clean, ADAIR_DEFAULT):.2f} dB",
                                    fontsize=8)
            col += 1
        axes[r][col].imshow(clean[sl])
        if r == 0: axes[r][col].set_title("ground truth", fontsize=10)
        col += 1
        for mname in names:
            e = np.clip(np.abs(row[mname] - clean).mean(axis=2) * AMP, 0, 1)
            axes[r][col].imshow(e[sl], cmap="inferno", vmin=0, vmax=1)
            if r == 0:
                axes[r][col].set_title(f"|err| x{AMP}\n{mname.split(' (')[0]}", fontsize=9)
            axes[r][col].set_xlabel(f"mean {np.abs(row[mname]-clean).mean()*255:.2f}/255",
                                    fontsize=8)
            col += 1
        for ax in axes[r]:
            ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(f"Rain100L held-out, {CROP}x{CROP} crops at the region of largest "
                 f"student disagreement.  Error maps amplified x{AMP}.", fontsize=11)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"wrote {args.out}")

    summary = {n: {m: float(psnr(o, outs[n][2], ADAIR_DEFAULT)) for m, o in outs[n][1].items()}
               for n, _ in picks}
    (args.out.with_suffix(".json")).write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
