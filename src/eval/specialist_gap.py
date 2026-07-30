"""Measure the single-task specialist advantage over the all-in-one teacher.

This decides whether the specialist->generalist (multi-teacher) direction is
worth its complexity in Phase 02. The premise is that specialists outperform the
all-in-one model on their own task, and that the surplus is knowledge a student
can inherit which ground truth alone does not provide. If the surplus is small,
the option buys complexity for nothing.

    python -m src.eval.specialist_gap --device cuda

**Confound, which must travel with any number this produces.** The released
specialists were not trained on a common protocol — epoch counts, step counts
and steps-per-epoch all differ from each other and from the all-in-one
(see ``reports/checkpoint_audit.md``). Differing steps-per-epoch implies
differing training-set sizes. Any measured advantage is therefore *part*
specialisation and *part* a longer or differently-sourced training run, and the
two cannot be separated from these artifacts alone.
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import torch

from src.eval.evaluate import evaluate
from src.eval.metrics import ADAIR_DEFAULT
from src.eval.reproduce import build_tasks
from src.models.teacher_wrapper import load_teacher
from src.utils.config import REPO_ROOT, load_paths
from src.utils.seeding import seed_everything

#: Which specialist checkpoint serves which task.
SPECIALISTS = {
    "denoise": "adair-single-denoise.pth",
    "derain": "adair-single-derain.pth",
    "dehaze": "adair-single-dehaze.pth",
}
#: Decision thresholds on the mean PSNR advantage, in dB.
THRESHOLD_STRONG = 0.5
THRESHOLD_MARGINAL = 0.2


def task_family(name: str) -> str:
    """Map an evaluation-set name onto its degradation type."""
    return name.split("_", 1)[0]


def main() -> None:
    ap = argparse.ArgumentParser(description="Specialist vs all-in-one gap.")
    ap.add_argument("--ckpt-dir", default="data/ckpt/inference")
    ap.add_argument("--all-in-one", default="adair3d.pth")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    seed_everything(args.seed)
    data_root = Path(load_paths()["data_root"])
    if not data_root.is_absolute():
        data_root = REPO_ROOT / data_root
    ckpt_dir = Path(args.ckpt_dir)
    if not ckpt_dir.is_absolute():
        ckpt_dir = REPO_ROOT / ckpt_dir

    tasks = build_tasks(data_root, seed_mode="filename")
    all_in_one = load_teacher(ckpt_dir / args.all_in_one, device=args.device)

    rows = []
    for name, dataset in tasks.items():
        family = task_family(name)
        spec_file = ckpt_dir / SPECIALISTS[family]
        if not spec_file.exists():
            print(f"[gap] SKIP {name}: {spec_file.name} not found")
            continue
        specialist = load_teacher(spec_file, device=args.device)

        def run(model):
            samples = iter(dataset)
            if args.limit:
                samples = itertools.islice(samples, args.limit)
            return evaluate(model, samples, name=name, config=ADAIR_DEFAULT,
                            device=args.device)

        aio = run(all_in_one)
        spec = run(specialist)
        delta = spec.psnr - aio.psnr
        rows.append((name, aio.psnr, spec.psnr, delta,
                     spec.ssim - aio.ssim, aio.n_images))
        print(f"[gap] {name:22s} all-in-one {aio.psnr:6.2f}  "
              f"specialist {spec.psnr:6.2f}  Δ {delta:+.3f} dB")

    if not rows:
        raise SystemExit("no tasks evaluated")

    print()
    print("| test set | n | all-in-one PSNR | specialist PSNR | ΔPSNR | ΔSSIM |")
    print("|---|---|---|---|---|---|")
    for name, aio, spec, dp, ds, n in rows:
        print(f"| {name} | {n} | {aio:.2f} | {spec:.2f} | **{dp:+.3f}** | {ds:+.4f} |")

    mean_gap = sum(r[3] for r in rows) / len(rows)
    print(f"\nmean specialist advantage: **{mean_gap:+.3f} dB**")
    if mean_gap > THRESHOLD_STRONG:
        verdict = ("REAL HEADROOM — Option 2 is live; 'student beats the "
                   "all-in-one teacher' becomes a plausible headline result")
    elif mean_gap > THRESHOLD_MARGINAL:
        verdict = ("MARGINAL — viable only if it costs little; note that "
                   "multi-teacher adds ~zero GPU cost, only storage and routing")
    else:
        verdict = "PREMISE FAILS — drop Option 2"
    print(f"verdict: {verdict}")
    print("\nCONFOUND: the specialists were trained on task-specific protocols "
          "(differing epochs/steps/steps-per-epoch), so this gap mixes "
          "specialisation with a different training run. See "
          "reports/checkpoint_audit.md.")


if __name__ == "__main__":
    main()
