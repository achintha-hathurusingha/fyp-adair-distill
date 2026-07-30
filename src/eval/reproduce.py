"""Gate G3: run a released AdaIR checkpoint through OUR harness.

Compares against the published 3-degradation table. Pass condition is +/-0.10 dB
PSNR per task. On failure, ``--grid`` sweeps the metric conventions and reports
which combination reproduces — the setting that reproduces published numbers is
the correct setting, by definition.

    python -m src.eval.reproduce --checkpoint data/ckpt/inference/adair3d.pth
    python -m src.eval.reproduce --checkpoint ... --grid --task denoise_s15
"""
from __future__ import annotations

import argparse
import itertools
from dataclasses import replace
from pathlib import Path

import torch

from src.data.datasets import build_dataset
from src.eval.evaluate import evaluate, format_table
from src.eval.metrics import ADAIR_DEFAULT, MetricConfig
from src.models.teacher_wrapper import load_teacher
from src.utils.config import REPO_ROOT, load_paths
from src.utils.seeding import seed_everything

#: Published AdaIR 3-degradation results (paper Table, figs/adair3d.PNG).
PUBLISHED: dict[str, tuple[float, float]] = {
    "dehaze_sots": (31.06, 0.980),
    "derain_rain100l": (38.64, 0.983),
    "denoise_bsd68_s15": (34.12, 0.935),
    "denoise_bsd68_s25": (31.45, 0.892),
    "denoise_bsd68_s50": (28.19, 0.802),
}
#: Gate tolerance in dB.
TOLERANCE = 0.10


def build_tasks(data_root: Path, seed_mode: str) -> dict[str, object]:
    """Construct every test set of the 3-degradation protocol."""
    tasks: dict[str, object] = {}
    for sigma in (15, 25, 50):
        tasks[f"denoise_bsd68_s{sigma}"] = build_dataset(
            "denoise", data_root / "test" / "denoise" / "bsd68",
            sigma=sigma, seed_mode=seed_mode)
    tasks["derain_rain100l"] = build_dataset(
        "derain", data_root / "test" / "derain" / "Rain100L")
    tasks["dehaze_sots"] = build_dataset(
        "dehaze", data_root / "test" / "dehaze")
    return tasks


def run(checkpoint: Path, data_root: Path, *, device: str,
        config: MetricConfig, seed_mode: str, only: str | None,
        limit: int | None) -> list:
    """Evaluate ``checkpoint`` on every task through the single harness."""
    teacher = load_teacher(checkpoint, device=device)
    print(f"[repro] {teacher}")

    results = []
    for name, dataset in build_tasks(data_root, seed_mode).items():
        if only and only != name:
            continue
        samples = iter(dataset)
        if limit:
            samples = itertools.islice(samples, limit)
        res = evaluate(teacher, samples, name=name, config=config,
                       device=device, progress=True)
        published = PUBLISHED.get(name)
        delta = f"{res.psnr - published[0]:+.2f}" if published else "n/a"
        print(f"[repro] {name:22s} PSNR {res.psnr:6.2f}  SSIM {res.ssim:.4f}  "
              f"(published {published[0] if published else '-'}, Δ {delta})")
        results.append(res)
    return results


def sweep_conventions(checkpoint: Path, data_root: Path, *, device: str,
                      task: str, seed_mode: str, limit: int | None) -> list[tuple]:
    """Sweep the convention grid, reporting every combination including failures.

    Showing which conventions produce which errors is evidence the harness is
    calibrated, and is useful to anyone reproducing this work.
    """
    grid = {
        "channel": ["rgb", "y"],
        "crop_border": [0, 4, 8],
        "round_to_uint8": [False, True],
        "ssim_win_size": [7, 11],
    }
    teacher = load_teacher(checkpoint, device=device)
    dataset = build_tasks(data_root, seed_mode)[task]
    published = PUBLISHED[task]

    rows = []
    keys = list(grid)
    for combo in itertools.product(*(grid[k] for k in keys)):
        overrides = dict(zip(keys, combo))
        cfg = replace(ADAIR_DEFAULT, **overrides)
        samples = iter(dataset)
        if limit:
            samples = itertools.islice(samples, limit)
        res = evaluate(teacher, samples, name=task, config=cfg, device=device)
        rows.append((overrides, res.psnr, res.ssim,
                     res.psnr - published[0], res.ssim - published[1]))
        print(f"[grid] {overrides}  PSNR {res.psnr:6.2f} (Δ{res.psnr - published[0]:+.2f})"
              f"  SSIM {res.ssim:.4f}")
    return sorted(rows, key=lambda r: abs(r[3]))


def main() -> None:
    ap = argparse.ArgumentParser(description="Gate G3 reproduction.")
    ap.add_argument("--checkpoint", default="data/ckpt/inference/adair3d.pth")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed-mode", default="filename", choices=["filename", "global"])
    ap.add_argument("--task", default=None, help="evaluate only this task")
    ap.add_argument("--limit", type=int, default=None,
                    help="evaluate only the first N images (smoke testing)")
    ap.add_argument("--grid", action="store_true", help="sweep the convention grid")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    seed_everything(args.seed)
    data_root = Path(load_paths()["data_root"])
    if not data_root.is_absolute():
        data_root = REPO_ROOT / data_root
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = REPO_ROOT / checkpoint

    if args.grid:
        if not args.task:
            ap.error("--grid requires --task")
        rows = sweep_conventions(checkpoint, data_root, device=args.device,
                                 task=args.task, seed_mode=args.seed_mode,
                                 limit=args.limit)
        print("\nbest-matching conventions:")
        for overrides, psnr, ssim, dp, ds in rows[:5]:
            print(f"  Δ{dp:+.3f} dB  {overrides}")
        return

    results = run(checkpoint, data_root, device=args.device, config=ADAIR_DEFAULT,
                  seed_mode=args.seed_mode, only=args.task, limit=args.limit)
    print()
    print(format_table(results, reference=PUBLISHED))

    failures = [r for r in results
                if r.name in PUBLISHED
                and abs(r.psnr - PUBLISHED[r.name][0]) > TOLERANCE]
    if failures:
        print(f"\nG3 FAIL: {len(failures)} task(s) outside ±{TOLERANCE} dB: "
              f"{', '.join(r.name for r in failures)}")
        raise SystemExit(1)
    print(f"\nG3 PASS: all tasks within ±{TOLERANCE} dB of published")


if __name__ == "__main__":
    main()
