"""One-time precompute: cache the frozen teacher's (response, latent_pre)
for a large finite pool of (degraded, clean) samples drawn from the exact
same live multi-task loader training already uses. See
reports/kd_feature_multitask/plan_cached_teacher.md.

Why: profile_step_cost.py found the teacher's forward pass is ~78% of every
training step's wall-clock time, paid fresh on every sample because live
training draws a fresh random crop (and, for denoise, a fresh continuous
sigma) every single time -- there is no repetition to exploit online. This
script converts that into a one-time cost: draw N samples once, run the
teacher once each, cache the result, then train by reading the cache.

Storage: single set of memmap files (not per-task -- simpler, and a JSON
index already records task/sigma per row, so per-task filtering at train
time is just index slicing, matching how BalancedTaskBatchSampler already
works):
  degraded.dat    (N, 3, 128, 128) uint8   -- CHW, matches to_tensor() layout
  clean.dat       (N, 3, 128, 128) uint8
  response.dat    (N, 3, 128, 128) uint8   -- teacher's output, same domain
  latent_pre.dat  (N, 384, 16, 16) float16 -- half precision (numpy has no
                                               bf16; float16's range is ample
                                               for these activation magnitudes)
  index.json      -- {"task": [...], "sigma": [...], "n": N, "shapes": {...}}

Usage:
    python scripts/build_teacher_cache.py --total 180000 --out-dir /path/to/cache
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

import sys
sys.path.insert(0, ".")

from src.data.build import build_multitask_loader
from src.models.teacher_wrapper import load_teacher
from src.utils.config import teacher_checkpoint

LATENT_C, LATENT_H, LATENT_W = 384, 16, 16
PATCH = 128


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=180_000)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = yaml.safe_load(Path("configs/paths.local.yaml").read_text())
    data_root = Path(paths["data_root"])
    sources = {
        "denoise": data_root / paths["datasets"]["denoise_train"],
        "derain": data_root / paths["datasets"]["derain_train"],
        "dehaze": {"input": data_root / "Train/Dehaze/synthetic",
                  "target": data_root / "Train/Dehaze/clear"},
    }

    print(f"Building teacher cache: {args.total:,} samples -> {out_dir}")
    loader = build_multitask_loader(
        sources, batch_size=args.batch_size, patch_size=PATCH,
        sigma_range=(0.0, 55.0), clean_prob=0.05,
        num_workers=args.num_workers, seed=args.seed, length=args.total)

    teacher = load_teacher(teacher_checkpoint("all_in_one"), device=args.device)

    n = args.total
    degraded_mm = np.memmap(out_dir / "degraded.dat", dtype=np.uint8, mode="w+",
                            shape=(n, 3, PATCH, PATCH))
    clean_mm = np.memmap(out_dir / "clean.dat", dtype=np.uint8, mode="w+",
                         shape=(n, 3, PATCH, PATCH))
    response_mm = np.memmap(out_dir / "response.dat", dtype=np.uint8, mode="w+",
                            shape=(n, 3, PATCH, PATCH))
    latent_mm = np.memmap(out_dir / "latent_pre.dat", dtype=np.float16, mode="w+",
                          shape=(n, LATENT_C, LATENT_H, LATENT_W))

    tasks: list[int] = []
    sigmas: list[float] = []

    # `length=` passed to build_multitask_loader is not a guaranteed exact
    # total -- BalancedTaskBatchSampler's own batch-count arithmetic can
    # come up short by up to one batch (confirmed: length=512 delivered
    # only 496). Loop until exactly `n` rows are actually written, re-
    # iterating the loader (fresh random draws each pass, same as any
    # epoch boundary) rather than trusting a single pass to reach `n`.
    row = 0
    t0 = time.time()
    loader_iter = iter(loader)
    with torch.no_grad():
        while row < n:
            try:
                degraded, clean, prov = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                continue
            b = min(degraded.shape[0], n - row)
            degraded, clean = degraded[:b], clean[:b]
            degraded_dev = degraded.to(args.device)

            soft, latent = teacher.forward_with_latent(degraded_dev.float())

            degraded_mm[row:row + b] = (degraded * 255).round().clamp(0, 255) \
                .byte().cpu().numpy()
            clean_mm[row:row + b] = (clean * 255).round().clamp(0, 255) \
                .byte().cpu().numpy()
            response_mm[row:row + b] = (soft[:b].clamp(0, 1) * 255).round() \
                .clamp(0, 255).byte().cpu().numpy()
            latent_mm[row:row + b] = latent[:b].to(torch.float16).cpu().numpy()

            tasks.extend(int(t) for t in prov["task"][:b].tolist())
            sigmas.extend(float(s) for s in prov["sigma"][:b].tolist())

            row += b
            if row % (args.batch_size * 50) < args.batch_size:
                elapsed = time.time() - t0
                rate = row / elapsed if elapsed > 0 else 0
                eta = (n - row) / rate if rate > 0 else float("inf")
                print(f"  {row:,}/{n:,}  ({rate:.1f} samples/s, "
                      f"ETA {eta/60:.1f} min)")

    assert row == n, f"expected exactly {n} rows written, got {row}"
    assert len(tasks) == n and len(sigmas) == n, \
        f"index length mismatch: {len(tasks)} tasks, {len(sigmas)} sigmas, expected {n}"

    for mm in (degraded_mm, clean_mm, response_mm, latent_mm):
        mm.flush()

    index = {
        "n": n, "task": tasks, "sigma": sigmas,
        "patch": PATCH, "latent_shape": [LATENT_C, LATENT_H, LATENT_W],
        "task_ids": {"denoise": 0, "derain": 1, "dehaze": 2},
    }
    (out_dir / "index.json").write_text(json.dumps(index))

    elapsed = time.time() - t0
    print(f"\nDone: {n:,} samples in {elapsed/60:.1f} min "
          f"({n/elapsed:.1f} samples/s)")
    for name in ("denoise", "derain", "dehaze"):
        tid = index["task_ids"][name]
        count = sum(1 for t in tasks if t == tid)
        print(f"  {name}: {count:,} samples")


if __name__ == "__main__":
    main()
