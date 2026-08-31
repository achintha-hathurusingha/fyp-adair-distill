"""One-time precompute: cache the frozen teacher's (response, latent_pre)
for a large finite pool of (degraded, clean) samples drawn from the exact
same live per-task sampling logic training already uses. See
reports/kd_feature_multitask/plan_cached_teacher.md.

Why: profile_step_cost.py found the teacher's forward pass is ~78% of every
training step's wall-clock time, paid fresh on every sample because live
training draws a fresh random crop (and, for denoise, a fresh continuous
sigma) every single time -- there is no repetition to exploit online. This
script converts that into a one-time cost: draw N samples once, run the
teacher once each, cache the result, then train by reading the cache.

Storage: one set of memmap files, TASK-CONTIGUOUS (all denoise rows, then
all derain, then all dehaze) rather than interleaved -- this is what lets
CachedTeacherDataset reuse `BalancedTaskBatchSampler` directly (it works
from a `{task: range}` map, needing genuinely contiguous per-task ranges,
not just an index column to filter on):
  degraded.dat    (N, 3, 128, 128) uint8   -- CHW, matches to_tensor() layout
  clean.dat       (N, 3, 128, 128) uint8
  response.dat    (N, 3, 128, 128) uint8   -- teacher's output, same domain
  latent_pre.dat  (N, 384, 16, 16) float16 -- half precision (numpy has no
                                               bf16; float16's range is ample
                                               for these activation magnitudes)
  index.json      -- {"task_ranges": {task: [start, stop]}, "sigma": [...],
                       "n": N, ...}  -- sigma is a flat per-row list (only
                       meaningful for denoise rows, -1.0 elsewhere)

Usage:
    python scripts/build_teacher_cache.py --out-dir /path/to/cache
    (per-task counts are the plan's own sizing -- see TASK_POOL_SIZES below;
    override with --denoise/--derain/--dehaze for a smoke run)
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

#: Per-task pool sizes from plan_cached_teacher.md's own sizing table.
TASK_POOL_SIZES = {"denoise": 80_000, "derain": 20_000, "dehaze": 80_000}

#: `data:` section of the training config this cache is being built FOR.
#: Populated in main() from --config; the cache is only valid for arms
#: whose data section matches it.
_DATA_CFG: dict = {}


def build_task_loader(task: str, source, n_rows: int, batch_size: int,
                      num_workers: int, seed: int):
    """A single-task-only pass through build_multitask_loader -- same
    per-task sampling code path live training uses (paired_transform, the
    F10 continuous-sigma fix for denoise, etc.), just restricted to one
    task so its rows land in one contiguous range."""
    sources = {task: source}
    kwargs = dict(batch_size=batch_size, patch_size=PATCH,
                 num_workers=num_workers, seed=seed, length=n_rows)
    if task == "denoise":
        # Sampling now MIRRORS the training config rather than being hardcoded.
        # v1 hardcoded sigma_range=(0,55)+clean_prob=0.05 while every current arm
        # uses discrete [15,25,50]; a cache built that way trains an arm on a
        # different noise distribution from its own control. See the module
        # docstring.
        if _DATA_CFG.get("sigma_range") is not None:
            kwargs["sigma_range"] = tuple(_DATA_CFG["sigma_range"])
            kwargs["clean_prob"] = float(_DATA_CFG.get("clean_prob", 0.0))
        else:
            kwargs["sigmas"] = tuple(_DATA_CFG.get("sigmas", (15, 25, 50)))
    return build_multitask_loader(sources, **kwargs)


def fill_rows(loader, teacher, degraded_mm, clean_mm, response_mm, latent_mm,
             sigmas: list, offset: int, n_rows: int, device: str,
             batch_size: int, label: str) -> None:
    """Fills memmap rows [offset, offset+n_rows) from `loader`, looping past
    any single pass (length= is not an exact guarantee -- confirmed by an
    earlier smoke test, came up 1 batch short) until exactly n_rows land."""
    row = 0
    t0 = time.time()
    loader_iter = iter(loader)
    with torch.no_grad():
        while row < n_rows:
            try:
                degraded, clean, prov = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                continue
            b = min(degraded.shape[0], n_rows - row)
            degraded, clean = degraded[:b], clean[:b]
            degraded_dev = degraded.to(device)

            soft, latent = teacher.forward_with_latent(degraded_dev.float())

            r = offset + row
            degraded_mm[r:r + b] = (degraded * 255).round().clamp(0, 255) \
                .byte().cpu().numpy()
            clean_mm[r:r + b] = (clean * 255).round().clamp(0, 255) \
                .byte().cpu().numpy()
            response_mm[r:r + b] = (soft[:b].clamp(0, 1) * 255).round() \
                .clamp(0, 255).byte().cpu().numpy()
            latent_mm[r:r + b] = latent[:b].to(torch.float16).cpu().numpy()
            sigmas[r:r + b] = [float(s) for s in prov["sigma"][:b].tolist()]

            row += b
            if row % (batch_size * 50) < batch_size:
                elapsed = time.time() - t0
                rate = row / elapsed if elapsed > 0 else 0
                eta = (n_rows - row) / rate if rate > 0 else float("inf")
                print(f"  [{label}] {row:,}/{n_rows:,}  "
                      f"({rate:.1f} samples/s, ETA {eta/60:.1f} min)")
    assert row == n_rows, f"[{label}] expected {n_rows} rows, got {row}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--config", required=True,
                    help="training config whose data: section this cache "
                         "must mirror; recorded in index.json")
    ap.add_argument("--denoise", type=int, default=TASK_POOL_SIZES["denoise"])
    ap.add_argument("--derain", type=int, default=TASK_POOL_SIZES["derain"])
    ap.add_argument("--dehaze", type=int, default=TASK_POOL_SIZES["dehaze"])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    global _DATA_CFG
    _cfg = yaml.safe_load(Path(args.config).read_text())
    _DATA_CFG = _cfg.get("data", {}) or {}
    print(f"[cache] mirroring data: from {args.config}")
    print(f"[cache]   sigmas={_DATA_CFG.get('sigmas')} "
          f"sigma_range={_DATA_CFG.get('sigma_range')} "
          f"clean_prob={_DATA_CFG.get('clean_prob')}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pool_sizes = {"denoise": args.denoise, "derain": args.derain, "dehaze": args.dehaze}
    n = sum(pool_sizes.values())
    print(f"Building teacher cache: {pool_sizes} (total {n:,}) -> {out_dir}")

    paths = yaml.safe_load(Path("configs/paths.local.yaml").read_text())
    data_root = Path(paths["data_root"])
    sources = {
        "denoise": data_root / paths["datasets"]["denoise_train"],
        "derain": data_root / paths["datasets"]["derain_train"],
        "dehaze": {"input": data_root / "Train/Dehaze/synthetic",
                  "target": data_root / "Train/Dehaze/clear"},
    }

    teacher = load_teacher(teacher_checkpoint("all_in_one"), device=args.device)

    degraded_mm = np.memmap(out_dir / "degraded.dat", dtype=np.uint8, mode="w+",
                            shape=(n, 3, PATCH, PATCH))
    clean_mm = np.memmap(out_dir / "clean.dat", dtype=np.uint8, mode="w+",
                         shape=(n, 3, PATCH, PATCH))
    response_mm = np.memmap(out_dir / "response.dat", dtype=np.uint8, mode="w+",
                            shape=(n, 3, PATCH, PATCH))
    latent_mm = np.memmap(out_dir / "latent_pre.dat", dtype=np.float16, mode="w+",
                          shape=(n, LATENT_C, LATENT_H, LATENT_W))
    sigmas: list[float] = [-1.0] * n

    t0 = time.time()
    task_ranges: dict[str, list[int]] = {}
    offset = 0
    for task in ("denoise", "derain", "dehaze"):
        n_rows = pool_sizes[task]
        loader = build_task_loader(task, sources[task], n_rows,
                                   args.batch_size, args.num_workers, args.seed)
        fill_rows(loader, teacher, degraded_mm, clean_mm, response_mm, latent_mm,
                 sigmas, offset, n_rows, args.device, args.batch_size, task)
        task_ranges[task] = [offset, offset + n_rows]
        offset += n_rows

    assert offset == n
    for mm in (degraded_mm, clean_mm, response_mm, latent_mm):
        mm.flush()

    index = {
        "n": n, "task_ranges": task_ranges, "sigma": sigmas,
        "built_for_config": args.config,
        "data_cfg": {k: _DATA_CFG.get(k) for k in
                     ("sigmas", "sigma_range", "clean_prob", "patch_size",
                      "tasks", "mixed_task")},
        "patch": PATCH, "latent_shape": [LATENT_C, LATENT_H, LATENT_W],
        "task_ids": {"denoise": 0, "derain": 1, "dehaze": 2},
    }
    (out_dir / "index.json").write_text(json.dumps(index))

    elapsed = time.time() - t0
    print(f"\nDone: {n:,} samples in {elapsed/60:.1f} min "
          f"({n/elapsed:.1f} samples/s)")
    for task, (a, b) in task_ranges.items():
        print(f"  {task}: rows [{a}, {b}) = {b - a:,} samples")


if __name__ == "__main__":
    main()
