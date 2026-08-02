"""Validate the multi-task loader on REAL data before committing GPU time.

Unit tests use ~13 synthetic images. This runs the actual loader over the actual
training set and checks the properties that a wrong answer would cost days to
discover:

1. **Task balance** — every batch really does hold an equal share, over hundreds
   of batches, on real file counts (5,144 denoise vs 200 derain pairs). This is
   the check that F11 would have failed.
2. **Worker-count independence** — identical batches at different
   ``num_workers``. Proven for the denoise loader by ``determinism_check.py``;
   re-proven here because the multi-task index space is new.
3. **Noise coverage** — the sampled sigma distribution actually reaches below
   15, which is the region B0-denoise never saw (F10).
4. **Pairing** — every derain/dehaze input resolved to a target at construction.

Dehaze is included automatically when its directory exists; until RESIDE-OTS
lands this validates two tasks, and the third slots in with no code change.

    python scripts/validate_multitask.py
    python scripts/validate_multitask.py --batches 500 --workers 6 12
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import torch
import yaml

from src.data.build import TASK_IDS, build_multitask_loader
from src.utils.config import REPO_ROOT, load_paths

CONFIG = REPO_ROOT / "configs" / "train" / "b0v2_multitask.yaml"


def _resolve_sources(cfg: dict, data_root: Path) -> tuple[dict[str, Path], list[str]]:
    """Task roots that exist, and the names of those that do not."""
    present, missing = {}, []
    for task, rel in cfg["data"]["tasks"].items():
        root = data_root / rel
        (present.__setitem__(task, root) if root.exists() else missing.append(task))
    return present, missing


def check_balance(loader, n_batches: int) -> bool:
    """Every batch holds an equal share of every task, +/- the rotating extra."""
    per_batch, totals = [], Counter()
    for i, (_deg, _clean, meta) in enumerate(loader):
        if i >= n_batches:
            break
        counts = Counter(int(t) for t in meta["task"])
        per_batch.append(counts)
        totals.update(counts)

    names = {v: k for k, v in TASK_IDS.items()}
    spread = {max(c.values()) - min(c.values()) for c in per_batch}
    ok = spread <= {0, 1} and len(per_batch) == n_batches
    print(f"  batches inspected : {len(per_batch)}")
    print(f"  within-batch spread: {sorted(spread)}  (0 or 1 expected)")
    for tid, n in sorted(totals.items()):
        print(f"  {names[tid]:<8}: {n:>7,} samples "
              f"({100 * n / sum(totals.values()):.1f}%)")
    share = max(totals.values()) - min(totals.values())
    ok = ok and share <= len(per_batch)
    print(f"  epoch imbalance   : {share} samples "
          f"(<= {len(per_batch)} allowed, one per batch)")
    return ok


def check_worker_independence(sources, worker_counts, n_batches, **kw) -> bool:
    """Different worker counts must give byte-identical batches."""
    refs = None
    for w in worker_counts:
        loader = build_multitask_loader(sources, num_workers=w, **kw)
        got = []
        for i, (deg, clean, meta) in enumerate(loader):
            if i >= n_batches:
                break
            got.append((deg.clone(), clean.clone(), meta["task"].clone()))
        if refs is None:
            refs = got
            print(f"  workers {w:>2}: reference ({len(got)} batches)")
            continue
        same = all(torch.equal(a, b) for r, g in zip(refs, got) for a, b in zip(r, g))
        print(f"  workers {w:>2}: {'identical' if same else 'DIFFERS'}")
        if not same:
            return False
    return True


def check_sigma_coverage(loader, n_batches: int) -> bool:
    """The F10 region -- sigma below 15 -- must actually be sampled."""
    sigmas = []
    for i, (_d, _c, meta) in enumerate(loader):
        if i >= n_batches:
            break
        sigmas += [float(s) for s, t in zip(meta["sigma"], meta["task"])
                   if int(t) == TASK_IDS["denoise"]]
    if not sigmas:
        print("  no denoise samples seen")
        return False
    below = sum(1 for s in sigmas if s < 15.0)
    zeros = sum(1 for s in sigmas if s == 0.0)
    print(f"  denoise samples   : {len(sigmas):,}")
    print(f"  range             : {min(sigmas):.2f} .. {max(sigmas):.2f}")
    print(f"  below sigma 15    : {below:,} ({100 * below / len(sigmas):.1f}%) "
          "<- the region B0-denoise never saw")
    print(f"  exactly sigma 0   : {zeros:,} ({100 * zeros / len(sigmas):.1f}%)")
    return below > 0 and zeros > 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--batches", type=int, default=300)
    ap.add_argument("--workers", type=int, nargs="+", default=[0, 4],
                    help="worker counts to compare for determinism")
    ap.add_argument("--det-batches", type=int, default=12,
                    help="batches compared across worker counts (slow path)")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data_root = Path(load_paths()["data_root"])
    if not data_root.is_absolute():
        data_root = REPO_ROOT / data_root
    sources, missing = _resolve_sources(cfg, data_root)
    if not sources:
        print("no task roots exist; nothing to validate", file=sys.stderr)
        return 2

    d = cfg["data"]
    kw = dict(batch_size=d["batch_size"], patch_size=d["patch_size"],
              sigmas=tuple(d["sigmas"]),
              sigma_range=tuple(d["sigma_range"]) if d.get("sigma_range") else None,
              clean_prob=d.get("clean_prob", 0.0), seed=0,
              length=args.batches * d["batch_size"] * 2,
              cache_budget_gb=d["cache_budget_gb"])

    print(f"config      : {args.config.relative_to(REPO_ROOT)}")
    print(f"tasks       : {', '.join(sources)}")
    if missing:
        print(f"NOT PRESENT : {', '.join(missing)} -- validating without it")
    print(f"batch size  : {d['batch_size']}   patch {d['patch_size']}")
    print()

    results = {}
    loader = build_multitask_loader(sources, num_workers=0, **kw)
    ds = loader.dataset
    print("[pairing] resolved at construction, no missing targets")
    for task, items in ds.items.items():
        slots = len(ds.task_ranges()[task])
        # Equal index slots across tasks means a small task repeats. AdaIR
        # achieves the same effect with hand-tuned multipliers (derain x120);
        # ours falls out of the layout, so print it rather than leave it implied.
        print(f"  {task:<8}: {len(items):>6,} items -> {slots:,} index slots "
              f"({slots / len(items):.1f}x per item)")
    print()

    print("[1] task balance")
    results["balance"] = check_balance(loader, args.batches)
    print()

    print("[2] worker-count independence")
    results["determinism"] = check_worker_independence(
        sources, args.workers, args.det_batches, **kw)
    print()

    print("[3] sigma coverage (F10)")
    results["sigma"] = check_sigma_coverage(
        build_multitask_loader(sources, num_workers=0, **kw), args.batches)
    print()

    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
