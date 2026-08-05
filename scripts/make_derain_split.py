"""Seeded train / held-out split of Rain100L for the derain gap demo.

    python scripts/make_derain_split.py --heldout 40

**Splits by filename, and that is safe here — unlike dehaze.** RESIDE-OTS renders
~35 hazy variants from each clear source, so a filename split would put the same
scene on both sides and the held-out score would measure memorisation; that is
why `make_dehaze_split.py` splits by clear stem. Rain100L is strict 1:1 —
verified below, not assumed — so every file is its own scene and a filename
split has no leakage path. The check is performed each run rather than trusted,
because the whole argument rests on it.

**The set is small, and that is a real limitation.** 200 pairs total. Reserving
a held-out set costs training data that the protocol expects to use, and leaves
an evaluation set whose per-image variance matters. Both are reported rather
than hidden; see reports/report_demo_derain.md.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from src.utils.config import REPO_ROOT, load_paths


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--heldout", type=int, default=40)
    args = ap.parse_args()

    data_root = Path(load_paths()["data_root"])
    if not data_root.is_absolute():
        data_root = REPO_ROOT / data_root
    root = data_root / "Train" / "Derain"
    inp, tgt = root / "input", root / "target"
    for d in (inp, tgt):
        if not d.exists():
            raise SystemExit(f"missing {d}")

    inputs = sorted(p.name for p in inp.iterdir() if p.suffix.lower() == ".png")
    targets = {p.name for p in tgt.iterdir() if p.suffix.lower() == ".png"}

    # The claim that a filename split is safe rests entirely on 1:1 pairing.
    # Verify it; do not assume it.
    missing = [n for n in inputs if n not in targets]
    if missing:
        raise SystemExit(f"{len(missing)} inputs have no target, e.g. {missing[:3]}")
    stems = {n.rsplit(".", 1)[0] for n in inputs}
    if len(stems) != len(inputs):
        raise SystemExit(
            f"{len(inputs)} inputs share only {len(stems)} stems — Rain100L is "
            "not 1:1 here, so a filename split WOULD leak. Use stem grouping, "
            "as make_dehaze_split.py does.")
    print(f"Rain100L: {len(inputs)} pairs, 1:1 verified "
          f"({len(stems)} unique stems, every input has a target)")

    if args.heldout >= len(inputs):
        raise SystemExit(f"--heldout {args.heldout} leaves no training data")

    rng = random.Random(args.seed)
    shuffled = inputs[:]
    rng.shuffle(shuffled)
    held = sorted(shuffled[:args.heldout])
    train = sorted(shuffled[args.heldout:])
    assert not (set(held) & set(train))

    header = (f"# Rain100L derain demo split, seed={args.seed}\n"
              f"# {len(train)} train / {len(held)} held-out, disjoint by filename\n"
              f"# 1:1 pairing verified at split time — no multi-variant leakage\n")
    out = REPO_ROOT / "reports"
    (out / "derain_train_list.txt").write_text(
        header + "\n".join(train) + "\n", encoding="utf-8")
    print(f"wrote reports/derain_train_list.txt ({len(train)} pairs)")

    demo = data_root / "test" / "derain" / "demo"
    for sub in ("input", "target"):
        d = demo / sub
        d.mkdir(parents=True, exist_ok=True)
        for old in d.iterdir():
            old.unlink()
    for n in held:
        (demo / "input" / n).symlink_to(inp / n)
        (demo / "target" / n).symlink_to(tgt / n)
    print(f"wrote {demo} ({len(held)} pairs, symlinked)")

    print(f"\ntraining on {len(train)} of {len(inputs)} pairs — the held-out set "
          f"costs {args.heldout} pairs the protocol would otherwise train on")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
