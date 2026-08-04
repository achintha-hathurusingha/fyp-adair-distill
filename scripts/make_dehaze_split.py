"""Seeded train / held-out split of RESIDE-OTS for the dehaze gap demo.

    python scripts/make_dehaze_split.py --train-pairs 4000 --heldout-images 150

**Splits by CLEAR SOURCE STEM, not by hazy file.** RESIDE-OTS renders roughly 35
hazy variants from each of its 2,061 clear images (`0025_0.8_0.04.jpg`,
`0025_0.9_0.20.jpg`, ... all from `0025.png`). Splitting on hazy filenames would
put different hazings of the *same scene* on both sides of the split, and the
held-out score would then measure memorisation of scenes the model had already
seen. The stem is the unit of independence, so it is the unit of the split.

Writes three things, all recorded so the run is repeatable:

* ``reports/dehaze_train_list.txt``  — hazy paths relative to `synthetic/`
* ``reports/dehaze_heldout_list.txt`` — the held-out hazy paths
* ``data/test/dehaze/demo/{input,target}/`` — symlinks for the held-out set, so
  the existing ``PairedTestDataset`` reads it with no code change

The seed and every count are printed and written into the list headers.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from src.data.datasets import resolve_pair_target
from src.utils.config import REPO_ROOT, load_paths


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--train-pairs", type=int, default=4000,
                    help="hazy images in the training subset")
    ap.add_argument("--heldout-images", type=int, default=150,
                    help="hazy images in the held-out evaluation set")
    ap.add_argument("--heldout-stems", type=int, default=150,
                    help="clear sources reserved for held-out (one hazy each)")
    args = ap.parse_args()

    data_root = Path(load_paths()["data_root"])
    if not data_root.is_absolute():
        data_root = REPO_ROOT / data_root
    syn = data_root / "Train" / "Dehaze" / "synthetic"
    clear = data_root / "Train" / "Dehaze" / "clear"
    for d in (syn, clear):
        if not d.exists():
            raise SystemExit(f"missing {d} — run scripts/reside_manifest.py verify")

    files = sorted(p for p in syn.rglob("*") if p.suffix.lower() == ".jpg")
    by_stem: dict[str, list[Path]] = {}
    for p in files:
        by_stem.setdefault(p.stem.split("_")[0], []).append(p)
    stems = sorted(by_stem)
    print(f"source: {len(files):,} hazy images from {len(stems):,} clear stems "
          f"({len(files) / len(stems):.1f} per stem)")

    rng = random.Random(args.seed)
    shuffled = stems[:]
    rng.shuffle(shuffled)
    if args.heldout_stems >= len(shuffled):
        raise SystemExit(f"--heldout-stems {args.heldout_stems} leaves no training data")
    held_stems = sorted(shuffled[:args.heldout_stems])
    train_stems = sorted(shuffled[args.heldout_stems:])
    assert not (set(held_stems) & set(train_stems)), "stem split overlaps"

    # Held-out: one hazy rendering per reserved stem, so the set spans 150
    # distinct scenes rather than 150 hazings of a handful.
    held = []
    for s in held_stems[:args.heldout_images]:
        held.append(rng.choice(sorted(by_stem[s])))
    # Train: sample hazy files from the training stems only.
    pool = [p for s in train_stems for p in by_stem[s]]
    rng.shuffle(pool)
    train = sorted(pool[:args.train_pairs])

    out_dir = REPO_ROOT / "reports"
    header = (f"# RESIDE-OTS dehaze demo split, seed={args.seed}\n"
              f"# stems: {len(train_stems):,} train / {len(held_stems)} held-out, "
              f"disjoint by clear source\n")
    for name, items in (("dehaze_train_list.txt", train),
                        ("dehaze_heldout_list.txt", held)):
        rel = [p.relative_to(syn).as_posix() for p in items]
        (out_dir / name).write_text(
            header + f"# {len(rel):,} hazy images\n" + "\n".join(rel) + "\n",
            encoding="utf-8")
        print(f"wrote reports/{name}  ({len(rel):,} images)")

    # Held-out directory in the input/ + target/ shape PairedTestDataset expects.
    demo = data_root / "test" / "dehaze" / "demo"
    for sub in ("input", "target"):
        d = demo / sub
        d.mkdir(parents=True, exist_ok=True)
        for old in d.iterdir():
            old.unlink()
    n_pairs = 0
    for p in held:
        tgt = resolve_pair_target(p, clear, "dehaze")
        (demo / "input" / p.name).symlink_to(p)
        # Target keeps the CLEAR STEM name, so PairedTestDataset resolves it by
        # the real dehaze rule (name.split('_')[0]) rather than a special case
        # invented for this directory.
        link = demo / "target" / f"{p.stem.split('_')[0]}{tgt.suffix}"
        if not link.exists():
            link.symlink_to(tgt)
        n_pairs += 1
    print(f"wrote {demo} ({n_pairs} pairs, symlinked)")

    # A held-out image whose stem appears in training is the failure this split
    # exists to prevent, so assert it rather than trusting the logic above.
    train_stem_set = {p.stem.split('_')[0] for p in train}
    leaked = sorted({p.stem.split('_')[0] for p in held} & train_stem_set)
    if leaked:
        raise SystemExit(f"LEAK: {len(leaked)} held-out stems appear in training, "
                         f"e.g. {leaked[:5]}")
    print(f"\nno leakage: {len(held)} held-out scenes share no clear source with "
          f"the {len(train):,} training images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
