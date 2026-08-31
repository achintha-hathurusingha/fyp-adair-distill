"""Install the REAL Rain100L test split as data/test/derain/rain100L/.

Why this exists
---------------
`data/test/derain/demo` is 40 images carved out of `Train/Derain` by
`make_derain_split.py`. That is only safe if training then EXCLUDES them via the
`list:` key -- which the single-task demo configs do and the 3-task configs do
NOT. So every 3-task arm has been scored on 40 images it trained on (verified:
all 40 are byte-identical to training files).

The published Rain100L test split (100 pairs) already exists on this machine and
is fully disjoint from our 200 training images (verified: 0 shared content
hashes out of 200 vs 200). Using it fixes the leak WITHOUT sacrificing training
data, gives 100 eval images instead of 40, and makes our derain number directly
comparable to AdaIR's published 38.64 dB.

Note the filename hazard this whole mess came from: Rain100L's test inputs are
named rain-001.png..rain-100.png and our TRAINING files are also named
rain-001.png..rain-200.png -- same names, different images. Names cannot be used
to reason about this dataset; only content hashes can.

Layout produced (the harness's input/ + target/ convention, matching names):
    data/test/derain/rain100L/input/rain-NNN.png   <- Rain100L/rainy/rain-NNN.png
    data/test/derain/rain100L/target/rain-NNN.png  <- Rain100L/norain-NNN.png
"""
from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

SRC = Path("/home/minura/FYP/Workspace/Minura/AdaIR/data/rain100L/rain100L_test/Rain100L")
DST = Path("/home/minura/fyp-adair-distill/data/test/derain/rain100L")
TRAIN = Path("/home/minura/fyp-adair-distill/data/Train/Derain/input")


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def main() -> int:
    rainy = SRC / "rainy"
    if not rainy.is_dir():
        raise SystemExit(f"missing {rainy}")

    inp_d, tgt_d = DST / "input", DST / "target"
    inp_d.mkdir(parents=True, exist_ok=True)
    tgt_d.mkdir(parents=True, exist_ok=True)

    pairs, missing = [], []
    for r in sorted(rainy.glob("rain-*.png")):
        n = r.name.split("rain-")[1]
        clean = SRC / f"norain-{n}"
        if not clean.exists():
            missing.append(clean.name)
            continue
        pairs.append((r, clean))
    if missing:
        raise SystemExit(f"{len(missing)} rainy images have no clean pair, "
                         f"e.g. {missing[:3]}")

    for r, c in pairs:
        shutil.copy2(r, inp_d / r.name)
        shutil.copy2(c, tgt_d / r.name)   # target renamed to MATCH the input
    print(f"installed {len(pairs)} pairs -> {DST}")

    # --- verification, not assumption -------------------------------------
    n_in = len(list(inp_d.glob("*.png")))
    n_tg = len(list(tgt_d.glob("*.png")))
    assert n_in == n_tg == len(pairs), f"count mismatch {n_in}/{n_tg}"
    print(f"  input {n_in}, target {n_tg}, 1:1 names verified")

    # input and target must NOT be identical (that would mean we copied the
    # clean image into both sides and PSNR would read ~99 dB)
    same = sum(1 for p in inp_d.glob("*.png") if md5(p) == md5(tgt_d / p.name))
    assert same == 0, f"{same} pairs have identical input and target"
    print(f"  input != target for all {n_in} pairs")

    # the whole point: zero content overlap with the training set
    train_h = {md5(p) for p in TRAIN.glob("*.png")}
    leak_in = sum(1 for p in inp_d.glob("*.png") if md5(p) in train_h)
    leak_tg = sum(1 for p in tgt_d.glob("*.png") if md5(p) in train_h)
    print(f"  overlap with Train/Derain: inputs {leak_in}, targets {leak_tg} "
          f"(of {len(train_h)} training hashes)")
    assert leak_in == 0 and leak_tg == 0, "LEAK: test content found in training set"
    print("  PASS - disjoint from training data")
    return 0


if __name__ == "__main__":
    sys.exit(main())
