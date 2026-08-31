"""Off-axis rain evaluation sets — the condition under which the oriented
block's mechanism can actually express itself.

S0.1 measured that the 4-orientation bank's advantage over a cheap axis-aligned
kernel exists ONLY off-axis: at 0/90 deg a rank-2 separable kernel matches the
oracle, and at 45 deg it collapses (-0.458 dB) while the bank holds (-0.015).
Both of our corpora are near-vertical -- synthetic add_rain is U(-15,15) and
real Rain100L measures 93 +/- 13 deg. So on existing data the block is PREDICTED
to show nothing, and a null would say nothing about orientation.

These sets fix that. Rain is synthesised at controlled angles onto the 100
Rain100L *clean targets* — images no arm has trained on (Rain100L test is
disjoint from RainTrainL by content hash, verified in build_rain100L_test.py).
Content stays upright; only the streak angle varies. That is the single-variable
design: rotating whole images instead would shift content statistics too and
confound the result.

Falsifiable prediction this enables for S3.3:
    the oriented block beats the plain-k11 control on OFF-AXIS rain and ties it
    on native Rain100L. If it ties everywhere, the spatial-orientation route is
    closed -- a publishable negative, and one the plan already anticipates.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "/home/minura/FYP/Workspace/Himeth/scripts/distill")

import numpy as np
from PIL import Image

from degradations import add_rain  # noqa: E402

SRC = Path("/home/minura/fyp-adair-distill/data/test/derain/rain100L/target")
OUT = Path("/home/minura/fyp-adair-distill/data/test/derain")
ANGLES = [0.0, 22.5, 45.0, 67.5, 90.0]


def main() -> int:
    cleans = sorted(SRC.glob("*.png"))
    if not cleans:
        raise SystemExit(f"no clean images under {SRC}")
    print(f"{len(cleans)} clean source images (Rain100L test targets, unseen)")

    for ang in ANGLES:
        tag = f"rain100L_a{int(ang * 10):04d}"      # a0000, a0225, a0450, ...
        d_in, d_tg = OUT / tag / "input", OUT / tag / "target"
        d_in.mkdir(parents=True, exist_ok=True)
        d_tg.mkdir(parents=True, exist_ok=True)
        n = 0
        for p in cleans:
            clean = np.asarray(Image.open(p).convert("RGB"))
            # seed from filename+angle: order-independent and stable across
            # filesystems, the convention reports/eval_conventions.md settled on
            rng = np.random.default_rng(
                abs(hash(f"{p.name}@{ang}")) % (2 ** 31))
            deg = np.asarray(add_rain(clean, rng, angle=ang))
            if deg.shape != clean.shape:
                continue
            Image.fromarray(deg).save(d_in / p.name)
            Image.fromarray(clean).save(d_tg / p.name)
            n += 1
        # never let a set be silently empty or half-built
        assert n == len(cleans), f"{tag}: wrote {n} of {len(cleans)}"
        assert len(list(d_in.glob('*.png'))) == len(list(d_tg.glob('*.png'))) == n
        print(f"  {tag}: {n} pairs at {ang:>5.1f} deg -> {d_in.parent}")

    print("\nsanity: inputs must differ from targets (rain actually applied)")
    for ang in (ANGLES[0], ANGLES[2]):
        tag = f"rain100L_a{int(ang * 10):04d}"
        a = np.asarray(Image.open(OUT / tag / "input" / cleans[0].name), dtype=float)
        b = np.asarray(Image.open(OUT / tag / "target" / cleans[0].name), dtype=float)
        mae = float(np.abs(a - b).mean())
        print(f"  {tag}: mean|input-target| = {mae:.2f}")
        assert mae > 1.0, f"{tag}: rain layer is ~absent"
    return 0


if __name__ == "__main__":
    sys.exit(main())
