"""Build the SOTS-outdoor subset that no OTS-trained model has seen.

    python scripts/make_sots_clean.py

**SOTS-outdoor is not automatically a clean test set for a model trained on
OTS.** They are the published test and train splits of RESIDE, but they are not
disjoint: 75 of SOTS-outdoor's 492 clear scenes also appear among OTS's 2,061
clear sources, and the overlap is by *content*, not merely by filename —
verified at 64x64 grayscale normalised cross-correlation, all 75 at 1.0000.

This matters because B0-v2 trained on the entire OTS synthetic directory, so
evaluating it on full SOTS-outdoor would score 75 scenes it has memorised. The
single-task demo students drew their 4,000 images from OTS stems too, so the
same exclusion covers them.

The result is a 417-scene subset that neither B0-v2 nor the demo students have
seen, materialised as ``input/`` + ``target/`` so ``PairedTestDataset`` reads it
unchanged.

**A caveat worth stating rather than burying:** published SOTS numbers in the
dehazing literature are computed on all 500 hazy images, so figures measured
here are NOT directly comparable to them. They are comparable to each other,
which is what the capacity question needs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from src.utils.config import REPO_ROOT, load_paths

#: Correlation above which two images are treated as the same scene. Same-image
#: pairs score 1.0000 and unrelated outdoor photographs score well below 0.5, so
#: the threshold is not near anything.
SAME_IMAGE_CORR = 0.95


def _thumb(path: Path, n: int = 64) -> np.ndarray:
    with Image.open(path) as im:
        a = np.asarray(im.convert("L").resize((n, n), Image.BILINEAR), dtype=np.float64)
    return (a - a.mean()) / (a.std() + 1e-8)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ots-clear", type=Path, default=None,
                    help="OTS clear directory; defaults under data_root")
    ap.add_argument("--verify-content", action="store_true",
                    help="confirm name collisions are the same IMAGE, not just "
                         "the same number (needs both directories present)")
    args = ap.parse_args()

    data_root = Path(load_paths()["data_root"])
    if not data_root.is_absolute():
        data_root = REPO_ROOT / data_root
    sots = data_root / "dehaze" / "RESIDE" / "SOTS" / "outdoor"
    if not (sots / "target").exists():
        raise SystemExit(f"missing {sots}; run scripts/prepare_datasets.py first")

    scenes = sorted(p.stem for p in (sots / "target").iterdir())
    ots_stems = {l.strip() for l in
                 (REPO_ROOT / "reports" / "reside_required_clear.txt")
                 .read_text(encoding="utf-8").splitlines() if l.strip()}
    overlap = sorted(set(scenes) & ots_stems)
    print(f"SOTS-outdoor scenes : {len(scenes)}")
    print(f"OTS clear stems     : {len(ots_stems)}")
    print(f"name collisions     : {len(overlap)}")

    if args.verify_content:
        ots_clear = args.ots_clear or (data_root / "Train" / "Dehaze" / "clear")
        if not ots_clear.exists():
            raise SystemExit(f"--verify-content needs {ots_clear}")
        same = 0
        for stem in overlap:
            a = next(ots_clear.glob(f"{stem}.*"), None)
            b = next((sots / "target").glob(f"{stem}.*"), None)
            if a and b and float((_thumb(a) * _thumb(b)).mean()) > SAME_IMAGE_CORR:
                same += 1
        print(f"confirmed same image: {same} / {len(overlap)}")
        if same != len(overlap):
            print("  NOTE: some collisions are different images; excluding them "
                  "anyway is conservative and costs only test-set size")

    keep = [s for s in scenes if s not in set(overlap)]
    dest = data_root / "test" / "dehaze" / "sots_clean"
    for sub in ("input", "target"):
        d = dest / sub
        d.mkdir(parents=True, exist_ok=True)
        for old in d.iterdir():
            old.unlink()

    kept_hazy = 0
    for hazy in sorted((sots / "input").iterdir()):
        if hazy.stem.split("_")[0] not in set(keep):
            continue
        (dest / "input" / hazy.name).symlink_to(hazy)
        kept_hazy += 1
    for stem in keep:
        tgt = next((sots / "target").glob(f"{stem}.*"))
        (dest / "target" / tgt.name).symlink_to(tgt)

    print(f"\nkept {len(keep)} scenes / {kept_hazy} hazy images "
          f"(excluded {len(overlap)} scenes that OTS-trained models have seen)")
    print(f"wrote {dest}")
    print("\nPublished SOTS figures use all 500 hazy images, so numbers measured "
          "here are NOT comparable to the literature -- only to each other.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
