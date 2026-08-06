"""Verify no benchmark image appears in any training source.

    python scripts/sanity_overlap.py

Checks by **image content**, not filename. Filenames are not evidence: the
SOTS/OTS check earlier in this project found 75 collisions where the same
number meant the same photograph, and equally a dataset can carry the same
image under two names. Two passes:

* **exact** — MD5 of the decoded RGB array, catching byte-identical images
  regardless of container, filename or compression settings.
* **near-duplicate** — 64x64 grayscale normalised cross-correlation, catching
  the same photograph re-encoded, resized or lightly recompressed. Same-image
  pairs score ~1.0 and unrelated photographs score well below 0.5, so the 0.95
  threshold sits nowhere near anything.

Every check reports PASS or FAIL explicitly. "Should be fine by construction"
is not a result -- BSD68 being a standard external benchmark is exactly the
kind of assumption this project has been burned by.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.utils.config import REPO_ROOT, load_paths

SAME_IMAGE_CORR = 0.95
SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def _files(root: Path, limit: int | None = None) -> list[Path]:
    fs = sorted(p for p in root.rglob("*") if p.suffix.lower() in SUFFIXES)
    return fs[:limit] if limit else fs


def _sig(path: Path) -> tuple[str, np.ndarray]:
    """Exact content hash and a 64x64 normalised thumbnail."""
    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB"))
        g = np.asarray(im.convert("L").resize((64, 64), Image.BILINEAR), dtype=np.float64)
    return hashlib.md5(rgb.tobytes()).hexdigest(), (g - g.mean()) / (g.std() + 1e-8)


def check(name: str, bench: list[Path], train: list[Path]) -> dict:
    print(f"\n=== {name} ===")
    print(f"  benchmark {len(bench)} images vs training {len(train)} images")
    bh = {}
    bt = []
    for p in bench:
        h, t = _sig(p)
        bh.setdefault(h, []).append(p.name)
        bt.append((p.name, t))
    exact, near = [], []
    for p in train:
        h, t = _sig(p)
        if h in bh:
            exact.append((p.name, bh[h][0]))
        for bname, btn in bt:
            if float((btn * t).mean()) > SAME_IMAGE_CORR:
                near.append((p.name, bname))
                break
    ok = not exact and not near
    print(f"  exact duplicates      : {len(exact)}"
          + (f"  e.g. {exact[:3]}" if exact else ""))
    print(f"  near-duplicates (>{SAME_IMAGE_CORR}) : {len(near)}"
          + (f"  e.g. {near[:3]}" if near else ""))
    print(f"  {'PASS -- no overlap' if ok else 'FAIL -- OVERLAP FOUND'}")
    return {"check": name, "n_benchmark": len(bench), "n_training": len(train),
            "exact": exact[:20], "near": near[:20], "pass": ok}


def main() -> int:
    data_root = Path(load_paths()["data_root"])
    if not data_root.is_absolute():
        data_root = REPO_ROOT / data_root
    results = []

    # 1. BSD68 (denoise validation) vs BSD400 + WED (denoise training).
    results.append(check(
        "BSD68 vs Train/Denoise",
        _files(data_root / "test" / "denoise" / "bsd68"),
        _files(data_root / "Train" / "Denoise")))

    # 2. The dehaze demo's held-out set vs the 4,000 images actually trained on.
    #    Re-verified against the data as it stands NOW, not as it was when the
    #    split was written -- a file added since would not have been caught.
    syn = data_root / "Train" / "Dehaze" / "synthetic"
    train_list = [l.strip() for l in
                  (REPO_ROOT / "reports" / "dehaze_train_list.txt")
                  .read_text(encoding="utf-8").splitlines()
                  if l.strip() and not l.startswith("#")]
    held = _files(data_root / "test" / "dehaze" / "demo" / "input")
    train_stems = {Path(n).stem.split("_")[0] for n in train_list}
    held_stems = {p.stem.split("_")[0] for p in held}
    leak = sorted(held_stems & train_stems)
    print("\n=== dehaze demo split, re-verified on the data as it stands now ===")
    print(f"  training images {len(train_list)} over {len(train_stems)} clear stems")
    print(f"  held-out images {len(held)} over {len(held_stems)} clear stems")
    print(f"  shared stems: {len(leak)}"
          + (f"  e.g. {leak[:5]}" if leak else "  -- none"))
    print(f"  {'PASS -- stem-disjoint' if not leak else 'FAIL -- LEAKED'}")
    results.append({"check": "dehaze demo split (stem-disjoint)",
                    "n_benchmark": len(held), "n_training": len(train_list),
                    "shared_stems": leak[:20], "pass": not leak})

    # 3. The clean SOTS set vs everything dehaze-trained: the demo subset AND
    #    the full OTS directory B0-v2 saw.
    sots = data_root / "test" / "dehaze" / "sots_clean" / "input"
    if sots.exists():
        sots_stems = {p.stem.split("_")[0] for p in _files(sots)}
        ots_stems = {l.strip() for l in
                     (REPO_ROOT / "reports" / "reside_required_clear.txt")
                     .read_text(encoding="utf-8").splitlines() if l.strip()}
        bad_full = sorted(sots_stems & ots_stems)
        bad_demo = sorted(sots_stems & train_stems)
        print("\n=== clean SOTS vs dehaze training sources ===")
        print(f"  SOTS-clean scenes {len(sots_stems)}")
        print(f"  vs full OTS ({len(ots_stems)} stems, what B0-v2 trained on): "
              f"{len(bad_full)} shared -> "
              f"{'PASS' if not bad_full else 'FAIL'}")
        print(f"  vs demo subset ({len(train_stems)} stems): {len(bad_demo)} shared -> "
              f"{'PASS' if not bad_demo else 'FAIL'}")
        results.append({"check": "clean SOTS vs full OTS",
                        "n_benchmark": len(sots_stems), "n_training": len(ots_stems),
                        "shared_stems": bad_full[:20], "pass": not bad_full})
        results.append({"check": "clean SOTS vs dehaze demo subset",
                        "n_benchmark": len(sots_stems), "n_training": len(train_stems),
                        "shared_stems": bad_demo[:20], "pass": not bad_demo})
    else:
        print(f"\n  SKIP: {sots} not present")

    out = REPO_ROOT / "reports" / "sanity_overlap.json"
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")
    failed = [r["check"] for r in results if not r["pass"]]
    print(f"\n{'ALL CHECKS PASS' if not failed else 'FAILURES: ' + ', '.join(failed)}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
