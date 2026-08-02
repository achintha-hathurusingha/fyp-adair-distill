"""Work out exactly which RESIDE files are needed, and verify them on arrival.

    python scripts/reside_manifest.py plan     # what to download, and how much
    python scripts/reside_manifest.py verify   # check what arrived

WHY A MANIFEST. RESIDE in full is very large, and most of it is unused. AdaIR
does not consume the dataset wholesale — it reads an explicit file list,
`third_party/AdaIR/data_dir/hazy/hazy_outside.txt`, which selects the OTS
*synthetic* subset only. Downloading to that list instead of the whole archive is
the difference between tens of gigabytes and a few.

The same directory also carries `noisy/denoise.txt` and `rainy/rainTrain.txt`,
and both already match what we hold exactly (5,144 denoise files; 199 derain
entries against our 200 pairs) — which is good evidence the lists are the right
authority for the dehaze set too.

PAIRING. A hazy filename encodes its source: `0025_0.8_0.04.jpg` derives from
clear image `0025`. This is the same `name.split('_')[0]` rule
`PairedTestDataset._target_for` already uses for dehaze evaluation, so training
and evaluation resolve pairs identically.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from src.utils.config import REPO_ROOT, load_paths

HAZY_LIST = REPO_ROOT / "third_party" / "AdaIR" / "data_dir" / "hazy" / "hazy_outside.txt"
#: Typical OTS synthetic JPEG, measured from the published dataset description.
#: Used only to size the download; `verify` reports the real figure.
EST_HAZY_KB = 60
EST_CLEAR_KB = 110


def _entries() -> list[str]:
    if not HAZY_LIST.exists():
        raise FileNotFoundError(
            f"{HAZY_LIST} missing — is third_party/AdaIR vendored?")
    return [ln.strip() for ln in HAZY_LIST.read_text().splitlines() if ln.strip()]


def _clear_stem(entry: str) -> str:
    """`synthetic/part1/0025_0.8_0.04.jpg` -> `0025`."""
    return Path(entry).stem.split("_")[0]


def plan() -> None:
    entries = _entries()
    stems = {_clear_stem(e) for e in entries}
    parts = Counter(Path(e).parent.as_posix() for e in entries)

    print(f"hazy entries required : {len(entries):,}")
    print(f"unique clear sources  : {len(stems):,}")
    print("\nby sub-directory:")
    for d, n in sorted(parts.items()):
        print(f"  {d:<24} {n:>8,}")

    hazy_gb = len(entries) * EST_HAZY_KB / 1e6
    clear_gb = len(stems) * EST_CLEAR_KB / 1e6
    print(f"\nESTIMATED size (verify on arrival, do not trust this):")
    print(f"  hazy   ~{hazy_gb:5.1f} GB   ({len(entries):,} x ~{EST_HAZY_KB} KB)")
    print(f"  clear  ~{clear_gb:5.1f} GB   ({len(stems):,} x ~{EST_CLEAR_KB} KB)")
    print(f"  total  ~{hazy_gb + clear_gb:5.1f} GB")
    print("\nThis is the OTS *synthetic* subset only — NOT all of RESIDE, and NOT")
    print("ITS. Earlier estimates in this project of 45-100 GB were wrong: they")
    print("assumed the whole dataset rather than the list AdaIR actually reads.")

    paths = load_paths()
    root = Path(paths["data_root"])
    if not root.is_absolute():
        root = REPO_ROOT / root
    print(f"\nEXPECTED LAYOUT under {root}:")
    print("  data/Train/Dehaze/synthetic/part1..4/   <- hazy images from the list")
    print("  data/Train/Dehaze/clear/                <- the clear sources")
    print("\nThen: python scripts/reside_manifest.py verify")

    out = REPO_ROOT / "reports" / "reside_required_files.txt"
    out.write_text("\n".join(entries) + "\n", encoding="utf-8")
    stem_out = REPO_ROOT / "reports" / "reside_required_clear.txt"
    stem_out.write_text("\n".join(sorted(stems)) + "\n", encoding="utf-8")
    print(f"\nwrote {out.relative_to(REPO_ROOT)} ({len(entries):,} lines)")
    print(f"wrote {stem_out.relative_to(REPO_ROOT)} ({len(stems):,} lines)")


def verify() -> None:
    """Check what actually arrived. Fails loudly, never silently."""
    paths = load_paths()
    root = Path(paths["data_root"])
    if not root.is_absolute():
        root = REPO_ROOT / root
    dehaze = root / "Train" / "Dehaze"
    if not dehaze.exists():
        raise SystemExit(f"FAIL: {dehaze} does not exist — nothing downloaded yet")

    entries = _entries()
    clear_dir = dehaze / "clear"
    have_hazy, missing_hazy, total_bytes = 0, [], 0
    for e in entries:
        p = dehaze / e
        if p.exists():
            have_hazy += 1
            total_bytes += p.stat().st_size
        elif len(missing_hazy) < 10:
            missing_hazy.append(e)

    stems = sorted({_clear_stem(e) for e in entries})
    have_clear, missing_clear = 0, []
    for s in stems:
        hit = next((c for c in clear_dir.glob(f"{s}.*")), None) if clear_dir.exists() else None
        if hit:
            have_clear += 1
            total_bytes += hit.stat().st_size
        elif len(missing_clear) < 10:
            missing_clear.append(s)

    print(f"hazy  : {have_hazy:,} / {len(entries):,}")
    print(f"clear : {have_clear:,} / {len(stems):,}")
    print(f"size  : {total_bytes / 2**30:.2f} GB")

    ok = True
    if have_hazy < len(entries):
        ok = False
        print(f"\nMISSING {len(entries) - have_hazy:,} hazy files, e.g.:")
        for e in missing_hazy:
            print(f"  {e}")
    if have_clear < len(stems):
        ok = False
        print(f"\nMISSING {len(stems) - have_clear:,} clear sources, e.g.:")
        for s in missing_clear:
            print(f"  {s}")

    if ok:
        print("\nCOMPLETE — every hazy file present and every one resolves to a "
              "clear source by the name.split('_')[0] rule, matching how "
              "PairedTestDataset pairs dehaze at evaluation time.")
    else:
        raise SystemExit("\nINCOMPLETE — do not train on a partial set. A missing "
                         "pair fails at load time mid-run rather than up front.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["plan", "verify"])
    (plan if ap.parse_args().stage == "plan" else verify)()
