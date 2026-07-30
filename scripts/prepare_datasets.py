"""Extract downloaded archives into the directory layout AdaIR expects.

The released archives do not match AdaIR's expected tree, and the mismatch is
not cosmetic. AdaIR pairs deraining ground truth by string substitution
(``dataset_utils.py:327``)::

    gt_name = degraded_name.replace("input", "target")

so the target file must carry the **same basename as its input**. Rain100L ships
clean images as ``norain-001.png`` and rainy ones as ``rainy/rain-001.png``; a
copy that preserved those names would silently fail to pair. This script renames
so that ``input/rain-001.png`` maps to ``target/rain-001.png``.

    python -m scripts.prepare_datasets --derain
"""
from __future__ import annotations

import argparse
import re
import shutil
import zipfile
from pathlib import Path

from src.utils.config import REPO_ROOT, load_paths

_NUM = re.compile(r"(\d+)")


def _index(name: str) -> str:
    """Extract the numeric index from a Rain100L filename."""
    m = _NUM.search(Path(name).stem)
    if not m:
        raise ValueError(f"no numeric index in {name!r}")
    return m.group(1).lstrip("0") or "0"


def prepare_rain(zip_path: Path, dest: Path) -> dict[str, int]:
    """Extract a Rain100L-family archive into ``dest/{input,target}``.

    Pairs are matched on numeric index, then written under a shared basename so
    AdaIR's ``replace("input", "target")`` pairing works.
    """
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if not n.endswith("/")
                 and n.lower().endswith(".png")]

        # Clean images are 'norain-*'; degraded are 'rain-*' (but NOT
        # 'rainregion-*' / 'rainstreak-*', which are auxiliary masks).
        clean = {_index(n): n for n in names
                 if Path(n).stem.startswith("norain")}
        degraded = {_index(n): n for n in names
                    if re.fullmatch(r"rain-\d+", Path(n).stem)}

        shared = sorted(set(clean) & set(degraded), key=int)
        if not shared:
            raise ValueError(f"{zip_path.name}: no clean/degraded pairs found")

        inp, tgt = dest / "input", dest / "target"
        inp.mkdir(parents=True, exist_ok=True)
        tgt.mkdir(parents=True, exist_ok=True)

        for idx in shared:
            basename = f"rain-{int(idx):03d}.png"
            with z.open(degraded[idx]) as src, (inp / basename).open("wb") as fh:
                shutil.copyfileobj(src, fh)
            with z.open(clean[idx]) as src, (tgt / basename).open("wb") as fh:
                shutil.copyfileobj(src, fh)

    return {"pairs": len(shared),
            "clean_only": len(set(clean) - set(degraded)),
            "degraded_only": len(set(degraded) - set(clean))}


def prepare_sots(zip_path: Path, dest: Path, split: str = "outdoor") -> dict[str, int]:
    """Extract SOTS ``hazy``/``gt`` into ``dest/{input,target}``.

    Names are preserved verbatim: AdaIR resolves the dehaze target as
    ``name.split('_')[0] + '.png'`` (``dataset_utils.py:329-331``), which the
    released ``0001_0.8_0.2.jpg`` -> ``0001.png`` naming already satisfies.

    Pairing here is legitimately **many-to-one** — SOTS-outdoor ships 500 hazy
    images over 492 unique scenes, 8 scenes carrying two atmospheric-parameter
    variants.
    """
    with zipfile.ZipFile(zip_path) as z:
        members = [n for n in z.namelist()
                   if not n.endswith("/") and f"SOTS/{split}/" in n]
        hazy = [n for n in members if "/hazy/" in n]
        gt = [n for n in members if "/gt/" in n]
        if not hazy or not gt:
            raise ValueError(f"{zip_path.name}: no hazy/gt under SOTS/{split}")

        inp, tgt = dest / "input", dest / "target"
        inp.mkdir(parents=True, exist_ok=True)
        tgt.mkdir(parents=True, exist_ok=True)
        for src_name, out_dir in ((h, inp) for h in hazy):
            with z.open(src_name) as src, \
                    (out_dir / Path(src_name).name).open("wb") as fh:
                shutil.copyfileobj(src, fh)
        for src_name in gt:
            with z.open(src_name) as src, \
                    (tgt / Path(src_name).name).open("wb") as fh:
                shutil.copyfileobj(src, fh)

    gt_ids = {Path(n).stem for n in gt}
    unpaired = [Path(n).name for n in hazy
                if Path(n).stem.split("_")[0] not in gt_ids]
    return {"inputs": len(hazy), "targets": len(gt),
            "unique_scenes": len(gt_ids), "unpaired": len(unpaired)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare datasets into AdaIR layout.")
    ap.add_argument("--dl-dir", default="data/_dl_rain")
    ap.add_argument("--sots-zip", default="data/_dl_sots/SOTS.zip")
    ap.add_argument("--derain", action="store_true")
    ap.add_argument("--dehaze", action="store_true")
    args = ap.parse_args()

    data_root = Path(load_paths()["data_root"])
    if not data_root.is_absolute():
        data_root = REPO_ROOT / data_root
    dl = Path(args.dl_dir)
    if not dl.is_absolute():
        dl = REPO_ROOT / dl

    if args.derain:
        jobs = [
            (dl / "Rain100L.zip", data_root / "test" / "derain" / "Rain100L", 100),
            (dl / "RainTrainL.zip", data_root / "Train" / "Derain", 200),
        ]
        for zip_path, dest, expected in jobs:
            if not zip_path.exists():
                print(f"[prepare] SKIP {zip_path.name} (not downloaded)")
                continue
            stats = prepare_rain(zip_path, dest)
            status = "ok" if stats["pairs"] == expected else "COUNT MISMATCH"
            print(f"[prepare] {zip_path.name:18s} -> {dest}  "
                  f"pairs={stats['pairs']} (expected {expected})  {status}")
            if stats["clean_only"] or stats["degraded_only"]:
                print(f"           unpaired: {stats['clean_only']} clean-only, "
                      f"{stats['degraded_only']} degraded-only")
            if stats["pairs"] != expected:
                raise SystemExit(
                    f"{zip_path.name}: {stats['pairs']} pairs != expected {expected}")
    if args.dehaze:
        sots = Path(args.sots_zip)
        if not sots.is_absolute():
            sots = REPO_ROOT / sots
        if not sots.exists():
            raise SystemExit(f"SOTS archive not found at {sots}")
        dest = data_root / "test" / "dehaze"
        stats = prepare_sots(sots, dest, split="outdoor")
        print(f"[prepare] SOTS.zip (outdoor)   -> {dest}  "
              f"inputs={stats['inputs']} targets={stats['targets']} "
              f"scenes={stats['unique_scenes']} unpaired={stats['unpaired']}")
        if stats["unpaired"]:
            raise SystemExit(
                f"{stats['unpaired']} hazy image(s) have no ground truth")

    if not (args.derain or args.dehaze):
        ap.error("nothing to do; pass --derain and/or --dehaze")


if __name__ == "__main__":
    main()
