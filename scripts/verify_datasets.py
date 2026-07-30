"""Verify dataset integrity against the manifest — fail loudly on any mismatch.

A dataset with 199 instead of 200 pairs silently shifts every number that
depends on it, and the error surfaces months later as an unexplained gap
against published results. This checks counts, pairing, dimensions and channel
counts, and writes a checksum manifest so a later re-download can be compared
byte-for-byte.

    python -m scripts.verify_datasets --split test
    python -m scripts.verify_datasets --split test --write-manifest
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from src.utils.config import REPO_ROOT, load_paths, load_yaml, require

_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")


@dataclass
class DatasetCheck:
    """Verification outcome for one dataset entry."""

    name: str
    path: Path
    exists: bool = False
    n_files: int = 0
    expected: int | None = None
    layout: str = "flat"
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    sizes: dict[str, int] = field(default_factory=dict)
    modes: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.exists and not self.problems


def _images(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir()
                  if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES)


def _inspect(files: list[Path], check: DatasetCheck, limit: int = 200) -> None:
    """Record image dimensions and colour modes; flag non-RGB inputs."""
    from PIL import Image

    for p in files[:limit]:
        try:
            with Image.open(p) as im:
                check.sizes[f"{im.size[0]}x{im.size[1]}"] = \
                    check.sizes.get(f"{im.size[0]}x{im.size[1]}", 0) + 1
                check.modes[im.mode] = check.modes.get(im.mode, 0) + 1
        except Exception as exc:  # noqa: BLE001
            check.problems.append(f"unreadable image {p.name}: {exc}")

    non_rgb = {m: n for m, n in check.modes.items() if m != "RGB"}
    if non_rgb:
        # Not fatal: AdaIR calls .convert('RGB') on load. Recorded so a
        # greyscale-heavy set is noticed rather than silently converted.
        check.problems.append(
            f"non-RGB source images present {non_rgb} (loader converts, but "
            "confirm this matches the published protocol)")


def check_entry(name: str, spec: dict, data_root: Path) -> DatasetCheck:
    """Verify one manifest entry."""
    rel = require(spec, "path", context=f"datasets.{name}")
    path = data_root / rel
    check = DatasetCheck(name=name, path=path,
                         expected=spec.get("expect_files"),
                         layout=spec.get("layout", "flat"))
    if not path.exists():
        check.problems.append(f"missing directory {path}")
        return check
    check.exists = True

    if check.layout == "input_target":
        inp, tgt = path / "input", path / "target"
        for d in (inp, tgt):
            if not d.exists():
                check.problems.append(f"missing {d.name}/ under {path}")
        if check.problems:
            return check

        ins, tgts = _images(inp), _images(tgt)
        check.n_files = len(ins)

        # Pairing may legitimately be many-to-one: SOTS-outdoor ships 500 hazy
        # images over 492 unique scenes (8 scenes carry two atmospheric-parameter
        # variants), and AdaIR resolves the target by name.split('_')[0]
        # (dataset_utils.py:329-331). So the check that matters is "every input
        # resolves to a target", NOT "counts are equal".
        tgt_stems = {p.stem for p in tgts}
        unpaired = [p.name for p in ins
                    if p.stem not in tgt_stems
                    and p.stem.split("_")[0] not in tgt_stems]
        if unpaired:
            check.problems.append(
                f"{len(unpaired)} input(s) with no matching target, e.g. "
                f"{unpaired[:3]}")

        orphan_targets = tgt_stems - {p.stem for p in ins} - {
            p.stem.split("_")[0] for p in ins}
        if orphan_targets:
            check.problems.append(
                f"{len(orphan_targets)} target(s) with no matching input, e.g. "
                f"{sorted(orphan_targets)[:3]}")

        if len(ins) != len(tgts):
            # Informational, not a failure — recorded so the ratio stays visible.
            check.notes.append(
                f"many-to-one pairing: {len(ins)} inputs over {len(tgts)} targets")
        _inspect(ins, check)
    else:
        files = _images(path)
        check.n_files = len(files)
        _inspect(files, check)

    if check.expected is not None and check.n_files != check.expected:
        check.problems.append(
            f"expected {check.expected} files, found {check.n_files}")
    return check


def sha256_manifest(root: Path, out: Path) -> int:
    """Write a {relative_path: sha256} manifest for every image under root."""
    entries = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES:
            h = hashlib.sha256()
            with p.open("rb") as fh:
                for block in iter(lambda: fh.read(1 << 20), b""):
                    h.update(block)
            entries[str(p.relative_to(root)).replace("\\", "/")] = h.hexdigest()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(entries, indent=2, sort_keys=True), encoding="utf-8")
    return len(entries)


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify dataset integrity.")
    ap.add_argument("--config", default="configs/data/datasets.yaml")
    ap.add_argument("--split", default="test", choices=["test", "train", "all"])
    ap.add_argument("--write-manifest", action="store_true")
    ap.add_argument("--manifest-out", default="data/checksums.json")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    data_root = Path(load_paths()["data_root"])
    if not data_root.is_absolute():
        data_root = REPO_ROOT / data_root

    splits = ["test", "train"] if args.split == "all" else [args.split]
    checks: list[DatasetCheck] = []
    for split in splits:
        for name, spec in (cfg.get(split) or {}).items():
            checks.append(check_entry(f"{split}/{name}", spec, data_root))

    width = max(len(c.name) for c in checks) if checks else 10
    for c in checks:
        status = "OK  " if c.ok else "FAIL"
        exp = "-" if c.expected is None else str(c.expected)
        print(f"[verify] {status} {c.name:<{width}} files={c.n_files:<6} "
              f"expected={exp:<6} {c.sizes if c.sizes else ''}")
        for n in c.notes:
            print(f"           - {n}")
        for p in c.problems:
            print(f"           ! {p}")

    if args.write_manifest:
        n = sha256_manifest(data_root, Path(args.manifest_out))
        print(f"[verify] checksummed {n} images -> {args.manifest_out}")

    failed = [c for c in checks if not c.ok]
    if failed:
        raise SystemExit(
            f"{len(failed)} dataset check(s) failed: "
            f"{', '.join(c.name for c in failed)}")
    print(f"[verify] all {len(checks)} dataset check(s) passed")


if __name__ == "__main__":
    main()
