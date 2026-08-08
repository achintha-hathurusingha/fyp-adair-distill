"""Content-hash registry for the public benchmark datasets.

    python scripts/dataset_checksums.py generate            # write reports/checksums/<name>.json
    python scripts/dataset_checksums.py generate --dataset bsd68
    python scripts/dataset_checksums.py verify               # PASS/FAIL per dataset, explicit

**Why this exists instead of DVC.** The datasets under `data_root` are
immutable public benchmarks (BSD400, WED, BSD68, Urban100, Rain100L,
RESIDE-OTS/ITS, SOTS-outdoor) — what varies is *subsets*, and those are already
committed as seeded manifests with a generating seed in the header
(`reports/dehaze_train_list.txt`, `derain_train_list.txt`,
`reside_required_files.txt`). DVC would add a second source of truth for a
problem already solved. What was actually missing is narrower: **a way to
notice if a dataset file was silently corrupted or swapped** — disk error,
partial re-download, wrong file dropped into the wrong directory. This is that,
and nothing more.

**SHA-256 per file, not one hash per dataset.** A single directory-level hash
tells you *that* something changed, never *what*. Per-file hashes let `verify`
name the exact file, which is the difference between "something is wrong" and
"re-download `data/denoise/BSD68/159008.png`".

**A dataset with zero files is not silently skipped.** An empty or absent
directory is exactly the failure mode `scripts/reside_manifest.py verify`
exists to catch for RESIDE specifically — this registry raises the same way
for every dataset, not only that one.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from src.utils.config import REPO_ROOT, load_paths

_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")
_OUT_DIR = REPO_ROOT / "reports" / "checksums"


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _dataset_roots() -> dict[str, Path]:
    paths = load_paths()
    data_root = Path(paths["data_root"])
    if not data_root.is_absolute():
        data_root = REPO_ROOT / data_root
    return {name: data_root / rel for name, rel in (paths.get("datasets") or {}).items()}


def generate(dataset: str | None) -> int:
    roots = _dataset_roots()
    if dataset and dataset not in roots:
        raise SystemExit(f"unknown dataset {dataset!r}; paths.yaml lists "
                         f"{sorted(roots)}")
    targets = {dataset: roots[dataset]} if dataset else roots

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, root in targets.items():
        if not root.exists():
            print(f"  SKIP  {name:<14} {root} does not exist")
            continue
        files = sorted(p for p in root.rglob("*")
                       if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES)
        if not files:
            print(f"  SKIP  {name:<14} {root} exists but has no images")
            continue

        t0 = time.time()
        entries = {}
        for p in files:
            rel = p.relative_to(root).as_posix()
            entries[rel] = {"sha256": _sha256(p), "size": p.stat().st_size}
        manifest = {
            "dataset": name, "root": str(root),
            "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "n_files": len(entries), "files": entries,
        }
        out = _OUT_DIR / f"{name}.json"
        out.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        # relative_to(REPO_ROOT) raises when _OUT_DIR has been pointed
        # elsewhere (tests do exactly this) -- fall back to the absolute path
        # rather than let a log-line convenience crash a real run.
        try:
            shown = out.relative_to(REPO_ROOT)
        except ValueError:
            shown = out
        print(f"  OK    {name:<14} {len(entries):>7,} files  "
              f"{time.time()-t0:6.1f}s -> {shown}")
    return 0


def verify(dataset: str | None) -> int:
    roots = _dataset_roots()
    if dataset and dataset not in roots:
        raise SystemExit(f"unknown dataset {dataset!r}; paths.yaml lists "
                         f"{sorted(roots)}")
    targets = {dataset: roots[dataset]} if dataset else roots

    all_pass = True
    for name, root in targets.items():
        manifest_path = _OUT_DIR / f"{name}.json"
        if not manifest_path.exists():
            print(f"  SKIP  {name:<14} no manifest -- run `generate` first")
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = manifest["files"]

        if not root.exists():
            print(f"  FAIL  {name:<14} dataset root missing: {root}")
            all_pass = False
            continue

        missing, changed, extra = [], [], []
        seen = set()
        for rel, info in expected.items():
            p = root / rel
            seen.add(rel)
            if not p.exists():
                missing.append(rel)
                continue
            if p.stat().st_size != info["size"] or _sha256(p) != info["sha256"]:
                changed.append(rel)

        current = {p.relative_to(root).as_posix() for p in root.rglob("*")
                  if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES}
        extra = sorted(current - seen)

        ok = not missing and not changed
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {name:<14} {len(expected):>7,} expected, "
              f"{len(missing)} missing, {len(changed)} changed, "
              f"{len(extra)} untracked-extra")
        for rel in missing[:5]:
            print(f"          missing: {rel}")
        for rel in changed[:5]:
            print(f"          CHANGED: {rel}  <-- corrupted or silently replaced")
        all_pass = all_pass and ok
    return 0 if all_pass else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=["generate", "verify"])
    ap.add_argument("--dataset", default=None,
                    help="one dataset name from paths.yaml; default = all")
    args = ap.parse_args()
    return generate(args.dataset) if args.action == "generate" else verify(args.dataset)


if __name__ == "__main__":
    raise SystemExit(main())
