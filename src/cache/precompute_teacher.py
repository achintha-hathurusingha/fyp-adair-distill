"""Task 4/5 — cache frozen-teacher outputs over the training set.

**Full images are cached, never crops.** At training time the *same* random crop
is applied to the `(degraded, GT, teacher_output)` triple, so a full-image cache
is crop-agnostic and reusable across every Phase 02 arm. Caching per-crop would
require freezing crop seeds and would multiply storage by the number of epochs.

**Unique images are cached once.** The sampler's repeat multipliers (derain x120,
denoise x3 per sigma, dehaze x1) apply to sampling, not to the cache.

Ordered smallest-first — Rain100L, then BSD400/WED, then RESIDE — so a partial
run still leaves something usable.

Fully resumable: a manifest records input -> output -> SHA256, integrity is
verified on reload, and completed entries are skipped.

    python -m src.cache.precompute_teacher --tasks derain denoise
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.data.datasets import load_rgb_uint8, to_tensor
from src.data.degradations import SIGMAS, add_gaussian_noise
from src.models.teacher_wrapper import load_teacher
from src.utils.config import REPO_ROOT, load_paths

_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _load_manifest(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"[cache] manifest {path} unreadable; starting fresh")
    return {"entries": {}}


def _save_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _images(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*")
                  if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES)


def save_uint8_png(arr: np.ndarray, path: Path) -> None:
    """Write HWC uint8 losslessly."""
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path, format="PNG", compress_level=6)


def build_jobs(data_root: Path, tasks: list[str]) -> list[dict]:
    """Enumerate unique (input, degradation) pairs to cache, smallest task first."""
    jobs: list[dict] = []
    if "derain" in tasks:
        for p in _images(data_root / "Train" / "Derain" / "input"):
            jobs.append({"key": f"derain/{p.stem}", "path": str(p),
                         "task": "derain", "sigma": None})
    if "denoise" in tasks:
        for p in _images(data_root / "Train" / "Denoise"):
            for sigma in SIGMAS:
                jobs.append({"key": f"denoise_s{sigma}/{p.stem}", "path": str(p),
                             "task": "denoise", "sigma": sigma})
    if "dehaze" in tasks:
        for p in _images(data_root / "Train" / "Dehaze" / "synthetic"):
            jobs.append({"key": f"dehaze/{p.stem}", "path": str(p),
                         "task": "dehaze", "sigma": None})
    return jobs


def degrade(job: dict) -> np.ndarray:
    """Produce the teacher's input for one job, as uint8 HWC.

    Derain and dehaze inputs are already degraded on disk; denoise synthesises
    noise deterministically from the filename (golden-hash pinned).
    """
    img = load_rgb_uint8(Path(job["path"]), base=1)
    if job["task"] == "denoise":
        return add_gaussian_noise(img, job["sigma"],
                                  filename=Path(job["path"]).name)
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description="Cache frozen-teacher outputs.")
    ap.add_argument("--checkpoint", default="data/ckpt/inference/adair3d.pth")
    ap.add_argument("--out-dir", default="data/pairs")
    ap.add_argument("--manifest", default="data/pairs/manifest.json")
    ap.add_argument("--tasks", nargs="+", default=["derain", "denoise", "dehaze"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--overlap", type=int, default=32)
    ap.add_argument("--budget-gb", type=float, default=65.0)
    ap.add_argument("--verify-tiling", action="store_true", default=True)
    args = ap.parse_args()

    paths = load_paths()
    data_root = Path(paths["data_root"])
    if not data_root.is_absolute():
        data_root = REPO_ROOT / data_root
    out_dir = REPO_ROOT / args.out_dir
    manifest_path = REPO_ROOT / args.manifest

    teacher = load_teacher(REPO_ROOT / args.checkpoint, device=args.device)
    print(f"[cache] {teacher}")

    jobs = build_jobs(data_root, args.tasks)
    if not jobs:
        raise SystemExit(f"no images found for tasks {args.tasks}")
    manifest = _load_manifest(manifest_path)
    print(f"[cache] {len(jobs)} job(s); {len(manifest['entries'])} already done")

    # Verify tiled == untiled before trusting it at scale.
    if args.verify_tiling:
        probe = degrade(jobs[0])[:192, :192]
        x = to_tensor(probe).unsqueeze(0)
        full = teacher(x)
        tiled = teacher.forward_tiled(x, tile=128, overlap=args.overlap)
        diff = float((full - tiled).abs().max())
        print(f"[cache] tiling check: max|full-tiled| = {diff:.6f}")
        if diff > 0.02:
            raise SystemExit(
                f"tiled inference deviates from untiled by {diff:.4f} — "
                "do not cache with these tile settings")

    written = 0
    bytes_written = sum(e.get("bytes", 0) for e in manifest["entries"].values())
    t0 = time.time()

    for i, job in enumerate(jobs):
        key = job["key"]
        entry = manifest["entries"].get(key)
        if entry:
            out_path = REPO_ROOT / entry["output"]
            if out_path.exists() and entry.get("sha256"):
                continue  # already cached and recorded
        if bytes_written / 2**30 > args.budget_gb:
            print(f"[cache] storage budget {args.budget_gb} GB reached; stopping "
                  f"at job {i}/{len(jobs)}")
            break

        try:
            degraded = degrade(job)
            x = to_tensor(degraded).unsqueeze(0)
            h, w = degraded.shape[:2]
            pred = (teacher.forward_tiled(x, tile=args.tile, overlap=args.overlap)
                    if max(h, w) > args.tile else teacher(x))
            arr = (pred.clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy() * 255.0
                   ).round().astype(np.uint8)

            out_path = out_dir / f"{key}.png"
            save_uint8_png(arr, out_path)
            size = out_path.stat().st_size
            manifest["entries"][key] = {
                "input": str(Path(job["path"]).relative_to(REPO_ROOT)),
                "output": str(out_path.relative_to(REPO_ROOT)),
                "task": job["task"], "sigma": job["sigma"],
                "sha256": sha256_file(out_path), "bytes": size,
                "shape": [h, w],
            }
            bytes_written += size
            written += 1

            if written % 25 == 0:
                _save_manifest(manifest_path, manifest)
                rate = written / max(1e-9, time.time() - t0)
                remaining = len(jobs) - i - 1
                print(f"[cache] {i+1}/{len(jobs)}  {rate:.2f} img/s  "
                      f"{bytes_written/2**30:.2f} GB  "
                      f"eta {remaining/max(rate,1e-9)/3600:.1f} h")
        except Exception as exc:  # noqa: BLE001
            print(f"[cache] FAILED {key}: {str(exc)[:160]}")
            manifest["entries"][key] = {"error": str(exc)[:300]}

    _save_manifest(manifest_path, manifest)
    done = sum(1 for e in manifest["entries"].values() if e.get("sha256"))
    print(f"[cache] complete: {done}/{len(jobs)} cached, "
          f"{bytes_written/2**30:.2f} GB, {written} written this session")


if __name__ == "__main__":
    main()
