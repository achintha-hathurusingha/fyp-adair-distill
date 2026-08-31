"""Score arms across rain streak ANGLE -- the measurement S3.3 turns on.

S0.1 established that the 4-orientation bank's advantage over a cheap
axis-aligned kernel exists only off-axis. Both of our corpora are near-vertical,
so native Rain100L cannot discriminate. These sets can.

Reads the arm list from the command line so it can be run on whatever has
finished:  python scripts/score_rain_angles.py NAME=ckpt [NAME=ckpt ...]
With no arguments it scores every S3.3 arm that has a checkpoint, plus the
no-block control.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, ".")

import torch

from src.train.train import build_model
from src.train.trainer import Trainer
from src.utils.config import REPO_ROOT, load_paths

ANGLE_SETS = [("0", "test/derain/rain100L_a0000"),
              ("22.5", "test/derain/rain100L_a0225"),
              ("45", "test/derain/rain100L_a0450"),
              ("67.5", "test/derain/rain100L_a0675"),
              ("90", "test/derain/rain100L_a0900")]
NATIVE = ("native", "test/derain/rain100L")

DEFAULTS = {
    "B0V3-KD-FEAT (no block)":
        "runs/b0v3_kd_feat/B0V3-KD-FEAT/B0V3-KD-FEAT_seed0_20260831_083259/last.pth",
    "B0V3-KD-K11 (plain)":  "runs/s33_b0v3_kd_k11/*/*/last.pth",
    "B0V3-KD-ORI":          "runs/s33_b0v3_kd_ori/*/*/last.pth",
    "B0V3-KD-ORI-MID":      "runs/s33_b0v3_kd_ori_mid/*/*/last.pth",
}


def resolve(pat: str) -> str | None:
    if "*" in pat:
        hits = sorted(Path().glob(pat))
        return str(hits[-1]) if hits else None
    return pat if Path(pat).exists() else None


def score(ckpt: str, root: Path, device: str) -> float:
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = build_model(ck["config"])
    model.load_state_dict(ck["model"])
    rd = Path(tempfile.mkdtemp())
    try:
        t = Trainer(model, loader=None, cfg={}, run_dir=rd, device=device,
                    val_tasks={"derain": root})
        return float(t.validate()["psnr_derain"]), ck.get("iteration")
    finally:
        shutil.rmtree(rd, ignore_errors=True)


def main() -> int:
    data_root = Path(load_paths()["data_root"])
    if not data_root.is_absolute():
        data_root = REPO_ROOT / data_root
    device = "cuda" if torch.cuda.is_available() else "cpu"

    arms = {}
    if len(sys.argv) > 1:
        for a in sys.argv[1:]:
            name, _, pat = a.partition("=")
            arms[name] = pat
    else:
        arms = DEFAULTS

    resolved = {}
    for name, pat in arms.items():
        p = resolve(pat)
        if p:
            resolved[name] = p
        else:
            print(f"  (skip, no checkpoint yet) {name}")
    if not resolved:
        print("nothing to score")
        return 0

    sets = [NATIVE] + ANGLE_SETS
    print(f"device={device}   {len(resolved)} arm(s) x {len(sets)} sets\n")
    out = {}
    for name, ck in resolved.items():
        row = {}
        it = None
        for tag, rel in sets:
            root = data_root / rel
            if not root.exists():
                continue
            row[tag], it = score(ck, root, device)
        out[name] = {"iteration": it, "psnr_derain": row}
        cells = "".join(f"{row.get(t, float('nan')):>9.3f}" for t, _ in sets)
        print(f"  {name:<26}{cells}   (it {it})", flush=True)

    hdr = "".join(f"{t:>9}" for t, _ in sets)
    print(f"\n  {'arm':<26}{hdr}")
    for name, r in out.items():
        row = r["psnr_derain"]
        print(f"  {name:<26}" + "".join(f"{row.get(t, float('nan')):>9.3f}"
                                        for t, _ in sets))

    # the decisive contrast, if both arms are present
    a, b = "B0V3-KD-ORI", "B0V3-KD-K11 (plain)"
    if a in out and b in out:
        print(f"\n  ORIENTED minus PLAIN-K11 (isolates orientation at matched "
              f"receptive field):")
        for t, _ in sets:
            ra, rb = out[a]["psnr_derain"].get(t), out[b]["psnr_derain"].get(t)
            if ra is not None and rb is not None:
                print(f"    {t:>6} deg: {ra - rb:+.3f} dB")
        print("\n  S0.1 predicts: ~0 at native/0/90, POSITIVE at 45. A flat "
              "profile closes the spatial-orientation route.")

    Path("reports").mkdir(exist_ok=True)
    with open("reports/rain_angle_profile.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote reports/rain_angle_profile.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
