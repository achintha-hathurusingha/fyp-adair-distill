"""Re-score the finished 90k arms on LEAK-FREE evaluation sets.

Every multi-task number reported so far used test/derain/demo (40 images, all
byte-identical to training files) and test/dehaze/demo (150 images, all clear
stems present in training). This re-scores the same checkpoints on the published
protocol instead. No retraining: our training data already matches the protocol
(BSD400+WED 5,144 / Rain100L-train 200 / RESIDE-OTS 72,135), and the real test
splits are disjoint from it.

Uses last.pth (the final 90k weights), NOT best.pth. best.pth is selected by
validation PSNR, and validation ran on the test set -- so best.pth is model
selection on test. With a fixed 90k budget the final checkpoint carries no such
selection, which makes it the cleaner thing to report.

Dehaze is scored twice, because the two sets answer different questions:
  sots_clean (417)  genuinely unseen  -> the honest number
  SOTS full  (500)  what the dehazing literature reports -> comparable, but 75
                    of its 492 clear scenes also occur in OTS training
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

ARMS = {
    "B0V3-KD-FEAT": "runs/b0v3_kd_feat/B0V3-KD-FEAT/B0V3-KD-FEAT_seed0_20260831_083259/last.pth",
    "B0V2-KD-FEAT": "runs/b0v2_kd_feat/B0V2-KD-FEAT/B0V2-KD-FEAT_seed0_20260828_193951/last.pth",
    "B0V3":         "runs/b0v3/B0V3/B0V3_seed1_20260830_180851/last.pth",
}

CLEAN = {"denoise": "test/denoise/bsd68",
         "derain":  "test/derain/rain100L",
         "dehaze":  "test/dehaze/sots_clean"}
LEAKED = {"denoise": "test/denoise/bsd68",
          "derain":  "test/derain/demo",
          "dehaze":  "test/dehaze/demo"}
SOTS_FULL = {"dehaze": "dehaze/RESIDE/SOTS/outdoor"}


def roots(data_root: Path, spec: dict) -> dict:
    out = {}
    for task, rel in spec.items():
        p = data_root / rel
        if not p.exists():
            raise SystemExit(f"missing eval set: {p}")
        out[task] = p
    return out


def score(ckpt_path: str, val_tasks: dict, device: str) -> dict:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(ck["config"])
    model.load_state_dict(ck["model"])
    run_dir = Path(tempfile.mkdtemp())
    try:
        t = Trainer(model, loader=None, cfg={}, run_dir=run_dir, device=device,
                    val_tasks=val_tasks)
        return t.validate(), ck.get("iteration")
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def main() -> int:
    data_root = Path(load_paths()["data_root"])
    if not data_root.is_absolute():
        data_root = REPO_ROOT / data_root
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  data_root={data_root}\n")

    clean_r = roots(data_root, CLEAN)
    leaked_r = roots(data_root, LEAKED)
    sots_r = roots(data_root, SOTS_FULL)
    for tag, r in (("CLEAN", clean_r), ("leaked", leaked_r)):
        counts = {k: len(list((v / "input").glob("*"))) if (v / "input").exists()
                  else len(list(v.glob("*"))) for k, v in r.items()}
        print(f"  {tag:<7} {counts}")
    print()

    out = {}
    for name, ck in ARMS.items():
        if not Path(ck).exists():
            print(f"!! missing checkpoint for {name}: {ck}")
            continue
        m_clean, it = score(ck, clean_r, device)
        m_leak, _ = score(ck, leaked_r, device)
        m_sots, _ = score(ck, sots_r, device)
        out[name] = {"iteration": it, "clean": m_clean, "leaked": m_leak,
                     "sots_full": m_sots}
        print(f"=== {name}  (iteration {it}) ===")
        print(f"  {'set':<26}{'denoise':>9}{'derain':>9}{'dehaze':>9}{'mean3':>9}")
        for tag, m in (("LEAKED (as reported)", m_leak), ("CLEAN (published split)", m_clean)):
            d, r, h = m["psnr_denoise"], m["psnr_derain"], m["psnr_dehaze"]
            print(f"  {tag:<26}{d:>9.3f}{r:>9.3f}{h:>9.3f}{(d + r + h) / 3:>9.3f}")
        dd = m_clean["psnr_denoise"] - m_leak["psnr_denoise"]
        dr = m_clean["psnr_derain"] - m_leak["psnr_derain"]
        dh = m_clean["psnr_dehaze"] - m_leak["psnr_dehaze"]
        print(f"  {'delta (clean - leaked)':<26}{dd:>+9.3f}{dr:>+9.3f}{dh:>+9.3f}"
              f"{(dd + dr + dh) / 3:>+9.3f}")
        print(f"  full SOTS-outdoor (500, literature-comparable): "
              f"{m_sots['psnr_dehaze']:.3f} dB\n", flush=True)

    if len(out) > 1:
        print("=" * 68)
        print("MATCHED COMPARISON ON THE CLEAN SETS")
        print("=" * 68)
        print(f"  {'arm':<15}{'denoise':>9}{'derain':>9}{'dehaze':>9}{'mean3':>9}")
        for name, r in out.items():
            m = r["clean"]
            d, rn, h = m["psnr_denoise"], m["psnr_derain"], m["psnr_dehaze"]
            print(f"  {name:<15}{d:>9.3f}{rn:>9.3f}{h:>9.3f}{(d + rn + h) / 3:>9.3f}")
        if "B0V3-KD-FEAT" in out:
            b = out["B0V3-KD-FEAT"]["clean"]
            for name, r in out.items():
                if name == "B0V3-KD-FEAT":
                    continue
                m = r["clean"]
                print(f"  delta KD-FEAT - {name:<14}"
                      f"{b['psnr_denoise'] - m['psnr_denoise']:>+9.3f}"
                      f"{b['psnr_derain'] - m['psnr_derain']:>+9.3f}"
                      f"{b['psnr_dehaze'] - m['psnr_dehaze']:>+9.3f}")

    Path("reports/reparam_gate").mkdir(parents=True, exist_ok=True)
    with open("reports/clean_eval_rescore.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print("\nwrote reports/clean_eval_rescore.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
