"""Evaluate the ORIGINAL AdaIR teacher (all_in_one, adair3d.ckpt) on the
exact same 3 held-out sets the student is scored on -- the real
student-vs-teacher gap, using the identical harness (src/eval/evaluate.py)
so the numbers are directly comparable to Trainer._validate_multitask()'s
own output for B0V2-KD-FEAT / the B0V2 baseline.
"""
import sys
sys.path.insert(0, ".")

from pathlib import Path

import torch
import yaml

from src.data.datasets import build_dataset
from src.eval.evaluate import evaluate
from src.eval.metrics import ADAIR_DEFAULT
from src.models.teacher_wrapper import load_teacher
from src.utils.config import teacher_checkpoint

paths = yaml.safe_load(open("configs/paths.local.yaml"))
data_root = Path(paths["data_root"])
device = "cuda" if torch.cuda.is_available() else "cpu"

teacher = load_teacher(teacher_checkpoint("all_in_one"), device=device)


def model_fn(x):
    try:
        return teacher(x)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return teacher.forward_tiled(x)


results = {}
for sigma in (15, 25, 50):
    ds = build_dataset("denoise", data_root / "test/denoise/bsd68", sigma=sigma,
                       seed_mode="filename")
    res = evaluate(model_fn, iter(ds), name=f"denoise_s{sigma}",
                   config=ADAIR_DEFAULT, device=device, keep_per_image=False)
    results[f"psnr_denoise_s{sigma}"] = res.psnr
    results[f"ssim_denoise_s{sigma}"] = res.ssim

results["psnr_denoise"] = sum(results[f"psnr_denoise_s{s}"] for s in (15, 25, 50)) / 3
results["ssim_denoise"] = sum(results[f"ssim_denoise_s{s}"] for s in (15, 25, 50)) / 3

for task, rel in (("derain", "test/derain/rain100L"), ("dehaze", "test/dehaze/sots_clean")):
    ds = build_dataset(task, data_root / rel)
    res = evaluate(model_fn, iter(ds), name=task, config=ADAIR_DEFAULT,
                   device=device, keep_per_image=False)
    results[f"psnr_{task}"] = res.psnr
    results[f"ssim_{task}"] = res.ssim

results["psnr"] = (results["psnr_denoise"] + results["psnr_derain"] + results["psnr_dehaze"]) / 3
results["ssim"] = (results["ssim_denoise"] + results["ssim_derain"] + results["ssim_dehaze"]) / 3

print("\n=== AdaIR teacher (all_in_one), evaluated on the same 3 held-out sets ===")
for k in ("psnr", "psnr_denoise", "psnr_derain", "psnr_dehaze",
         "ssim", "ssim_denoise", "ssim_derain", "ssim_dehaze"):
    print(f"  {k}: {results[k]:.4f}")
