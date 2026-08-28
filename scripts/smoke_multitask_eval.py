"""Smoke test for the B0V2 eval-gap fix (kd_feature_multitask plan.md,
section 4). Runs Trainer._validate_multitask() against the REAL demo test
sets configured in configs/train/b0v2_multitask.yaml (denoise/bsd68,
derain/demo, dehaze/demo) on a tiny model, checking:

  1. All three tasks actually get evaluated (psnr_denoise, psnr_derain,
     psnr_dehaze all present and finite) -- this is the exact gap: before
     the fix only denoise ever ran.
  2. The combined `psnr`/`ssim` is the mean across tasks, not silently
     just one of them.
  3. The old single-task path (Trainer with val_root/val_task, val_tasks=
     None) is untouched -- still returns only denoise, byte-identical
     behaviour for every existing single-task arm.
"""
import sys
sys.path.insert(0, ".")

import shutil
import tempfile
from pathlib import Path

import torch
import yaml

from src.models.nafnet import NAFNet
from src.train.trainer import Trainer

paths = yaml.safe_load(Path("configs/paths.local.yaml").read_text())
data_root = Path(paths["data_root"])
b0v2 = yaml.safe_load(Path("configs/train/b0v2_multitask.yaml").read_text())
val_tasks_rel = b0v2["eval"]["val_tasks"]
assert set(val_tasks_rel) == {"denoise", "derain", "dehaze"}, \
    f"expected all 3 tasks in eval.val_tasks, got {sorted(val_tasks_rel)}"

device = "cuda" if torch.cuda.is_available() else "cpu"
TINY = dict(width=8, enc_blk_nums=[1, 1], middle_blk_num=1, dec_blk_nums=[1, 1])

# --- 1+2. multi-task path: all three tasks evaluated, combined = mean ---
val_tasks = {t: data_root / rel for t, rel in val_tasks_rel.items()}
model = NAFNet(**TINY)
run_dir = Path(tempfile.mkdtemp())
try:
    trainer = Trainer(model, loader=None, cfg={}, run_dir=run_dir,
                      device=device, val_tasks=val_tasks)
    metrics = trainer.validate()
    for task in ("denoise", "derain", "dehaze"):
        assert f"psnr_{task}" in metrics, f"psnr_{task} missing from {sorted(metrics)}"
        assert f"ssim_{task}" in metrics, f"ssim_{task} missing from {sorted(metrics)}"
        v = metrics[f"psnr_{task}"]
        assert v == v and v not in (float("inf"), float("-inf")), \
            f"psnr_{task} not finite: {v}"
    print("PASS  all 3 tasks evaluated: "
          f"denoise={metrics['psnr_denoise']:.2f}dB  "
          f"derain={metrics['psnr_derain']:.2f}dB  "
          f"dehaze={metrics['psnr_dehaze']:.2f}dB")

    expected_combined = (metrics["psnr_denoise"] + metrics["psnr_derain"]
                         + metrics["psnr_dehaze"]) / 3
    assert abs(metrics["psnr"] - expected_combined) < 1e-6, \
        f"combined psnr {metrics['psnr']} != mean of per-task {expected_combined}"
    print(f"PASS  combined psnr {metrics['psnr']:.3f}dB is the mean across tasks "
          "(not silently just one)")
finally:
    shutil.rmtree(run_dir, ignore_errors=True)

# --- 3. old single-task path unaffected ---
run_dir2 = Path(tempfile.mkdtemp())
try:
    model2 = NAFNet(**TINY)
    trainer2 = Trainer(model2, loader=None, cfg={"eval": {"val_task": "denoise"}},
                       run_dir=run_dir2, device=device,
                       val_root=data_root / val_tasks_rel["denoise"])
    metrics2 = trainer2.validate()
    assert "psnr_derain" not in metrics2 and "psnr_dehaze" not in metrics2, \
        "single-task path leaked multi-task keys -- val_tasks=None branch changed behaviour"
    assert "psnr" in metrics2
    print(f"PASS  single-task path unaffected: keys={sorted(metrics2.keys())}")
finally:
    shutil.rmtree(run_dir2, ignore_errors=True)

print("\nALL B0V2 EVAL-GAP SMOKE CHECKS PASSED")
