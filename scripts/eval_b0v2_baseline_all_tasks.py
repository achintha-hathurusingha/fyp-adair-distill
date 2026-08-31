"""Re-evaluate the OLD B0V2 baseline (GT-only, no distillation) on all 3
tasks -- it was only ever validated on denoise, before the B0V2 eval-gap
fix (reports/kd_feature_multitask/plan.md section 4). Gives a genuine
apples-to-apples comparison against B0V2-KD-FEAT's own per-task numbers:
same architecture, same data mix, only the loss differs (GT-only vs
+response+feature KD).
"""
import sys
sys.path.insert(0, ".")

from pathlib import Path

import torch
import yaml

from src.models.nafnet import NAFNet
from src.train.trainer import Trainer

CKPT = "runs/b0v2/B0V2/B0V2_seed0_20260803_210918/best.pth"

W16_SIDD = dict(width=16, enc_blk_nums=[2, 2, 4, 8], middle_blk_num=12,
                dec_blk_nums=[2, 2, 2, 2], norm_type="layernorm2d",
                full_res_norm_type="affine_clamp", enc_clamp_stages=[3],
                deep_clamp_bound=32.0)

paths = yaml.safe_load(open("configs/paths.local.yaml"))
data_root = Path(paths["data_root"])
val_tasks = {
    "denoise": data_root / "test/denoise/bsd68",
    "derain": data_root / "test/derain/demo",
    "dehaze": data_root / "test/dehaze/demo",
}

device = "cuda" if torch.cuda.is_available() else "cpu"
model = NAFNet(**W16_SIDD)
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model"])
print(f"Loaded {CKPT} (iteration {ckpt.get('iteration', '?')})")

import tempfile, shutil
run_dir = Path(tempfile.mkdtemp())
try:
    trainer = Trainer(model, loader=None, cfg={}, run_dir=run_dir, device=device,
                      val_tasks=val_tasks)
    # The checkpoint's own weights ARE the EMA-averaged weights already
    # (save_checkpoint saves the EMA copy) -- but Trainer.validate() also
    # swaps in its OWN internal EMA shadow, which was just freshly
    # initialized from these same loaded weights, so copy_to()/restore()
    # are a no-op here. Safe either way.
    metrics = trainer.validate()
    print("\n=== B0V2 baseline (GT-only), re-evaluated on ALL 3 tasks ===")
    for k in ("psnr", "psnr_denoise", "psnr_derain", "psnr_dehaze",
             "ssim", "ssim_denoise", "ssim_derain", "ssim_dehaze"):
        print(f"  {k}: {metrics[k]:.4f}")
finally:
    shutil.rmtree(run_dir, ignore_errors=True)
