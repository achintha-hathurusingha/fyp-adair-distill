"""How much of B0V2-KD-FEAT's elapsed_s at each checkpoint is actually
validation, not training? Trainer.validate() runs SYNCHRONOUSLY inside the
timed training loop (train.py's elapsed_s = time.time() - t0 over the WHOLE
loop, validation included) -- and the B0V2 eval-gap fix made validation
3x bigger (394 held-out images across 3 tasks, full resolution, vs the old
buggy default's 204 denoise-only images; kd_feat's own single-task demo
validated only 150 images). Times the real _validate_multitask() path
directly, on the real architecture/checkpoint.
"""
import sys, time
sys.path.insert(0, ".")

import torch
import yaml
from pathlib import Path

from src.models.nafnet import NAFNet
from src.train.trainer import Trainer

paths = yaml.safe_load(open("configs/paths.local.yaml"))
data_root = Path(paths["data_root"])

W16_SIDD = dict(width=16, enc_blk_nums=[2, 2, 4, 8], middle_blk_num=12,
                dec_blk_nums=[2, 2, 2, 2], norm_type="layernorm2d",
                full_res_norm_type="affine_clamp", enc_clamp_stages=[3],
                deep_clamp_bound=32.0)

model = NAFNet(**W16_SIDD).to("cuda")
val_tasks = {
    "denoise": data_root / "test/denoise/bsd68",
    "derain": data_root / "test/derain/demo",
    "dehaze": data_root / "test/dehaze/demo",
}
cfg = {"eval": {}, "loss": {"name": "charbonnier", "eps": 1e-3},
      "optim": {}, "schedule": {}, "train": {}}
import tempfile, shutil
run_dir = Path(tempfile.mkdtemp())
try:
    trainer = Trainer(model, loader=None, cfg=cfg, run_dir=run_dir, device="cuda",
                      val_tasks=val_tasks)
    torch.cuda.synchronize()
    t0 = time.time()
    metrics = trainer.validate()
    torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"Full multi-task validate() (denoise 204 + derain 40 + dehaze 150 = 394 images): "
          f"{dt:.1f}s")
    print(f"  psnr={metrics.get('psnr'):.3f} (untrained model, numbers meaningless -- timing only)")

    # single-task denoise-only, for comparison against the OLD (buggy) default
    trainer2 = Trainer(model, loader=None,
                       cfg={"eval": {"val_task": "denoise"}, "loss": {"name": "charbonnier", "eps": 1e-3},
                            "optim": {}, "schedule": {}, "train": {}},
                       run_dir=run_dir, device="cuda",
                       val_root=data_root / "test/denoise/bsd68")
    torch.cuda.synchronize()
    t0 = time.time()
    trainer2.validate()
    torch.cuda.synchronize()
    dt2 = time.time() - t0
    print(f"Old single-task denoise-only validate() (204 images): {dt2:.1f}s")

    # single-task dehaze-only, matching kd_feat's OWN validation cost exactly
    trainer3 = Trainer(model, loader=None,
                       cfg={"eval": {"val_task": "dehaze"}, "loss": {"name": "charbonnier", "eps": 1e-3},
                            "optim": {}, "schedule": {}, "train": {}},
                       run_dir=run_dir, device="cuda",
                       val_root=data_root / "test/dehaze/demo")
    torch.cuda.synchronize()
    t0 = time.time()
    trainer3.validate()
    torch.cuda.synchronize()
    dt3 = time.time() - t0
    print(f"kd_feat-style dehaze-only validate() (150 images): {dt3:.1f}s")

    print(f"\nvalidation overhead added by the eval-gap fix, per checkpoint: "
          f"{dt - dt3:.1f}s more than kd_feat used to pay")
finally:
    shutil.rmtree(run_dir, ignore_errors=True)
