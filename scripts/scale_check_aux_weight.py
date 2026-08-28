"""Scale-check for distill.aux_weight, same discipline as kd_feat's own
feat_weight=0.01 note in configs/train/m_dehaze_kd_feat.yaml: measure the
RAW (unweighted) term at weight=1.0 against the other loss terms' actual
scale on real data/architecture, so the chosen weight is checked, not
guessed. Real W16_SIDD geometry, real all_in_one teacher, real multitask
loader (tiny length, a handful of optimizer steps -- enough for the term to
settle near its steady early-training scale, not just its random-init
value).
"""
import sys
sys.path.insert(0, ".")

import shutil
import tempfile
from pathlib import Path

import yaml

from src.data.build import build_multitask_loader
from src.models.nafnet import NAFNet
from src.train.trainer import Trainer

paths = yaml.safe_load(Path("configs/paths.local.yaml").read_text())
data_root = Path(paths["data_root"])
sources = {
    "denoise": data_root / paths["datasets"]["denoise_train"],
    "derain": data_root / paths["datasets"]["derain_train"],
    "dehaze": {"input": data_root / "Train/Dehaze/synthetic",
              "target": data_root / "Train/Dehaze/clear"},
}

W16_SIDD = dict(width=16, enc_blk_nums=[2, 2, 4, 8], middle_blk_num=12,
                dec_blk_nums=[2, 2, 2, 2], norm_type="layernorm2d",
                full_res_norm_type="affine_clamp", enc_clamp_stages=[3],
                deep_clamp_bound=32.0)

loader = build_multitask_loader(sources, batch_size=16, patch_size=128,
                                 num_workers=4, seed=0, length=16 * 20)
model = NAFNet(**W16_SIDD, use_degradation_head=True)

cfg = {
    "distill": {"teacher_task": "all_in_one", "weight": 1.0,
               "feat_weight": 0.01, "aux_weight": 1.0},
    "model": {"width": 16, "enc_blk_nums": [2, 2, 4, 8]},
    "optim": {"lr": 1e-3, "weight_decay": 1e-4, "betas": [0.9, 0.9]},
    "schedule": {"total_iters": 10, "warmup_iters": 2, "min_lr": 1e-6},
    "train": {"amp": True, "val_every": 10, "ckpt_every": 10, "accum_steps": 1},
    "loss": {"name": "charbonnier", "eps": 1e-3},
    "eval": {},
}
run_dir = Path(tempfile.mkdtemp())
try:
    trainer = Trainer(model, loader, cfg, run_dir, device="cuda")
    trainer.validate = lambda: {}
    state = trainer.train()
    last = state.history[-1]
    print(f"\nAfter {last['iteration']} steps:")
    print(f"  combined loss (pixel+kd+feat*0.01+aux*1.0): {last['loss']:.5f}")
    print(f"  raw feat term:                              {last.get('feat_last', float('nan')):.5f}"
          f"  (x0.01 = {last.get('feat_last', 0)*0.01:.5f})")
    print(f"  raw aux term (CE, ln(3)={__import__('math').log(3):.4f} at chance): "
          f"{last.get('aux_last', float('nan')):.5f}")
    print(f"  kd term (response KD, weight 1.0):          {trainer._kd_last:.5f}")
finally:
    shutil.rmtree(run_dir, ignore_errors=True)
