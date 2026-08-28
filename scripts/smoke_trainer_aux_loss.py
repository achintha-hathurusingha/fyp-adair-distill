"""Smoke test for the aux_weight wiring in Trainer (kd_feature_multitask
Step 3 -- see reports/kd_feature_multitask/plan.md). Runs a handful of REAL
optimizer steps on the real multi-task loader (tiny `length`, CPU or GPU
whichever is free) and checks:

  1. aux_weight > 0 without use_degradation_head=True raises at construction
     (the guard added in Trainer.__init__), not a silent no-op.
  2. A live run logs `aux_last` in its history rows and the value is finite.
  3. DegradationHead params actually move (their sum changes across steps) --
     the direct evidence gradient is really flowing THROUGH THE OPTIMIZER,
     not just reachable in isolation (smoke_nafnet_degradation_head.py
     already checked .grad is populated; this checks the weights themselves
     change, the thing that would matter if some code path zeroed grad
     between backward() and optimizer.step()).

Uses the real data on this machine (configs/paths.local.yaml) via
build_multitask_loader with a tiny `length` -- not synthetic tensors --
because the point is to catch anything only real provenance dicts /
real batch collation would expose (e.g. task id dtype/device mismatches).
"""
import sys
sys.path.insert(0, ".")

import shutil
import tempfile
from pathlib import Path

import torch
import yaml

from src.data.build import build_multitask_loader
from src.models.nafnet import NAFNet
from src.train.trainer import Trainer

paths = yaml.safe_load(Path("configs/paths.local.yaml").read_text())
data_root = Path(paths["data_root"])
sources = {
    "denoise": data_root / paths["datasets"]["denoise_train"],
    "derain": data_root / paths["datasets"]["derain_train"],
    "dehaze": {
        "input": data_root / "Train/Dehaze/synthetic",
        "target": data_root / "Train/Dehaze/clear",
    },
}

TINY = dict(width=8, enc_blk_nums=[1, 1], middle_blk_num=1, dec_blk_nums=[1, 1])

# --- 1. guard: aux_weight > 0 without use_degradation_head must raise ---
loader = build_multitask_loader(sources, batch_size=3, patch_size=64,
                                 num_workers=0, seed=0, length=12)
plain_model = NAFNet(**TINY)  # use_degradation_head defaults False
tmp_guard = Path(tempfile.mkdtemp())
try:
    Trainer(plain_model, loader, {"distill": {"aux_weight": 1.0}}, tmp_guard,
            device="cpu")
    print("FAIL: expected ValueError, aux_weight>0 with no DegradationHead")
    sys.exit(1)
except ValueError as e:
    print(f"PASS  guard raises as expected: {e}")
finally:
    shutil.rmtree(tmp_guard, ignore_errors=True)

# --- 2+3. real run: aux_last logged, DegradationHead weights actually move ---
loader = build_multitask_loader(sources, batch_size=3, patch_size=64,
                                 num_workers=0, seed=0, length=24)
model = NAFNet(**TINY, use_degradation_head=True)
before = model.degradation_head.classifier.weight.detach().clone()

cfg = {
    "distill": {"aux_weight": 1.0},
    "optim": {"lr": 1e-2, "weight_decay": 0.0, "betas": [0.9, 0.9]},
    "schedule": {"total_iters": 4, "warmup_iters": 1, "min_lr": 1e-3},
    "train": {"amp": False, "val_every": 4, "ckpt_every": 4, "accum_steps": 1},
    "loss": {"name": "charbonnier", "eps": 1e-3},
    "eval": {},
}
run_dir = Path(tempfile.mkdtemp())
try:
    trainer = Trainer(model, loader, cfg, run_dir, device="cpu")
    trainer.validate = lambda: {}  # skip real eval harness; not under test here
    state = trainer.train()

    rows_with_aux = [r for r in state.history if "aux_last" in r]
    assert rows_with_aux, "no history row carried aux_last -- not being logged"
    for r in rows_with_aux:
        assert r["aux_last"] == r["aux_last"] and r["aux_last"] != float("inf"), \
            f"aux_last not finite: {r['aux_last']}"
    print(f"PASS  aux_last logged and finite across {len(rows_with_aux)} row(s): "
          f"{[round(r['aux_last'], 4) for r in rows_with_aux]}")

    after = model.degradation_head.classifier.weight.detach()
    delta = (after - before).abs().sum().item()
    assert delta > 0, "DegradationHead.classifier weights did not move at all"
    print(f"PASS  DegradationHead weights moved: |delta|_1 = {delta:.6f}")
finally:
    shutil.rmtree(run_dir, ignore_errors=True)

print("\nALL TRAINER-LEVEL AUX-LOSS SMOKE CHECKS PASSED")
