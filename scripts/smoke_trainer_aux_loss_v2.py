"""Smoke test for the aux_weight wiring in Trainer against the v2
DecoderDegradationHead (kd_feature_multitask v2 Step 3 -- see
reports/kd_feature_multitask/plan_v2_decoder_film.md). Runs a handful of
REAL optimizer steps on the real multi-task loader. Checks:

  1. The Trainer.__init__ guard accepts use_decoder_degradation_head=True
     (the fix this smoke test exists for -- the guard used to check only
     `model.degradation_head`, which stays None for a v2 model even though
     `model.decoder_degradation_head` is set, and would have wrongly raised).
  2. A live run logs `aux_last`, finite.
  3. The classifier AND every one of the 4 FiLM heads' weights actually move
     across real optimizer steps -- gradient truly flowing through the
     optimizer to every stage, not just reachable in isolation.

Uses the real data on this machine, same as the v1 smoke test.
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

# --- 1. real run: guard accepts v2, aux_last logged, ALL weights move ---
loader = build_multitask_loader(sources, batch_size=3, patch_size=64,
                                 num_workers=0, seed=0, length=24)
model = NAFNet(**TINY, use_decoder_degradation_head=True)
head = model.decoder_degradation_head
before_clf = head.classifier.weight.detach().clone()
before_films = [f.weight.detach().clone() for f in head.films]

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
    trainer.validate = lambda: {}
    state = trainer.train()
    print("PASS  Trainer accepted use_decoder_degradation_head=True (guard fix confirmed)")

    rows_with_aux = [r for r in state.history if "aux_last" in r]
    assert rows_with_aux, "no history row carried aux_last -- not being logged"
    for r in rows_with_aux:
        assert r["aux_last"] == r["aux_last"] and r["aux_last"] != float("inf"), \
            f"aux_last not finite: {r['aux_last']}"
    print(f"PASS  aux_last logged and finite across {len(rows_with_aux)} row(s): "
          f"{[round(r['aux_last'], 4) for r in rows_with_aux]}")

    after_clf = head.classifier.weight.detach()
    delta_clf = (after_clf - before_clf).abs().sum().item()
    assert delta_clf > 0, "classifier weights did not move at all"
    print(f"PASS  classifier weights moved: |delta|_1 = {delta_clf:.6f}")

    for i, (bf, film) in enumerate(zip(before_films, head.films)):
        delta = (film.weight.detach() - bf).abs().sum().item()
        assert delta > 0, f"FiLM head {i} weights did not move at all"
        print(f"PASS  FiLM head {i} weights moved: |delta|_1 = {delta:.6f}")
finally:
    shutil.rmtree(run_dir, ignore_errors=True)

print("\nALL TRAINER-LEVEL V2 AUX-LOSS SMOKE CHECKS PASSED")
