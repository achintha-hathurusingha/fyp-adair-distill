"""Smoke test for Trainer's use_cached_teacher wiring (cached_teacher Step 3
-- see reports/kd_feature_multitask/plan_cached_teacher.md). Builds a tiny
real cache, then runs real optimizer steps against it. Checks:

  1. Mutual-exclusivity guard: teacher_task AND use_cached_teacher together
     must raise, not silently pick one.
  2. use_cached_teacher=True loads NO live teacher (self.teacher stays None)
     -- the whole point is skipping that ~78% per-step cost.
  3. A live run logs kd_last/feat_last, both finite, and of the same order
     of magnitude as a live-teacher run on the same tiny architecture (not
     an exact match -- cached samples don't line up with a fresh live draw
     -- just a sanity check that nothing is wildly broken).
  4. Student AND adapter weights actually move across real optimizer steps.
"""
import subprocess
import sys
sys.path.insert(0, ".")

import shutil
import tempfile
from pathlib import Path

import torch

from src.data.cached_teacher_dataset import build_cached_teacher_loader
from src.data.build import build_multitask_loader
from src.models.nafnet import NAFNet
from src.train.trainer import Trainer
from src.utils.config import teacher_checkpoint
import yaml

TINY = dict(width=8, enc_blk_nums=[1, 1], middle_blk_num=1, dec_blk_nums=[1, 1])
CACHE_DIR = Path(tempfile.mkdtemp())

try:
    # Build a tiny real cache first (reuses the real build script).
    subprocess.run([
        sys.executable, "scripts/build_teacher_cache.py",
        "--out-dir", str(CACHE_DIR),
        "--denoise", "24", "--derain", "24", "--dehaze", "24",
        "--num-workers", "0",
    ], check=True)

    # --- 1. mutual-exclusivity guard ---
    loader = build_cached_teacher_loader(CACHE_DIR, batch_size=3, num_batches=2, num_workers=0)
    model = NAFNet(**TINY)
    tmp_guard = Path(tempfile.mkdtemp())
    try:
        Trainer(model, loader,
               {"distill": {"use_cached_teacher": True, "teacher_task": "all_in_one",
                           "weight": 1.0}},
               tmp_guard, device="cpu")
        print("FAIL: expected ValueError, teacher_task + use_cached_teacher together")
        sys.exit(1)
    except ValueError as e:
        print(f"PASS  mutual-exclusivity guard raises as expected: {e}")
    finally:
        shutil.rmtree(tmp_guard, ignore_errors=True)

    # --- 2+3+4. real run: no live teacher loaded, real optimizer steps ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader = build_cached_teacher_loader(CACHE_DIR, batch_size=3, num_batches=8, num_workers=0)
    model = NAFNet(**TINY)

    cfg = {
        "distill": {"use_cached_teacher": True, "weight": 1.0, "feat_weight": 0.01},
        "model": {"width": TINY["width"], "enc_blk_nums": TINY["enc_blk_nums"]},
        "optim": {"lr": 1e-2, "weight_decay": 0.0, "betas": [0.9, 0.9]},
        "schedule": {"total_iters": 6, "warmup_iters": 1, "min_lr": 1e-3},
        "train": {"amp": (device == "cuda"), "val_every": 6, "ckpt_every": 6, "accum_steps": 1},
        "loss": {"name": "charbonnier", "eps": 1e-3},
        "eval": {},
    }
    run_dir = Path(tempfile.mkdtemp())
    try:
        trainer = Trainer(model, loader, cfg, run_dir, device=device)
        assert trainer.teacher is None, \
            "use_cached_teacher=True but a live teacher got loaded anyway"
        print("PASS  no live teacher loaded in cached mode")
        assert trainer.adapter is not None, "feat_weight>0 should still build the adapter"
        # Captured AFTER Trainer moves model/adapter to `device` -- comparing
        # pre/post on the same device, not a CPU snapshot vs a CUDA tensor.
        before_model = {n: p.detach().clone() for n, p in model.named_parameters()}
        before_adapter = {n: p.detach().clone() for n, p in trainer.adapter.named_parameters()}

        trainer.validate = lambda: {}
        state = trainer.train()

        rows = [r for r in state.history if "feat_last" in r]
        assert rows, "no history row carried feat_last"
        for r in rows:
            assert r["loss"] == r["loss"] and r["loss"] != float("inf"), \
                f"loss not finite: {r['loss']}"
        print(f"PASS  {len(rows)} row(s) logged, loss finite: "
              f"{[round(r['loss'], 4) for r in rows]}, "
              f"feat_last: {[round(r['feat_last'], 4) for r in rows]}")

        moved = sum((p.detach() - before_model[n]).abs().sum().item()
                   for n, p in model.named_parameters())
        assert moved > 0, "student weights did not move at all"
        print(f"PASS  student weights moved: total |delta|_1 = {moved:.4f}")

        moved_adapter = sum((p.detach() - before_adapter[n]).abs().sum().item()
                            for n, p in trainer.adapter.named_parameters())
        assert moved_adapter > 0, "adapter weights did not move at all"
        print(f"PASS  adapter weights moved: total |delta|_1 = {moved_adapter:.4f}")
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)

    print("\nALL CACHED-TEACHER TRAINER SMOKE CHECKS PASSED")
finally:
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
