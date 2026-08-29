"""Smoke test for CachedTeacherDataset + build_cached_teacher_loader. See
reports/kd_feature_multitask/plan_cached_teacher.md, Step 2.

Includes the D4-equivariance check that DISPROVED the plan's original
re-augmentation mitigation: of the 8 dihedral transforms, only identity
matches a live teacher re-run within tolerance (`_d4` itself is no longer
used by the dataset -- see its own module for the finding). This test stays
in place as a regression guard: if this ever starts passing for k>0 (e.g.
after a teacher checkpoint change), that would be worth revisiting the
re-augmentation mitigation for.
"""
import sys
sys.path.insert(0, ".")

import json
from pathlib import Path

import torch

from src.data.cached_teacher_dataset import (CachedTeacherDataset, _d4,
                                        build_cached_teacher_loader)
from src.models.teacher_wrapper import load_teacher
from src.utils.config import teacher_checkpoint

CACHE = Path("/tmp/cache_smoke")

# --- 1. basic dataset access, no augmentation ---
ds = CachedTeacherDataset(CACHE)
degraded, clean, response, latent, prov = ds[0]
assert degraded.shape == (3, ds.patch, ds.patch)
assert latent.shape == ds.latent_shape
assert prov["task"] in (0, 1, 2)
print(f"PASS  dataset[0]: degraded={tuple(degraded.shape)} "
      f"latent={tuple(latent.shape)} task={prov['task']} sigma={prov['sigma']:.2f}")

ranges = ds.task_ranges()
print(f"PASS  task_ranges(): {dict((k, (v.start, v.stop)) for k, v in ranges.items())}")

# --- 2. loader + BalancedTaskBatchSampler: task balance across batches ---
loader = build_cached_teacher_loader(CACHE, batch_size=6, num_batches=20,
                                     num_workers=0, seed=0)
task_counts = {0: 0, 1: 0, 2: 0}
for degraded_b, clean_b, response_b, latent_b, prov_b in loader:
    assert degraded_b.shape[1:] == (3, ds.patch, ds.patch)
    assert latent_b.shape[1:] == ds.latent_shape
    for t in prov_b["task"].tolist():
        task_counts[t] += 1
total = sum(task_counts.values())
for t, count in task_counts.items():
    frac = count / total
    assert 0.2 < frac < 0.45, \
        f"task {t} got {frac:.1%} of samples -- balance looks broken (expected ~1/3 each)"
print(f"PASS  task balance across 20 batches: {task_counts} "
      f"(fractions: {[f'{c/total:.1%}' for c in task_counts.values()]})")

# --- 3. THE critical check: D4 equivariance, verified against a live re-run ---
device = "cuda" if torch.cuda.is_available() else "cpu"
teacher = load_teacher(teacher_checkpoint("all_in_one"), device=device)

ds_raw = CachedTeacherDataset(CACHE)
degraded0, clean0, response0, latent0, prov0 = ds_raw[0]

RESP_TOL = 1.0 / 255.0 + 1e-6
results = {}
for k in range(8):
    deg_t, clean_t, resp_t, lat_t = _d4(
        [degraded0, clean0, response0, latent0], k)

    # Run the LIVE teacher on the manually-transformed degraded input --
    # this is the ground truth this whole mechanism depends on.
    with torch.no_grad():
        live_response, live_latent = teacher.forward_with_latent(
            deg_t.unsqueeze(0).to(device))
    live_response = live_response.cpu()[0]
    live_latent = live_latent.cpu()[0]

    # Compare to the CACHED response/latent, transformed the SAME way k
    # (not re-run -- this tests that transforming the cached value is
    # equivalent to transforming the input and re-running the teacher).
    resp_diff = (live_response.clamp(0, 1) - resp_t).abs().max().item()
    lat_diff = (live_latent - lat_t).abs().max().item()
    lat_scale = live_latent.abs().mean().item()
    lat_tol = 0.02 * max(lat_scale, 1.0)

    ok = resp_diff <= RESP_TOL and lat_diff <= lat_tol
    results[k] = ok
    rot, flip = k % 4, k // 4
    status = "SAFE" if ok else "UNSAFE"
    print(f"{status}  k={k} (rot={rot*90}deg, flip={bool(flip)}): "
          f"response diff={resp_diff:.5f} (tol {RESP_TOL:.5f}), "
          f"latent diff={lat_diff:.4f} (tol {lat_tol:.4f})")

safe_ks = [k for k, ok in results.items() if ok]
print(f"\nSAFE transforms: {safe_ks} of 8")
print(f"UNSAFE transforms: {[k for k in results if k not in safe_ks]} of 8")
assert 0 in safe_ks, "identity itself failed -- something more basic is broken"
print("\nCACHED-TEACHER-DATASET SMOKE TEST COMPLETE -- see SAFE/UNSAFE breakdown above")
