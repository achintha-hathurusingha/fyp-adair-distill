"""Verify the smoke-built cache: shapes, index consistency, and a correct
correctness check.

IMPORTANT methodology note: comparing a re-run of the teacher on the
CACHED (already uint8-quantized) degraded input against the cached response
is NOT a valid check -- AdaIR's FFT-based processing is nonlinear enough
that teacher(quantize(x)) can differ from quantize(teacher(x)) by more than
simple rounding tolerance (found directly: ~3/255 on one sample, while two
back-to-back live calls on the identical input matched exactly, ruling out
non-determinism as the cause). The correct check reproduces the exact same
deterministic build-time sampling (same seed -> same crops/sigma, bit for
bit, per MultiTaskTrainDataset's own `rng = np.random.default_rng((seed,
idx))`) and compares against what the cache actually stored -- verifying
the WRITE path (indexing, batching, offsets), not re-litigating whether
quantizing the input changes the teacher's output (it does, and that's fine
-- the cache is built from full precision, only stored in reduced precision).
"""
import sys
sys.path.insert(0, ".")

import json
from pathlib import Path

import numpy as np
import torch

from src.models.teacher_wrapper import load_teacher
from src.utils.config import teacher_checkpoint
from scripts.build_teacher_cache import build_task_loader

CACHE = Path("/tmp/cache_smoke")
index = json.loads((CACHE / "index.json").read_text())
n = index["n"]
print(f"index: n={n}, task_ranges={index['task_ranges']}")
assert len(index["sigma"]) == n
covered = set()
for task, (a, b) in index["task_ranges"].items():
    assert 0 <= a < b <= n, f"{task} range [{a},{b}) invalid for n={n}"
    covered.update(range(a, b))
assert covered == set(range(n)), "task_ranges don't exactly partition [0, n)"
print("PASS  index lengths match n, task_ranges exactly partition [0, n)")

degraded = np.memmap(CACHE / "degraded.dat", dtype=np.uint8, mode="r", shape=(n, 3, 128, 128))
clean = np.memmap(CACHE / "clean.dat", dtype=np.uint8, mode="r", shape=(n, 3, 128, 128))
response = np.memmap(CACHE / "response.dat", dtype=np.uint8, mode="r", shape=(n, 3, 128, 128))
latent = np.memmap(CACHE / "latent_pre.dat", dtype=np.float16, mode="r",
                   shape=(n, *index["latent_shape"]))
print(f"PASS  all memmaps open at expected shapes: "
      f"degraded={degraded.shape} latent={latent.shape}")

assert not np.all(degraded == 0), "degraded is all zeros -- write failed silently"
assert not np.all(latent == 0), "latent_pre is all zeros -- write failed silently"
print("PASS  data is non-trivial (not all-zero)")

# Correct check: reproduce derain's build-time sampling deterministically
# (same task, same n_rows, same seed=0 the smoke build used) and compare
# the FIRST few samples -- these must land at the task's own range start.
import yaml
paths = yaml.safe_load(Path("configs/paths.local.yaml").read_text())
data_root = Path(paths["data_root"])
task = "derain"
offset, stop = index["task_ranges"][task]
n_rows = stop - offset
source = data_root / paths["datasets"]["derain_train"]
loader = build_task_loader(task, source, n_rows, batch_size=16, num_workers=0, seed=0)

device = "cuda" if torch.cuda.is_available() else "cpu"
teacher = load_teacher(teacher_checkpoint("all_in_one"), device=device)

degraded_b, clean_b, prov = next(iter(loader))
with torch.no_grad():
    live_response, live_latent = teacher.forward_with_latent(degraded_b.to(device).float())

for k in range(3):
    r = offset + k
    cached_deg = torch.from_numpy(np.array(degraded[r])).float() / 255.0
    live_deg = degraded_b[k]
    deg_diff = (cached_deg - live_deg).abs().max().item()
    assert deg_diff <= 1.0 / 255.0 + 1e-6, \
        f"row {r}: degraded input mismatch, diff={deg_diff:.6f} -- WRITE PATH BUG " \
        f"(wrong row/offset, or non-deterministic sampling)"

    cached_resp = torch.from_numpy(np.array(response[r])).float() / 255.0
    expected_resp = live_response.cpu()[k].clamp(0, 1)
    resp_diff = (cached_resp - expected_resp).abs().max().item()
    assert resp_diff <= 1.0 / 255.0 + 1e-6, \
        f"row {r}: response mismatch after quantization, diff={resp_diff:.6f} " \
        "(should be pure rounding error only, since input matched)"

    cached_lat = torch.from_numpy(np.array(latent[r])).float()
    expected_lat = live_latent.cpu()[k]
    lat_diff = (cached_lat - expected_lat).abs().max().item()
    lat_scale = expected_lat.abs().mean().item()
    assert lat_diff <= 0.02 * max(lat_scale, 1.0), \
        f"row {r}: latent_pre mismatch, diff={lat_diff:.4f} vs scale {lat_scale:.4f}"

    print(f"PASS  row {r}: degraded matches build-time sample exactly "
          f"(diff={deg_diff:.5f}), response/latent match after quantization "
          f"(resp diff={resp_diff:.5f}, latent diff={lat_diff:.4f})")

print("\nALL CACHE VERIFICATION CHECKS PASSED")
