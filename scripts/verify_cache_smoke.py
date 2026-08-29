"""Verify the smoke-built cache: shapes, index consistency, and a direct
round-trip check that a cached (response, latent_pre) actually matches a
FRESH live teacher forward on the cached degraded input -- not just "the
file has the right shape", but "the cached values are actually correct."
"""
import sys
sys.path.insert(0, ".")

import json
from pathlib import Path

import numpy as np
import torch

from src.models.teacher_wrapper import load_teacher
from src.utils.config import teacher_checkpoint

CACHE = Path("/tmp/cache_smoke")
index = json.loads((CACHE / "index.json").read_text())
n = index["n"]
print(f"index: n={n}, tasks present={set(index['task'])}")
assert len(index["task"]) == n and len(index["sigma"]) == n
print("PASS  index lengths match n")

degraded = np.memmap(CACHE / "degraded.dat", dtype=np.uint8, mode="r", shape=(n, 3, 128, 128))
clean = np.memmap(CACHE / "clean.dat", dtype=np.uint8, mode="r", shape=(n, 3, 128, 128))
response = np.memmap(CACHE / "response.dat", dtype=np.uint8, mode="r", shape=(n, 3, 128, 128))
latent = np.memmap(CACHE / "latent_pre.dat", dtype=np.float16, mode="r",
                   shape=(n, *index["latent_shape"]))
print(f"PASS  all memmaps open at expected shapes: "
      f"degraded={degraded.shape} latent={latent.shape}")

assert degraded.min() >= 0 and degraded.max() <= 255
assert not np.all(degraded == 0), "degraded is all zeros -- write failed silently"
assert not np.all(latent == 0), "latent_pre is all zeros -- write failed silently"
print("PASS  data is non-trivial (not all-zero)")

# The real check: pick 3 random cached rows, re-run the LIVE teacher on the
# cached degraded input, and confirm the cached response/latent_pre match
# (within uint8/float16 quantization -- not bit-exact, since we stored
# reduced precision, but close).
device = "cuda" if torch.cuda.is_available() else "cpu"
teacher = load_teacher(teacher_checkpoint("all_in_one"), device=device)

rng = np.random.default_rng(0)
for i in rng.choice(n, size=3, replace=False):
    deg = torch.from_numpy(np.array(degraded[i])).float().unsqueeze(0).to(device) / 255.0
    with torch.no_grad():
        live_response, live_latent = teacher.forward_with_latent(deg)
    cached_response = torch.from_numpy(np.array(response[i])).float() / 255.0
    cached_latent = torch.from_numpy(np.array(latent[i])).float()

    resp_diff = (live_response.cpu()[0] - cached_response).abs().max().item()
    lat_diff = (live_latent.cpu()[0] - cached_latent).abs().max().item()
    lat_scale = live_latent.abs().mean().item()

    assert resp_diff < 2.0 / 255.0, \
        f"row {i}: response mismatch, max abs diff {resp_diff:.6f} (uint8 quantization tolerance)"
    assert lat_diff < 0.05 * max(lat_scale, 1.0), \
        f"row {i}: latent_pre mismatch, max abs diff {lat_diff:.4f} vs scale {lat_scale:.4f}"
    print(f"PASS  row {i}: cached matches live re-run "
          f"(response diff={resp_diff:.5f}, latent diff={lat_diff:.4f}, latent scale={lat_scale:.4f})")

print("\nALL CACHE VERIFICATION CHECKS PASSED")
