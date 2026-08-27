"""One-off smoke test for kd_feature Steps 1-2 (reports/kd_feature/plan.md).

Verifies, on CPU (kd_freq owns the GPU right now — this doesn't need it):
  1. FrozenTeacher.forward_with_latent returns latent_pre at the predicted
     shape (384ch @ 1/8 of input), without touching forward()'s existing
     contract.
  2. FeatureAdapter projects a student-shaped (256ch @ 1/16) tensor to
     exactly the teacher's latent_pre shape.

Not a training run — dummy tensors, no real images, no GPU. Just: does the
plumbing built in this turn actually produce the shapes the plan predicted,
before any loss/training code is written on top of it.
"""
import sys
sys.path.insert(0, ".")

import torch

from src.models.feature_adapter import FeatureAdapter
from src.models.teacher_wrapper import load_teacher
from src.utils.config import teacher_checkpoint

CKPT = teacher_checkpoint("dehaze")
print(f"teacher checkpoint: {CKPT}")

teacher = load_teacher(CKPT, device="cpu")
print(f"loaded: {teacher}")

x = torch.randn(1, 3, 128, 128)
out, latent_pre = teacher.forward_with_latent(x)
print(f"forward_with_latent: out={tuple(out.shape)}  latent_pre={tuple(latent_pre.shape)}")

expected_latent_shape = (1, 384, 16, 16)  # dim=48*2^3 channels, 128/8 spatial
assert tuple(latent_pre.shape) == expected_latent_shape, (
    f"latent_pre shape {tuple(latent_pre.shape)} != predicted {expected_latent_shape}")
assert tuple(out.shape) == (1, 3, 128, 128), "forward()'s own contract broke"
print("PASS: latent_pre shape matches plan.md's prediction exactly")

# Plain forward() must still work unchanged (additive-only guarantee).
out2 = teacher.forward(x)
assert torch.equal(out, out2), "forward_with_latent's output differs from plain forward() — hook leaked state"
print("PASS: forward() unchanged, no state leaked from the hook")

# Adapter: student middle_blks-shaped dummy (256ch @ 1/16 of 128 = 8x8) -> teacher space
student_middle = torch.randn(1, 256, 8, 8)
adapter = FeatureAdapter(in_channels=256, out_channels=384, scale_factor=2.0)
adapted = adapter.match_target(student_middle, latent_pre)
print(f"adapter output: {tuple(adapted.shape)}")
assert adapted.shape == latent_pre.shape, (
    f"adapter output {tuple(adapted.shape)} != latent_pre {tuple(latent_pre.shape)}")
print("PASS: adapter output shape matches latent_pre exactly")

n_adapter_params = sum(p.numel() for p in adapter.parameters())
print(f"adapter params: {n_adapter_params:,} (training-time only, never exported)")

print("\nALL SMOKE CHECKS PASSED")
