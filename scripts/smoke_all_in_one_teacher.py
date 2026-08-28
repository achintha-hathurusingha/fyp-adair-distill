"""Smoke test: does `teacher_task: all_in_one` (the new entry, pointing at
the released adair3d.ckpt) actually load through the REAL Trainer teacher
machinery -- same code path as every existing single-task KD arm, just a
different checkpoint. This is the correct teacher for a multi-task student
(B0V2-KD-FEAT/-COND); the single-degradation teachers used by kd_feat's
dehaze-only demo cannot sensibly supervise denoise/derain inputs.

Checks:
  1. teacher_checkpoint("all_in_one") resolves to a real, existing file.
  2. load_teacher() loads it with 0 missing / 0 unexpected keys and the
     expected AdaIR param count (FrozenTeacher's own assertion -- this
     script just needs to not raise).
  3. forward_with_latent() runs on all THREE degradation types (dehaze
     input, a denoise-style clean input, a derain input) without error and
     produces a `latent_pre` of the expected channel count -- the feature-KD
     term needs this for every task, not just dehaze.
"""
import sys
sys.path.insert(0, ".")

from pathlib import Path

import torch
import yaml

from src.utils.config import teacher_checkpoint
from src.models.teacher_wrapper import load_teacher

# --- 1. resolves to a real file ---
path = teacher_checkpoint("all_in_one")
assert path.exists(), f"resolved all_in_one teacher path does not exist: {path}"
assert path.name == "adair3d.ckpt", f"expected adair3d.ckpt, got {path.name}"
print(f"PASS  teacher_checkpoint('all_in_one') -> {path}")

# --- 2. loads cleanly (FrozenTeacher raises internally on any mismatch) ---
device = "cuda" if torch.cuda.is_available() else "cpu"
teacher = load_teacher(path, device=device)
print(f"PASS  loaded: {teacher.n_params:,} params on {device}")

# --- 3. forward_with_latent works for all three degradation shapes ---
paths_cfg = yaml.safe_load(Path("configs/paths.local.yaml").read_text())
data_root = Path(paths_cfg["data_root"])

samples = {
    "dehaze": data_root / "test/dehaze/demo/input",
    "derain": data_root / "test/derain/demo/input",
}
import torchvision.io as tvio
import torch.nn.functional as F

for task, d in samples.items():
    f = sorted(d.iterdir())[0]
    img = tvio.read_image(str(f)).float().unsqueeze(0) / 255.0
    img = F.interpolate(img, size=(128, 128), mode="bilinear").to(device)
    with torch.no_grad():
        soft, latent = teacher.forward_with_latent(img)
    assert soft.shape == img.shape, f"{task}: response shape {soft.shape} != input {img.shape}"
    assert latent.shape[1] == 384, f"{task}: latent_pre channels {latent.shape[1]} != 384"
    print(f"PASS  {task}: response {tuple(soft.shape)}  latent_pre {tuple(latent.shape)}")

# denoise: any clean image run through with synthetic noise added, same as
# the training-time denoise loader would feed it.
f = sorted(samples["dehaze"].iterdir())[0]  # any real image works as a base
img = tvio.read_image(str(f)).float().unsqueeze(0) / 255.0
img = F.interpolate(img, size=(128, 128), mode="bilinear")
noisy = (img + torch.randn_like(img) * (25 / 255.0)).clamp(0, 1).to(device)
with torch.no_grad():
    soft, latent = teacher.forward_with_latent(noisy)
assert soft.shape == noisy.shape
assert latent.shape[1] == 384
print(f"PASS  denoise: response {tuple(soft.shape)}  latent_pre {tuple(latent.shape)}")

print("\nALL ALL-IN-ONE TEACHER SMOKE CHECKS PASSED")
