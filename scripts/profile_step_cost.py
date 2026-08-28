"""Where is the per-step time actually going in B0V2-KD-FEAT? Measures each
piece in isolation, on the real architecture/data/teacher, rather than
guessing. Answers: data loader vs teacher forward (fp32, has FFT) vs student
forward+backward vs feature adapter.
"""
import sys, time
sys.path.insert(0, ".")

import torch
import yaml

from src.data.build import build_multitask_loader
from src.models.nafnet import NAFNet
from src.models.teacher_wrapper import load_teacher
from src.models.feature_adapter import FeatureAdapter
from src.losses.reconstruction import build_loss
from src.utils.config import teacher_checkpoint

paths = yaml.safe_load(open("configs/paths.local.yaml"))
data_root = __import__("pathlib").Path(paths["data_root"])
sources = {
    "denoise": data_root / paths["datasets"]["denoise_train"],
    "derain": data_root / paths["datasets"]["derain_train"],
    "dehaze": {"input": data_root / "Train/Dehaze/synthetic",
              "target": data_root / "Train/Dehaze/clear"},
}

device = "cuda"
W16_SIDD = dict(width=16, enc_blk_nums=[2, 2, 4, 8], middle_blk_num=12,
                dec_blk_nums=[2, 2, 2, 2], norm_type="layernorm2d",
                full_res_norm_type="affine_clamp", enc_clamp_stages=[3],
                deep_clamp_bound=32.0)

print("Building loader...")
loader = build_multitask_loader(sources, batch_size=16, patch_size=128,
                                 num_workers=6, seed=0, length=16 * 60)
model = NAFNet(**W16_SIDD).to(device)
teacher = load_teacher(teacher_checkpoint("all_in_one"), device=device)
adapter = FeatureAdapter(in_channels=256, out_channels=384, scale_factor=2.0).to(device)
criterion = build_loss({"name": "charbonnier", "eps": 1e-3})
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

it = iter(loader)


def sync(): torch.cuda.synchronize()


N = 20
# warm up (first batches pay cudnn autotune / worker startup cost)
for _ in range(5):
    degraded, clean, prov = next(it)
    degraded, clean = degraded.to(device), clean.to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred = model(degraded)
        loss = criterion(pred.float(), clean)
    loss.backward()
    opt.zero_grad(set_to_none=True)
    with torch.no_grad(), torch.autocast("cuda", enabled=False):
        teacher.forward_with_latent(degraded.float())
sync()

# 1. data loader alone
t0 = time.time()
batches = []
for _ in range(N):
    batches.append(next(it))
t_loader = time.time() - t0

# 2. student forward+backward alone (reusing already-fetched batches)
t0 = time.time()
for degraded, clean, prov in batches:
    degraded, clean = degraded.to(device), clean.to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred = model(degraded)
        loss = criterion(pred.float(), clean)
    loss.backward()
    opt.zero_grad(set_to_none=True)
sync()
t_student = time.time() - t0

# 3. teacher forward_with_latent alone (fp32, has FFT -- the suspect)
t0 = time.time()
for degraded, clean, prov in batches:
    degraded = degraded.to(device)
    with torch.no_grad(), torch.autocast("cuda", enabled=False):
        soft, latent = teacher.forward_with_latent(degraded.float())
sync()
t_teacher = time.time() - t0

# 4. full step, exactly as trainer.py does it (student fwd+bwd + teacher fwd
#    + response KD + feature KD via adapter), for comparison against the
#    sum of the isolated pieces above.
t0 = time.time()
for degraded, clean, prov in batches:
    degraded, clean = degraded.to(device), clean.to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred = model(degraded)
        loss = criterion(pred.float(), clean)
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            soft, teacher_latent = teacher.forward_with_latent(degraded.float())
        kd = criterion(pred.float(), soft.float())
        loss = loss + 1.0 * kd
        with torch.autocast("cuda", enabled=False):
            b, _, h, w = degraded.shape
            dummy_mid = torch.zeros(b, 256, h // 16, w // 16, device=device)
            adapted = adapter.match_target(dummy_mid, teacher_latent.float())
            # (dummy stand-in for the real middle_blks capture, correctly
            # shaped -- just to cost the adapter's own forward, not to be
            # numerically meaningful)
    loss.backward()
    opt.zero_grad(set_to_none=True)
sync()
t_full = time.time() - t0

print(f"\n{N} iterations:")
print(f"  data loader alone:            {t_loader:6.2f}s  ({t_loader/N*1000:6.1f} ms/it)")
print(f"  student fwd+bwd alone:        {t_student:6.2f}s  ({t_student/N*1000:6.1f} ms/it)")
print(f"  teacher forward_with_latent:  {t_teacher:6.2f}s  ({t_teacher/N*1000:6.1f} ms/it)")
print(f"  full step (as trainer does):  {t_full:6.2f}s  ({t_full/N*1000:6.1f} ms/it)")
print(f"\n  implied it/s if this were the only work: {N/t_full:.3f} it/s")
print(f"  teacher forward as % of full step: {t_teacher/t_full*100:.1f}%")
