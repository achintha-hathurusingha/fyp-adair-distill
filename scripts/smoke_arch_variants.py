"""One-off smoke test for the student_arch experiment (ECA + GroupNorm).

CPU-only, dummy tensors, no GPU — kd_freq and kd_feat own the GPU right
now. Just: do the two new NAFBlock variants actually build and forward
correctly, matching the same shape as the existing SCA/LayerNorm2d arms,
before trusting either in a real training config.
"""
import sys
sys.path.insert(0, ".")

import torch

from src.models.nafnet import NAFNet, build_nafnet

W16_SIDD = dict(width=16, enc_blk_nums=[2, 2, 4, 8], middle_blk_num=12,
                dec_blk_nums=[2, 2, 2, 2])

x = torch.randn(1, 3, 128, 128)

for name, kwargs in [
    ("baseline (sca + layernorm2d)", dict(norm_type="layernorm2d", attn_type="sca")),
    ("eca", dict(norm_type="layernorm2d", attn_type="eca")),
    ("groupnorm", dict(norm_type="groupnorm", attn_type="sca")),
    ("eca + groupnorm", dict(norm_type="groupnorm", attn_type="eca")),
]:
    model = NAFNet(**W16_SIDD, **kwargs)
    n_params = sum(p.numel() for p in model.parameters())
    with torch.no_grad():
        out = model(x)
    assert out.shape == x.shape, f"{name}: shape mismatch {out.shape} vs {x.shape}"
    print(f"PASS  {name:28s}  params={n_params:,}  out={tuple(out.shape)}")

# Bad channel-count case: GroupNorm2d must raise, not silently misgroup.
try:
    from src.models.norms import GroupNorm2d
    GroupNorm2d(channels=17)  # not divisible by num_groups=8
    print("FAIL: GroupNorm2d(17) should have raised")
except ValueError as e:
    print(f"PASS  GroupNorm2d(17) raises as expected: {e}")

print("\nALL SMOKE CHECKS PASSED")
