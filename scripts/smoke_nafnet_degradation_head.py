"""Smoke test for DegradationHead wired into NAFNet end-to-end (not the
isolated module -- see smoke_degradation_head.py for that). CPU-only,
dummy tensors, checks:
  1. use_degradation_head=False stays byte-identical in behaviour (no
     degradation_head submodule constructed at all).
  2. use_degradation_head=True produces the extra logits output and it has
     the right shape.
  3. Gradient actually reaches the DegradationHead's params through a full
     NAFNet forward+backward, not just the isolated module (real risk:
     something upstream/downstream could accidentally detach it).
"""
import sys
sys.path.insert(0, ".")

import torch
import torch.nn.functional as F

from src.models.nafnet import NAFNet, build_nafnet

x = torch.randn(2, 3, 64, 64)
labels = torch.tensor([0, 2])  # denoise, dehaze

# 1. Default (off) path: no degradation_head attribute active.
plain = NAFNet(width=16, enc_blk_nums=[1, 1], middle_blk_num=1, dec_blk_nums=[1, 1])
assert plain.degradation_head is None, "degradation_head should be None when use_degradation_head=False"
out_plain = plain(x)
assert plain.last_degradation_logits is None, "last_degradation_logits should stay None when head is off"
print(f"PASS  default path unaffected: out={tuple(out_plain.shape)}, degradation_head=None")

# 2. Head on: shapes.
net = NAFNet(width=16, enc_blk_nums=[1, 1], middle_blk_num=1, dec_blk_nums=[1, 1],
             use_degradation_head=True)
assert net.degradation_head is not None
out = net(x)
assert out.shape == out_plain.shape, f"output shape changed: {out.shape} vs {out_plain.shape}"
assert net.last_degradation_logits is not None
assert net.last_degradation_logits.shape == (2, 3), \
    f"logits shape wrong: {net.last_degradation_logits.shape}"
print(f"PASS  head-on path: out={tuple(out.shape)}  logits={tuple(net.last_degradation_logits.shape)}")

# 3. Gradient flow through the FULL network, not the isolated module.
recon_loss = out.pow(2).mean()
aux_loss = F.cross_entropy(net.last_degradation_logits, labels)
(recon_loss + aux_loss).backward()

assert net.degradation_head.classifier.weight.grad is not None and \
    net.degradation_head.classifier.weight.grad.abs().sum() > 0, \
    "classifier got no gradient through full NAFNet forward"
assert net.degradation_head.film.weight.grad is not None and \
    net.degradation_head.film.weight.grad.abs().sum() > 0, \
    "FiLM layer got no gradient through full NAFNet forward"
print("PASS  gradients reach DegradationHead through full NAFNet forward+backward")

# 4. build_nafnet() config passthrough.
cfg_off = {"width": 16, "enc_blk_nums": [1, 1], "middle_blk_num": 1, "dec_blk_nums": [1, 1]}
net_off = build_nafnet(cfg_off)
assert net_off.degradation_head is None, "build_nafnet default should leave head off"

cfg_on = dict(cfg_off, use_degradation_head=True)
net_on = build_nafnet(cfg_on)
assert net_on.degradation_head is not None, "build_nafnet should honour cfg use_degradation_head=True"

net_override = build_nafnet(cfg_off, use_degradation_head=True)
assert net_override.degradation_head is not None, "build_nafnet kwarg should override cfg"
print("PASS  build_nafnet() config passthrough + kwarg override both work")

n_params_off = sum(p.numel() for p in plain.parameters())
n_params_on = sum(p.numel() for p in net.parameters())
print(f"\nparam delta from DegradationHead: {n_params_on - n_params_off:,}")
print("ALL NAFNET-LEVEL SMOKE CHECKS PASSED")
