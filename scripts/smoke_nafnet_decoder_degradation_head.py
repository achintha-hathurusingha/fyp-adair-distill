"""Smoke test for DecoderDegradationHead wired into NAFNet end-to-end (not
the isolated module -- see smoke_decoder_degradation_head.py for that). See
reports/kd_feature_multitask/plan_v2_decoder_film.md. CPU-only, dummy tensors.
"""
import sys
sys.path.insert(0, ".")

import torch
import torch.nn.functional as F

from src.models.nafnet import NAFNet, build_nafnet

GEOM = dict(width=16, enc_blk_nums=[2, 2, 4, 8], middle_blk_num=1,
            dec_blk_nums=[2, 2, 2, 2])

x = torch.randn(2, 3, 64, 64)
labels = torch.tensor([0, 2])  # denoise, dehaze

# 1. Default (off) path unaffected.
plain = NAFNet(**GEOM)
assert plain.decoder_degradation_head is None
out_plain = plain(x)
assert plain.last_degradation_logits is None
print(f"PASS  default path unaffected: out={tuple(out_plain.shape)}")

# 2. use_degradation_head (v1, retired) + use_decoder_degradation_head (v2)
#    together must raise -- they're mutually exclusive.
try:
    NAFNet(**GEOM, use_degradation_head=True, use_decoder_degradation_head=True)
    print("FAIL: expected ValueError for mutually exclusive flags")
except ValueError as e:
    print(f"PASS  mutually-exclusive flags raise as expected: {e}")

# 3. Head-on path: shapes, and the critical property -- middle_blks' output
#    must be IDENTICAL to the head-off run, since classify() is read-only.
# Capture middle_blks' raw output via a forward hook on both nets (same
# random init via manual_seed, so a real apples-to-apples comparison).
torch.manual_seed(0)
plain2 = NAFNet(**GEOM)
torch.manual_seed(0)
net = NAFNet(**GEOM, use_decoder_degradation_head=True)
# net has extra params (the head), so encoder/middle/decoder weights before
# the head are seeded identically only if the head's own param init doesn't
# consume the RNG stream before them -- it does (head is built AFTER
# middle_blks in __init__), so encoder+middle_blks weights match exactly.

captured = {}
def hook(_m, _i, out):
    captured["mid"] = out.detach().clone()
plain2.middle_blks.register_forward_hook(hook)
out_a = plain2(x)
mid_a = captured["mid"]

net.middle_blks.register_forward_hook(hook)
out_b = net(x)
mid_b = captured["mid"]

assert torch.equal(mid_a, mid_b), \
    "middle_blks output differs between head-off and head-on runs -- the " \
    "head must never perturb this tensor (that was the v1 regression)"
print("PASS  middle_blks output identical whether the decoder head is on or off")

assert out_b.shape == out_a.shape
assert net.last_degradation_logits is not None
assert net.last_degradation_logits.shape == (2, 3), \
    f"logits shape wrong: {net.last_degradation_logits.shape}"
print(f"PASS  head-on path: out={tuple(out_b.shape)}  logits={tuple(net.last_degradation_logits.shape)}")

# 4. Gradient flow through the FULL network to every FiLM head.
recon_loss = out_b.pow(2).mean()
aux_loss = F.cross_entropy(net.last_degradation_logits, labels)
(recon_loss + aux_loss).backward()

head = net.decoder_degradation_head
assert head.classifier.weight.grad is not None and head.classifier.weight.grad.abs().sum() > 0, \
    "classifier got no gradient through full NAFNet forward"
for i, film in enumerate(head.films):
    assert film.weight.grad is not None and film.weight.grad.abs().sum() > 0, \
        f"FiLM head {i} got no gradient through full NAFNet forward"
print(f"PASS  gradients reach classifier + all {len(head.films)} FiLM heads through full NAFNet forward+backward")

# 5. build_nafnet() config passthrough.
cfg_off = {"width": 16, "enc_blk_nums": [2, 2, 4, 8], "middle_blk_num": 1,
          "dec_blk_nums": [2, 2, 2, 2]}
net_off = build_nafnet(cfg_off)
assert net_off.decoder_degradation_head is None

cfg_on = dict(cfg_off, use_decoder_degradation_head=True)
net_on = build_nafnet(cfg_on)
assert net_on.decoder_degradation_head is not None

net_override = build_nafnet(cfg_off, use_decoder_degradation_head=True)
assert net_override.decoder_degradation_head is not None
print("PASS  build_nafnet() config passthrough + kwarg override both work")

n_params_off = sum(p.numel() for p in plain.parameters())
n_params_on = sum(p.numel() for p in net.parameters())
print(f"\nparam delta from DecoderDegradationHead: {n_params_on - n_params_off:,}")
print("ALL NAFNET-LEVEL SMOKE CHECKS PASSED")
