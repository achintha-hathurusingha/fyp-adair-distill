"""Do ALL branches of the oriented bank actually learn?

The separable branches are a COMPOSITION of two convs, h_short(h_long(x)). If
BOTH factors are zero-initialised then
    dL/d(h_long)  is proportional to h_short.weight = 0
    dL/d(h_short) is proportional to h_long(x)      = 0
so neither can ever leave zero -- a product of two zeros never escapes. That
would permanently kill the 0 deg and 90 deg orientations while the single-conv
branches (d45/d135/iso) learn normally, leaving a "4-orientation bank" that is
really a 2-orientation bank.

Same class of defect as the one that killed the first K11 run, introduced by the
fix for it. Measured here rather than argued.
"""
import sys

sys.path.insert(0, ".")

import torch
import torch.nn.functional as F

from src.models.reparam_oriented import ReparamOrientedBlock

torch.manual_seed(0)
b = ReparamOrientedBlock(16, k=11).train()
opt = torch.optim.AdamW(b.parameters(), lr=1e-3)
torch.manual_seed(1)
x = torch.randn(4, 16, 32, 32)
tgt = torch.randn(4, 16, 32, 32)

NAMES = ["h_long", "h_short", "v_long", "v_short", "d45", "d135", "iso"]
for _ in range(40):
    opt.zero_grad(set_to_none=True)
    F.l1_loss(b(x), tgt).backward()
    opt.step()

print("  after 40 AdamW steps")
print("  %-9s%11s%13s   %s" % ("branch", "|w|max", "|grad|max", "status"))
dead = []
for n in NAMES:
    w = getattr(b.core, n).weight
    g = 0.0 if w.grad is None else float(w.grad.abs().max())
    wm = float(w.abs().max())
    if wm == 0.0:
        dead.append(n)
    print("  %-9s%11.3e%13.3e   %s"
          % (n, wm, g, "DEAD - never moved" if wm == 0.0 else "learning"))

print("\n  fuse |w|max = %.3e" % float(b.fuse.weight.abs().max()))
print("  coef |grad|max = %.3e" % float(b.core.coef.grad.abs().max()
                                        if b.core.coef.grad is not None else 0.0))
if dead:
    print("\n  *** %d branch(es) permanently dead: %s" % (len(dead), dead))
    print("  *** the 4-orientation bank is not 4 orientations")
else:
    print("\n  all branches learning")


# --- the PLAIN variant too -------------------------------------------------
# Caught by the peer session: B0V3-KD-K11 runs reparam_variant="plain" ->
# PlainLargeKernelBlock, which has NO branches, so everything above passes
# while a plain-variant defect runs free. That is exactly how the first K11
# regression went unexplained. The plain block's own liveness property is that
# conv must move off its delta init once fuse leaves zero.
from src.models.reparam_oriented import PlainLargeKernelBlock  # noqa: E402

torch.manual_seed(0)
pb = PlainLargeKernelBlock(16, k=11).train()
opt2 = torch.optim.AdamW(pb.parameters(), lr=1e-3)
torch.manual_seed(1)
x2, t2 = torch.randn(4, 16, 32, 32), torch.randn(4, 16, 32, 32)
for _ in range(40):
    opt2.zero_grad(set_to_none=True)
    F.l1_loss(pb(x2), t2).backward()
    opt2.step()

kh, kw = pb.conv.weight.shape[-2:]
centre = float(pb.conv.weight[:, 0, kh // 2, kw // 2].abs().mean())
offc = pb.conv.weight.clone()
offc[:, 0, kh // 2, kw // 2] = 0
print("")
print("  PlainLargeKernelBlock (the K11 arm)")
print("  %-9s%11s%13s   %s" % ("tensor", "|w|max", "|grad|max", "status"))
for nm, w in (("conv", pb.conv.weight), ("fuse", pb.fuse.weight)):
    g = 0.0 if w.grad is None else float(w.grad.abs().max())
    print("  %-9s%11.3e%13.3e   %s"
          % (nm, float(w.abs().max()), g,
             "learning" if float(w.abs().max()) > 0 else "DEAD - never moved"))
print("    centre tap mean = %.4f   max off-centre = %.4f" % (centre, float(offc.abs().max())))
plain_ok = float(pb.fuse.weight.abs().max()) > 0 and float(offc.abs().max()) > 0
print("  %s plain block: fuse moved AND conv adapted off delta"
      % ("all good --" if plain_ok else "*** FAIL ***"))
if not plain_ok:
    raise SystemExit(1)
