"""Verify the delta-init fix, in the same terms the defect was found in.

The failure signature was: conv.weight sat at |w|max=0.0909 (Kaiming) with
|grad|max=0.000e+00, while fuse.weight moved. So fuse amplified a frozen random
11x11 kernel into the decoder. This checks the properties that actually matter.
"""
import sys

sys.path.insert(0, ".")

import torch

from src.models.reparam_oriented import PlainLargeKernelBlock, ReparamOrientedBlock
from src.models.student_v3 import StudentV3

FAIL = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not cond:
        FAIL.append(name)


def build(**kw):
    torch.manual_seed(0)
    return StudentV3(width=16, enc_blk_nums=[2, 2, 4, 8], middle_blk_num=12,
                     dec_blk_nums=[2, 2, 2, 2], **kw)


def grads(model, tag):
    model.train()
    model.zero_grad(set_to_none=True)
    torch.manual_seed(1)
    model(torch.randn(1, 3, 64, 64)).sum().backward()
    print(f"    {tag}")
    rows = []
    for n, p in model.named_parameters():
        if "reparam" not in n or not n.endswith(("conv.weight", "fuse.weight")):
            continue
        g = 0.0 if p.grad is None else float(p.grad.abs().max())
        rows.append((n, float(p.abs().max()), g))
        print(f"      {n:<34} |w|max={float(p.abs().max()):.4f}   |grad|max={g:.3e}")
    return rows


def main():
    print("=" * 78)
    print("Delta-init verification")
    print("=" * 78)

    # ---- 1. the sub-block itself: does conv(x) == x now? ------------------
    print("\n[1] the actual fix: is the large kernel an identity at init?")
    for name, blk in (("PlainLargeKernelBlock", PlainLargeKernelBlock(32, k=11)),
                      ("ReparamOrientedBlock", ReparamOrientedBlock(32, k=11))):
        blk.eval()
        x = torch.randn(2, 32, 40, 40)
        with torch.no_grad():
            inner = blk.conv(x) if hasattr(blk, "conv") else blk.core(x)
            out = blk(x)
        check(f"{name}: inner operator is identity (conv(x)==x)",
              torch.allclose(inner, x, atol=1e-6),
              f"max|conv(x)-x|={float((inner - x).abs().max()):.3e}")
        check(f"{name}: whole block still exact identity",
              torch.equal(out, x), f"max|out-x|={float((out - x).abs().max()):.3e}")
        w = blk.conv.weight if hasattr(blk, "conv") else blk.core.iso.weight
        check(f"{name}: kernel is delta, not Kaiming",
              abs(float(w.abs().max()) - (1.0 if hasattr(blk, "conv") else 0.0)) < 1e-6,
              f"|w|max={float(w.abs().max()):.4f}")

    # ---- 2. inside StudentV3: byte identity vs the no-block control -------
    print("\n[2] StudentV3: identity against the matched control")
    torch.manual_seed(1)
    xin = torch.randn(1, 3, 64, 64)
    ctrl = build().eval()
    with torch.no_grad():
        y_ctrl = ctrl(xin)
    for variant in ("plain", "oriented"):
        m = build(use_reparam_oriented=True, reparam_variant=variant,
                  reparam_k=11).eval()
        with torch.no_grad():
            y = m(xin)
        d = float((y - y_ctrl).abs().max())
        check(f"{variant}: max |control - block| == 0.0", d == 0.0, f"{d:.3e}")

    # ---- 3. gradient flow, the measurement that found the bug -------------
    print("\n[3] gradient flow (the signature the defect was found in)")
    m = build(use_reparam_oriented=True, reparam_variant="plain", reparam_k=11)
    rows0 = grads(m, "at step 0:")
    conv0 = [r for r in rows0 if r[0].endswith("conv.weight")]
    fuse0 = [r for r in rows0 if r[0].endswith("fuse.weight")]
    check("conv kernels are now delta (|w|max == 1.0, was 0.0909)",
          all(abs(w - 1.0) < 1e-6 for _, w, _ in conv0))
    check("fuse still receives gradient at step 0 (the bootstrap path)",
          any(g > 0 for _, _, g in fuse0))

    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    opt.step()
    rows1 = grads(m, "after one optimiser step:")
    conv1 = [r for r in rows1 if r[0].endswith("conv.weight")]
    check("conv receives NON-ZERO gradient once fuse has moved",
          all(g > 0 for _, _, g in conv1),
          f"min |grad|max = {min(g for _, _, g in conv1):.3e}")
    check("conv is still ~delta after one step (not yet diverged)",
          all(abs(w - 1.0) < 0.5 for _, w, _ in conv1))

    print("\n" + "=" * 78)
    if FAIL:
        print(f"FAILED: {FAIL}")
        return 1
    print("ALL PASSED")
    print("\nNote, stated rather than glossed: with fuse exactly zero, dL/d(conv)")
    print("is exactly zero at step 0 REGARDLESS of how conv is initialised -- that")
    print("is structural. The fix does not change that and is not meant to. It")
    print("changes what fuse amplifies from a random kernel to the identity, so")
    print("the block is harmless while frozen. Getting a non-zero conv gradient at")
    print("literally step 0 would need a structural change (drop the residual add,")
    print("identity-initialise the 1x1 fuse), which is a separate decision.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
