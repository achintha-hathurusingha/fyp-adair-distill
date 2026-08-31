"""S3.1 deliverable: operators-off BYTE-IDENTITY check for StudentV3.

The no-confound check the plan asks for, and the same one used when v3 itself
landed. Adding the reparameterizable oriented block must not perturb any
existing arm by even one bit -- otherwise an S3.3 delta could come from an
incidental backbone change rather than from the block.

GOLDEN was captured by running the identical construction on the commit BEFORE
`use_reparam_oriented` existed:
    StudentV3(width=16, enc=[2,2,4,8], middle=12, dec=[2,2,2,2])
    torch.manual_seed(0) for construction, seed(1) for the input, eval mode
    sha256 over the fp32 output bytes of a (1,3,64,64) input.
"""
import hashlib
import sys

sys.path.insert(0, ".")

import torch

from src.models.student_v3 import StudentV3

GOLDEN_HASH = "ae0891c50c5d2f812032a1349e277ad50e6030ad6d619887efa197b0bf111adc"
GOLDEN_PARAMS = 7447331
GOLDEN_KEYS = 688
GOLDEN_SUM = 2297.45556640625

FAIL = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not cond:
        FAIL.append(name)


def build(**kw):
    torch.manual_seed(0)
    return StudentV3(width=16, enc_blk_nums=[2, 2, 4, 8], middle_blk_num=12,
                     dec_blk_nums=[2, 2, 2, 2], **kw).eval()


def run(m):
    torch.manual_seed(1)
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        return m(x)


def main():
    print("=" * 74)
    print("S3.1 -- operators-off byte-identity (no-confound check)")
    print("=" * 74)

    m = build()                       # defaults => block OFF
    y = run(m)
    h = hashlib.sha256(y.numpy().tobytes()).hexdigest()
    np_ = sum(p.numel() for p in m.parameters())
    nk = len(m.state_dict())

    check("output BYTE-IDENTICAL to pre-change golden", h == GOLDEN_HASH,
          f"{h[:16]}... vs {GOLDEN_HASH[:16]}...")
    check("parameter count unchanged", np_ == GOLDEN_PARAMS, f"{np_:,}")
    check("state_dict key count unchanged", nk == GOLDEN_KEYS, str(nk))
    check("output sum unchanged", float(y.sum()) == GOLDEN_SUM, f"{float(y.sum())!r}")
    check("no reparam keys in the OFF model",
          not [k for k in m.state_dict() if "reparam" in k])

    # --- block ON: must build, run, and be identity at init ---------------
    print("\n  --- block ON ---")
    on = build(use_reparam_oriented=True)
    y_on = run(on)
    n_on = sum(p.numel() for p in on.parameters())
    check("ON model builds and runs", y_on.shape == y.shape, str(tuple(y_on.shape)))
    check("ON adds parameters", n_on > np_, f"{n_on:,} (+{n_on - np_:,})")
    check("zero-init residual => ON == OFF at init (bitwise)",
          hashlib.sha256(y_on.numpy().tobytes()).hexdigest() == GOLDEN_HASH)
    keys = [k for k in on.state_dict() if "reparam" in k]
    check("reparam blocks are registered", len(keys) > 0, f"{len(keys)} keys")
    print(f"    stages: {on.reparam_stages}   added params: {n_on - np_:,}")

    # middle placement (S3.3 will ablate this)
    mid = build(use_reparam_oriented=True, reparam_middle=True)
    n_mid = sum(p.numel() for p in mid.parameters())
    check("middle placement builds", mid.mid_reparam is not None,
          f"{n_mid:,} params (+{n_mid - np_:,})")
    check("middle placement identity at init (bitwise)",
          hashlib.sha256(run(mid).numpy().tobytes()).hexdigest() == GOLDEN_HASH)

    # gradients actually reach the new block
    on.train()
    torch.manual_seed(1)
    out = on(torch.randn(1, 3, 64, 64))
    out.sum().backward()
    g = [n for n, p in on.named_parameters()
         if "reparam" in n and (p.grad is None or not torch.isfinite(p.grad).all())]
    check("all reparam params receive finite gradients", not g, str(g[:3]))
    # At EXACTLY step 0 the core is gradient-isolated: the block is
    # x + fuse(core(x)) with fuse zero-initialised, so d(out)/d(core) =
    # fuse.weight = 0. That is the repo's zero-init residual idiom
    # (OrientedStreakGate does the same) and it is why the block is inert at
    # init. It bootstraps immediately: fuse DOES get gradient at step 0, so
    # after one update fuse != 0 and the core starts learning. Assert both
    # halves, because "core has zero grad" is only correct at step 0 -- if it
    # were still zero after a step, the branches would never train at all.
    fuse_g = [n for n, p in on.named_parameters()
              if "reparam" in n and "fuse" in n and p.grad is not None
              and float(p.grad.abs().sum()) > 0]
    check("fuse receives gradient at step 0 (bootstrap path)", len(fuse_g) > 0,
          f"{len(fuse_g)} tensors")
    core_g0 = [n for n, p in on.named_parameters()
               if "reparam" in n and ".core." in n and p.grad is not None
               and float(p.grad.abs().sum()) > 0]
    check("core is gradient-isolated at step 0 (expected: zero-init residual)",
          len(core_g0) == 0, f"{len(core_g0)} non-zero")

    opt = torch.optim.SGD(on.parameters(), lr=0.1)
    opt.step()                       # fuse moves off zero
    on.zero_grad(set_to_none=True)
    torch.manual_seed(1)
    on(torch.randn(1, 3, 64, 64)).sum().backward()
    core_g1 = [n for n, p in on.named_parameters()
               if "reparam" in n and ".core." in n and p.grad is not None
               and float(p.grad.abs().sum()) > 0]
    check("core params receive NON-ZERO gradient after one step",
          len(core_g1) > 0, f"{len(core_g1)} tensors")

    print("\n" + "=" * 74)
    if FAIL:
        print(f"FAILED: {FAIL}")
        return 1
    print("ALL PASSED - existing arms are bit-identical; the block is inert at")
    print("init and trainable when enabled.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
