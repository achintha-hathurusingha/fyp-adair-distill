"""S3.1 module checks + the S3.2 GATE, against the real module.

S3.2 says: merge multi-branch -> one conv; assert max |diff| < 1e-5 on random
input; export and confirm a single Conv in the graph. That passed on the S0.2
stub; this re-runs it against `src/models/reparam_oriented.py` as the plan
requires. Do not train until this passes.
"""
import sys

sys.path.insert(0, ".")

import torch

from src.export.op_coverage import coverage_table, op_histogram
from src.models.reparam_oriented import (
    ReparamOrientedBlock, ReparamOrientedCore, diagonal_mask,
)

K, KP = 11, 3          # k=11 fixed by S0.1
FAIL = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not cond:
        FAIL.append(name)


def main():
    print("=" * 74)
    print(f"S3.1 module + S3.2 gate  (k={K}, kp={KP})")
    print("=" * 74)

    # ---- 1. identity at init ------------------------------------------
    torch.manual_seed(0)
    blk = ReparamOrientedBlock(32, K, KP).eval()
    x = torch.randn(2, 32, 40, 40, requires_grad=True)
    y = blk(x)
    check("identity at init", torch.allclose(y, x),
          f"max|y-x|={float((y - x).abs().max()):.2e}")
    y.sum().backward()
    check("gradients finite", x.grad is not None and bool(torch.isfinite(x.grad).all()))

    # ---- 2. diagonal support survives a step --------------------------
    opt = torch.optim.SGD(blk.parameters(), lr=1.0)
    opt.zero_grad(); blk(x).sum().backward(); opt.step()
    off45 = float((blk.core.d45.weight * (1 - blk.core.m45)).abs().max())
    off135 = float((blk.core.d135.weight * (1 - blk.core.m135)).abs().max())
    check("diagonal masks hold after an optimiser step",
          off45 == 0.0 and off135 == 0.0, f"max off-band {max(off45, off135):.1e}")
    m = diagonal_mask(K, KP, True)
    check("diagonal band is a proper subspace at k=11",
          int(m.sum()) < K * K, f"{int(m.sum())} of {K * K} taps")

    # ---- 3. S3.2 GATE: merge exactness --------------------------------
    print("\n  --- S3.2 gate: merge exactness ---")
    worst = 0.0
    for dim in (16, 32, 64):
        torch.manual_seed(1)
        core = ReparamOrientedCore(dim, K, KP)
        with torch.no_grad():
            for p in core.parameters():
                p.normal_(0, 0.3)
            core._apply_mask()
        core.eval()
        conv = core.merged().eval()
        for s in (K + 2, 32):          # K+2 => almost entirely boundary
            xx = torch.randn(2, dim, s, s)
            with torch.no_grad():
                d = float((core(xx) - conv(xx)).abs().max())
            worst = max(worst, d)
            print(f"    dim {dim:>3}  input {s:>3}x{s:<3}  max|diff| = {d:.3e}")
    check("merge exact (< 1e-5)", worst < 1e-5, f"worst {worst:.3e}")

    # ---- 4. block-level merge + ONNX ----------------------------------
    print("\n  --- deployment graph ---")
    torch.manual_seed(2)
    b = ReparamOrientedBlock(32, K, KP)
    with torch.no_grad():
        for p in b.core.parameters():
            p.normal_(0, 0.3)
        b.core._apply_mask()
        b.fuse.weight.normal_(0, 0.1)   # non-zero, else the block is identity
    b.eval()
    mg = b.merge().eval()
    xx = torch.randn(2, 32, 24, 24)
    with torch.no_grad():
        dblk = float((b(xx) - mg(xx)).abs().max())
    check("block merge exact", dblk < 1e-5, f"max|diff|={dblk:.3e}")

    dummy = torch.randn(1, 32, 64, 64)
    torch.onnx.export(mg, dummy, "runs/s31_merged_block.onnx", opset_version=17,
                      input_names=["x"], output_names=["y"])
    hist = op_histogram("runs/s31_merged_block.onnx")
    print(f"    merged block ops: {dict(hist)}")
    check("merged block is Conv x2 + Add x1 only",
          set(hist) <= {"Conv", "Add"} and hist.get("Conv") == 2)

    # the mergeable CORE alone must be exactly one Conv (S3.2's wording)
    class CoreOnly(torch.nn.Module):
        def __init__(self, c): super().__init__(); self.c = c
        def forward(self, t): return self.c(t)
    torch.onnx.export(CoreOnly(b.core.merged()).eval(), dummy,
                      "runs/s31_merged_core.onnx", opset_version=17,
                      input_names=["x"], output_names=["y"])
    h2 = op_histogram("runs/s31_merged_core.onnx")
    check("merged CORE is exactly one Conv node", dict(h2) == {"Conv": 1}, str(dict(h2)))

    _, _, risks = coverage_table("runs/s31_merged_block.onnx")
    unknown = sorted({o for r in risks.values() for o in r["UNKNOWN"]})
    caution = sorted({o for r in risks.values() for o in r["CAUTION"]})
    check("no UNKNOWN op on any backend", not unknown, str(unknown))
    check("no CAUTION op on any backend", not caution, str(caution))

    # ---- 5. ORT parity -------------------------------------------------
    import numpy as np
    import onnxruntime as ort
    sess = ort.InferenceSession("runs/s31_merged_block.onnx",
                                providers=["CPUExecutionProvider"])
    with torch.no_grad():
        eager = b(dummy).numpy()
    got = sess.run(None, {"x": dummy.numpy()})[0]
    d_ort = float(np.abs(eager - got).max())
    check("ORT(merged) == eager(multi-branch)", d_ort < 1e-4, f"max|diff|={d_ort:.3e}")

    # ---- 6. cost -------------------------------------------------------
    n_tr = sum(p.numel() for p in b.parameters())
    n_dp = sum(p.numel() for p in mg.parameters())
    print(f"\n  training {n_tr:,} params -> deployed {n_dp:,} "
          f"({n_tr / n_dp:.2f}x collapse) at dim=32, k={K}")

    print("\n" + "=" * 74)
    if FAIL:
        print(f"FAILED: {FAIL}")
        return 1
    print("ALL CHECKS PASSED - S3.1 module is sound, S3.2 gate re-passed on the")
    print("real module. Training may proceed (S3.3).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
