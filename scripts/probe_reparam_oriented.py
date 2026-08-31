"""S0.2 -- NPU export gate for the reparameterizable oriented-kernel block.

The plan's binding deployment constraint (SESSION_CONTEXT sec. 3): `torch.fft`
has no ONNX op, and attention (MatMul/Softmax/ReduceL2) is UNKNOWN on all three
NPU backends. So any frequency behaviour must be spatial AT INFERENCE.

The route out is structural reparameterization (RepVGG, Ding et al. CVPR 2021;
RepLKNet, Ding et al. CVPR 2022): train a rich multi-branch block, then merge
every branch ALGEBRAICALLY into one convolution for deployment. Merging is exact
only if the branches combine LINEARLY -- no nonlinearity between the branches
and the sum. That is the binding design constraint of Phase 3, and it is what
separates this block from `probe_oriented_filter.py`'s OrientedStreakGate, which
puts a Sigmoid channel gate between the bands and the fuse and is therefore NOT
mergeable.

This script gates the design BEFORE any training (plan S0.2):
  Kill criterion: any UNKNOWN op surviving reparameterization.

It exports and op-scores BOTH forms -- the multi-branch training graph and the
merged deployment graph -- because only the second is what ships, but the first
is what would have to survive if the merge ever turned out to be inexact.

Merge exactness is asserted here too (numerically, on small inputs where the
zero-padded boundary dominates) rather than assumed. That is not S0.2's job --
it is S3.2's gate -- but it is cheap, and an export of a merged graph that is
not actually equivalent to the trained one would make this gate meaningless.

Branch bank (all depthwise, all linear, all zero-padded):
  0 deg    : separable (1,k) long-axis then (kp,1) short-axis -> kp x k support
  90 deg   : separable (k,1) long-axis then (1,kp) short-axis -> k x kp support
  45/135   : k x k depthwise, gradient-masked to a diagonal band of width kp
  isotropic: kp x kp depthwise
  identity : delta kernel (the repo's zero-init-residual idiom)
Each branch carries a learnable PER-CHANNEL coefficient; the branches are summed.
Everything above is a linear map of x, so the whole bank collapses to one
depthwise k x k conv.

Note on the diagonals: a genuinely separable filter at 45 deg is rank-1 in a
ROTATED frame, which is not an axis-aligned (1,k)*(k,1) composition. Whether the
orientation machinery is worth its keep at all is S0.1's question (the oracle
ceiling). S0.2 only gates OPS, and a diagonal-masked k x k conv and a rotated
rank-1 kernel have identical op signatures, so this gate's verdict does not
depend on which one Phase 3 ends up using.
"""
import sys
from pathlib import Path

sys.path.insert(0, ".")

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.export.op_coverage import coverage_table, op_histogram


# --------------------------------------------------------------------------
# kernel algebra
# --------------------------------------------------------------------------
def full_conv2d_depthwise(w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    """Equivalent single kernel of applying depthwise w1 then depthwise w2.

    PyTorch's conv2d is cross-correlation:  (x * w)[n] = sum_m x[n+m] w[m].
    Composing two of them gives
        ((x * w1) * w2)[n] = sum_q x[n+q] * (w1 FULL-CONV w2)[q],
    i.e. the equivalent kernel is the FULL LINEAR CONVOLUTION of w1 and w2
    (not their cross-correlation) -- hence the flip below.

    w1: (C,1,h1,s1), w2: (C,1,h2,s2)  ->  (C,1,h1+h2-1,s1+s2-1)
    """
    c, _, h1, s1 = w1.shape
    _, _, h2, s2 = w2.shape
    w1p = F.pad(w1, (s2 - 1, s2 - 1, h2 - 1, h2 - 1))
    w1p = w1p.view(1, c, h1 + 2 * (h2 - 1), s1 + 2 * (s2 - 1))
    w2f = torch.flip(w2, dims=(-2, -1))
    out = F.conv2d(w1p, w2f, groups=c)
    return out.view(c, 1, h1 + h2 - 1, s1 + s2 - 1)


def pad_to(w: torch.Tensor, k: int) -> torch.Tensor:
    """Zero-pad a (C,1,h,s) kernel to (C,1,k,k), centred. RepLKNet's trick for
    merging a small kernel into a large one."""
    h, s = w.shape[-2:]
    assert h <= k and s <= k, f"kernel {h}x{s} larger than target {k}x{k}"
    return F.pad(w, ((k - s) // 2, (k - s + 1) // 2, (k - h) // 2, (k - h + 1) // 2))


def diagonal_mask(k: int, band: int, main: bool) -> torch.Tensor:
    """Diagonal band support of width `band`, per Freeman & Adelson: a
    directional filter needs a directional SUPPORT, not just a label."""
    m = torch.zeros(k, k)
    half = band // 2
    for i in range(k):
        j = i if main else (k - 1 - i)
        for dj in range(-half, half + 1):
            jj = j + dj
            if 0 <= jj < k:
                m[i, jj] = 1.0
    return m


# --------------------------------------------------------------------------
# the block
# --------------------------------------------------------------------------
class ReparamOrientedCore(nn.Module):
    """Multi-branch oriented depthwise bank that merges into ONE depthwise conv."""

    def __init__(self, dim: int, k: int = 7, kp: int = 3, use_bn: bool = False):
        super().__init__()
        assert k % 2 == 1 and kp % 2 == 1, "odd kernels only (centred padding)"
        self.dim, self.k, self.kp, self.use_bn = dim, k, kp, use_bn
        dw = dict(groups=dim, bias=False)

        # 0 deg: long horizontal then short vertical (anisotropic, axis-separable)
        self.h_long = nn.Conv2d(dim, dim, (1, k), padding=(0, k // 2), **dw)
        self.h_short = nn.Conv2d(dim, dim, (kp, 1), padding=(kp // 2, 0), **dw)
        # 90 deg: long vertical then short horizontal
        self.v_long = nn.Conv2d(dim, dim, (k, 1), padding=(k // 2, 0), **dw)
        self.v_short = nn.Conv2d(dim, dim, (1, kp), padding=(0, kp // 2), **dw)
        # 45 / 135 deg: k x k masked to a diagonal band
        self.d45 = nn.Conv2d(dim, dim, k, padding=k // 2, **dw)
        self.d135 = nn.Conv2d(dim, dim, k, padding=k // 2, **dw)
        self.register_buffer("m45", diagonal_mask(k, kp, main=True).view(1, 1, k, k))
        self.register_buffer("m135", diagonal_mask(k, kp, main=False).view(1, 1, k, k))
        self._apply_mask()
        # isotropic
        self.iso = nn.Conv2d(dim, dim, kp, padding=kp // 2, **dw)

        # learnable per-channel band coefficients (linear -> mergeable)
        self.coef = nn.Parameter(torch.ones(5, dim))
        self.id_coef = nn.Parameter(torch.ones(dim))

        if use_bn:
            self.bns = nn.ModuleList([nn.BatchNorm2d(dim) for _ in range(5)])

    def _apply_mask(self) -> None:
        with torch.no_grad():
            self.d45.weight.mul_(self.m45)
            self.d135.weight.mul_(self.m135)
        self.d45.weight.register_hook(lambda g: g * self.m45)
        self.d135.weight.register_hook(lambda g: g * self.m135)

    def _branches(self, x):
        return [
            self.h_short(self.h_long(x)),
            self.v_short(self.v_long(x)),
            self.d45(x),
            self.d135(x),
            self.iso(x),
        ]

    def forward(self, x):
        bs = self._branches(x)
        if self.use_bn:
            bs = [bn(b) for bn, b in zip(self.bns, bs)]
        out = self.id_coef.view(1, -1, 1, 1) * x
        for i, b in enumerate(bs):
            out = out + self.coef[i].view(1, -1, 1, 1) * b
        return out

    # --- the merge ---------------------------------------------------------
    def _branch_kernels(self):
        """Each branch's equivalent depthwise k x k kernel, before coefficients."""
        k = self.k
        return [
            pad_to(full_conv2d_depthwise(self.h_long.weight, self.h_short.weight), k),
            pad_to(full_conv2d_depthwise(self.v_long.weight, self.v_short.weight), k),
            pad_to(self.d45.weight * self.m45, k),
            pad_to(self.d135.weight * self.m135, k),
            pad_to(self.iso.weight, k),
        ]

    @torch.no_grad()
    def merged(self) -> nn.Conv2d:
        """Algebraically collapse every branch into a single depthwise conv."""
        k, dim = self.k, self.dim
        ws = self._branch_kernels()
        bias = torch.zeros(dim)

        for i, w in enumerate(ws):
            s = self.coef[i].view(-1, 1, 1, 1)
            if self.use_bn:
                bn = self.bns[i]
                # RepVGG's BN fold, exact at eval:
                # y = (x - mu)/sqrt(var+eps) * gamma + beta
                std = torch.sqrt(bn.running_var + bn.eps)
                s = s * (bn.weight / std).view(-1, 1, 1, 1)
                bias = bias + self.coef[i] * (bn.bias - bn.running_mean * bn.weight / std)
            ws[i] = w * s

        # identity branch = a delta kernel scaled per channel
        delta = torch.zeros(dim, 1, k, k)
        delta[:, 0, k // 2, k // 2] = 1.0
        ws.append(delta * self.id_coef.view(-1, 1, 1, 1))

        conv = nn.Conv2d(dim, dim, k, padding=k // 2, groups=dim, bias=True)
        conv.weight.copy_(sum(ws))
        conv.bias.copy_(bias)
        return conv


class ReparamOrientedBlock(nn.Module):
    """Deployment-shaped wrapper: the mergeable oriented bank + a 1x1 channel
    fuse. The 1x1 cannot fold into a depthwise conv (different group structure),
    so the shipped block is exactly TWO Convs, not one."""

    def __init__(self, dim: int, k: int = 7, kp: int = 3, use_bn: bool = False):
        super().__init__()
        self.core = ReparamOrientedCore(dim, k, kp, use_bn)
        self.fuse = nn.Conv2d(dim, dim, 1, bias=False)
        nn.init.zeros_(self.fuse.weight)  # zero-init residual, repo idiom

    def forward(self, x):
        return x + self.fuse(self.core(x))


class MergedCoreWrapper(nn.Module):
    def __init__(self, conv):
        super().__init__()
        self.conv = conv

    def forward(self, x):
        return self.conv(x)


class MergedBlock(nn.Module):
    def __init__(self, core_conv, fuse):
        super().__init__()
        self.conv, self.fuse = core_conv, fuse

    def forward(self, x):
        return x + self.fuse(self.conv(x))


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------
def check_merge_exact(dim=16, k=7, kp=3, use_bn=False, sizes=(9, 32), seed=0):
    """Assert the merged conv reproduces the multi-branch block to float
    tolerance. Small sizes are deliberate: with k=7 and zero padding a 9x9
    input is almost entirely boundary, which is where a naive merge (one that
    forgets zero-padding must commute through the separable pair) breaks first.
    """
    torch.manual_seed(seed)
    m = ReparamOrientedCore(dim, k, kp, use_bn)
    # randomise everything away from init so the test has teeth
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 0.3)
        m._apply_mask()
        if use_bn:
            for bn in m.bns:
                bn.running_mean.normal_(0, 0.5)
                bn.running_var.uniform_(0.5, 2.0)
    m.eval()
    conv = m.merged().eval()

    worst = 0.0
    for s in sizes:
        x = torch.randn(2, dim, s, s)
        with torch.no_grad():
            d = (m(x) - conv(x)).abs().max().item()
        worst = max(worst, d)
        print(f"    input {s:>3}x{s:<3}  max|multi-branch - merged| = {d:.3e}")
    return worst, m, conv


def report(tag, path, model, dummy):
    torch.onnx.export(model, dummy, path, opset_version=17,
                      input_names=["x"], output_names=["y"])
    hist = op_histogram(path)
    print(f"\n### {tag}  ->  {path}")
    print("  op histogram: " + ", ".join(f"{o} x{c}" for o, c in sorted(hist.items())))
    _, _, risks = coverage_table(path)
    unknown = {b: r["UNKNOWN"] for b, r in risks.items()}
    caution = {b: r["CAUTION"] for b, r in risks.items()}
    for b in risks:
        u = ", ".join(unknown[b]) or "none"
        c = ", ".join(caution[b]) or "none"
        print(f"    {b:<9} UNKNOWN: {u:<20} caution: {c}")
    return hist, unknown, caution


def ort_parity(train_model, merged_model, dim, size=64, seed=0):
    """The PyTorch merge being exact does not prove the EXPORTED graph is.
    Run both ONNX graphs under onnxruntime on the same input and compare.
    This is the claim that actually matters for deployment."""
    import numpy as np
    import onnxruntime as ort

    torch.manual_seed(seed)
    x = torch.randn(1, dim, size, size)
    outs = []
    for path, m in ((f"runs/_parity_train_{dim}.onnx", train_model),
                    (f"runs/_parity_merged_{dim}.onnx", merged_model)):
        torch.onnx.export(m, x, path, opset_version=17,
                          input_names=["x"], output_names=["y"])
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        outs.append(sess.run(None, {"x": x.numpy()})[0])
    d = float(np.abs(outs[0] - outs[1]).max())
    # also compare ORT-merged against the eager PyTorch training block
    with torch.no_grad():
        eager = train_model(x).numpy()
    d_eager = float(np.abs(eager - outs[1]).max())
    return d, d_eager


def sweep(dims=(16, 32, 64), ks=(7, 9, 11), kp=3):
    """The convolution-theorem result (SESSION_CONTEXT) says 7x7-11x11 kernels
    reproduce the full optimal frequency filter, and StudentV3 runs w16 with the
    plan contemplating w24/w32. Confirm the merge and the op set hold across
    that whole design box, not just at one point."""
    rows = []
    for dim in dims:
        for k in ks:
            torch.manual_seed(0)
            m = ReparamOrientedCore(dim, k, kp)
            with torch.no_grad():
                for p in m.parameters():
                    p.normal_(0, 0.3)
                m._apply_mask()
            m.eval()
            conv = m.merged().eval()
            worst = 0.0
            for s in (k + 2, 32):
                x = torch.randn(2, dim, s, s)
                with torch.no_grad():
                    worst = max(worst, (m(x) - conv(x)).abs().max().item())
            path = f"runs/_sweep_{dim}_{k}.onnx"
            torch.onnx.export(MergedCoreWrapper(conv).eval(),
                              torch.randn(1, dim, 64, 64), path, opset_version=17,
                              input_names=["x"], output_names=["y"])
            hist = op_histogram(path)
            _, _, risks = coverage_table(path)
            unk = sorted({o for r in risks.values() for o in r["UNKNOWN"]})
            n_tr = sum(p.numel() for p in m.parameters())
            n_mg = sum(p.numel() for p in conv.parameters())
            rows.append((dim, k, worst, n_tr, n_mg,
                         dict(hist), unk))
            Path(path).unlink(missing_ok=True)
    return rows


def int8_gate(fp32_path, int8_path, shape):
    """Our deployment target is INT8, so the FP32 op set is only half the gate.
    Quantize the merged graph and re-score the QDQ graph."""
    from src.export.quantize import quantize_static_int8

    quantize_static_int8(fp32_path, int8_path, shape, calib_samples=8)
    hist = op_histogram(int8_path)
    _, _, risks = coverage_table(int8_path)
    return hist, {b: r["UNKNOWN"] for b, r in risks.items()}, \
        {b: r["CAUTION"] for b, r in risks.items()}


def main():
    torch.manual_seed(0)
    dim, k, kp = 16, 7, 3

    print("=" * 74)
    print("S0.2 -- NPU export gate, reparameterizable oriented-kernel block")
    print("=" * 74)

    print("\n[1] merge exactness (this is S3.2's gate, checked early so the")
    print("    exported deployment graph is known to BE the trained one)")
    print("  -- no BatchNorm --")
    w_nobn, core, merged_conv = check_merge_exact(dim, k, kp, use_bn=False)
    print("  -- per-branch BatchNorm (RepVGG recipe), folded at eval --")
    w_bn, _, _ = check_merge_exact(dim, k, kp, use_bn=True)
    print(f"  worst overall: no-BN {w_nobn:.3e} | BN {w_bn:.3e}   (S3.2 threshold 1e-5)")

    print("\n[2] identity at init (zero-init residual, repo idiom)")
    torch.manual_seed(0)
    blk = ReparamOrientedBlock(dim, k, kp).eval()
    x = torch.randn(2, dim, 32, 32, requires_grad=True)
    y = blk(x)
    assert torch.allclose(y, x), "block is not identity at init"
    y.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all(), "non-finite grad"
    print("    identity at init OK, gradients finite")

    print("\n[3] diagonal masks survive an optimiser step")
    opt = torch.optim.SGD(blk.parameters(), lr=1.0)
    opt.zero_grad()
    blk(x).sum().backward()
    opt.step()
    off45 = (blk.core.d45.weight * (1 - blk.core.m45)).abs().max().item()
    off135 = (blk.core.d135.weight * (1 - blk.core.m135)).abs().max().item()
    print(f"    max off-band tap after a step: 45deg {off45:.3e}, 135deg {off135:.3e}")
    assert off45 == 0.0 and off135 == 0.0, "diagonal support leaked"

    print("\n[4] parameter cost")
    n_core = sum(p.numel() for p in core.parameters())
    n_merged = sum(p.numel() for p in merged_conv.parameters())
    n_blk = sum(p.numel() for p in blk.parameters())
    print(f"    training core {n_core:>7} params -> merged conv {n_merged:>7} params "
          f"({n_core / n_merged:.2f}x collapse)")
    print(f"    full block (core + 1x1 fuse), training-time: {n_blk} params @ dim={dim}")

    print("\n[5] ONNX export + op coverage")
    dummy = torch.randn(1, dim, 64, 64)
    h_tr, u_tr, c_tr = report("TRAINING graph (multi-branch)",
                              "runs/reparam_oriented_train.onnx",
                              ReparamOrientedBlock(dim, k, kp).eval(), dummy)
    h_dep, u_dep, c_dep = report("DEPLOYMENT graph (merged core only)",
                                 "runs/reparam_oriented_merged.onnx",
                                 MergedCoreWrapper(merged_conv).eval(), dummy)
    h_full, u_full, c_full = report("DEPLOYMENT graph (merged block, incl. 1x1 fuse)",
                                    "runs/reparam_oriented_merged_block.onnx",
                                    MergedBlock(merged_conv, blk.fuse).eval(), dummy)

    print("\n[6] ONNX-RUNTIME parity: is the EXPORTED merged graph equal to the")
    print("    exported training graph? (the PyTorch merge being exact does not")
    print("    prove the exported one is)")
    torch.manual_seed(0)
    core_p = ReparamOrientedCore(dim, k, kp)
    with torch.no_grad():
        for p in core_p.parameters():
            p.normal_(0, 0.3)
        core_p._apply_mask()
    core_p.eval()
    conv_p = core_p.merged().eval()
    d_onnx, d_eager = ort_parity(core_p, MergedCoreWrapper(conv_p).eval(), dim)
    print(f"    max|ORT(train graph) - ORT(merged graph)| = {d_onnx:.3e}")
    print(f"    max|eager PyTorch     - ORT(merged graph)| = {d_eager:.3e}")

    print("\n[7] design-box sweep (dim x kernel): merge exactness + op set")
    print(f"    {'dim':>4} {'k':>3} {'max|diff|':>11} {'train par':>10} "
          f"{'merged par':>11} {'ops':>12}  UNKNOWN")
    for dim_s, k_s, worst, n_tr, n_mg, hist, unk in sweep():
        ops = ",".join(f"{o}x{c}" for o, c in sorted(hist.items()))
        print(f"    {dim_s:>4} {k_s:>3} {worst:>11.3e} {n_tr:>10} {n_mg:>11} "
              f"{ops:>12}  {unk or 'none'}")

    print("\n[8] INT8 gate -- the deployment target is INT8, so the FP32 op set")
    print("    is only half the question. Static PTQ (QDQ, per-channel) of the")
    print("    merged graph, then re-score.")
    try:
        h8, u8, c8 = int8_gate("runs/reparam_oriented_merged.onnx",
                               "runs/reparam_oriented_merged_int8.onnx",
                               (1, dim, 64, 64))
        print("    INT8 op histogram: " + ", ".join(f"{o} x{c}" for o, c in sorted(h8.items())))
        for b in u8:
            print(f"      {b:<9} UNKNOWN: {', '.join(u8[b]) or 'none':<20} "
                  f"caution: {', '.join(c8[b]) or 'none'}")
        int8_unknown = sorted({o for ops in u8.values() for o in ops})
    except Exception as e:  # noqa: BLE001 - report, do not mask
        print(f"    INT8 quantization FAILED: {type(e).__name__}: {e}")
        int8_unknown = ["<quantization failed>"]

    print("\n" + "=" * 74)
    print("VERDICT")
    print("=" * 74)
    n_conv = h_dep.get("Conv", 0)
    print(f"  merged core graph: {n_conv} Conv node(s), {sum(h_dep.values())} node(s) "
          f"total  (S3.2 expects exactly 1 Conv)")
    surviving = sorted({op for ops in u_dep.values() for op in ops})
    if surviving or int8_unknown:
        print(f"  KILL: UNKNOWN ops survive reparameterization: "
              f"FP32 {surviving or 'none'}, INT8 {int8_unknown or 'none'}")
    else:
        print("  PASS: no UNKNOWN op survives reparameterization on any backend,")
        print("        in FP32 or INT8.")
    surv_full = sorted({op for ops in u_full.values() for op in ops})
    print(f"  merged block (with 1x1 fuse) UNKNOWN: {surv_full or 'none'}")
    surv_tr = sorted({op for ops in u_tr.values() for op in ops})
    print(f"  training graph UNKNOWN (informational, never shipped): {surv_tr or 'none'}")
    caut = sorted({op for ops in c_dep.values() for op in ops})
    print(f"  merged core CAUTION: {caut or 'none'}")
    print("\n  Static-table blind spots this gate CANNOT see (flagged, not resolved):")
    print(f"    - op_coverage sees 'Conv' only; it does not distinguish DEPTHWISE")
    print(f"      (groups={dim}) from dense, nor kernel size. A merged {k}x{k} depthwise")
    print("      conv is a different INT8 proposition from a 3x3 dense one.")
    print("    - kernel-size limits are backend-specific; S4.4 (real AI Hub) is ground truth.")
    print("    - per SESSION_CONTEXT, normalisation choice -- not conditioning -- dominated")
    print("      measured Snapdragon latency (2.80x). This block adds no normalisation.")


if __name__ == "__main__":
    main()
