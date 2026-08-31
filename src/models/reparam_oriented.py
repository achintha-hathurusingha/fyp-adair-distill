"""S3.1 -- Reparameterizable oriented-kernel block.

Frequency-selective behaviour realised SPATIALLY, because `torch.fft` has no
ONNX op and attention is UNKNOWN on all three NPU backends. Trained as a rich
multi-branch bank; merged algebraically into ONE depthwise convolution for
deployment (RepVGG, Ding et al. CVPR 2021; RepLKNet, CVPR 2022).

Every design choice here was settled by measurement, not preference. Do not
re-litigate them without re-running the gate that fixed them:

* **Four orientations, kept.** S0.1 refit every kernel family against rain at
  controlled angles: the 4-orientation bank is within 0.09 dB of the
  unconstrained oracle at EVERY angle, and it is the only family that is. Plain
  axis-aligned low rank (rank2/rank3) matches it on near-vertical rain but
  collapses at 45 deg (-0.458 / -0.288 dB). Dropping the diagonals would look
  free on our corpora and fail on off-axis rain.
* **k = 11.** S0.1: the oriented-over-isotropic gain saturates by k=11
  (+0.384 dB vs +0.385 at k=15, +0.365 at k=7), and k=11 is where the oriented
  support first becomes a genuine proper subspace (97 of 121 taps) rather than
  the whole window. Consistent with the convolution-theorem result that
  7x7-11x11 carries essentially all of the optimal filter.
* **NO normalisation and NO nonlinearity between the branches and the sum.**
  This is the binding constraint, not a style choice: the merge is exact only
  if the branches combine linearly. Anything inserted there silently destroys
  the entire deployment argument (plan risk 2). `tests`/the smoke script guard
  it -- if you add a nonlinearity, merge exactness fails loudly.
* **Rain-only, in effect.** S0.1 measured orientation at +0.385 dB on derain and
  +0.009 / +0.001 on denoise / dehaze. This block is not expected to help the
  other two tasks, and a result claiming otherwise needs explaining.

S0.2 exported the stub form of this design: the merged graph is ONE `Conv` node,
with zero UNKNOWN and zero CAUTION ops on qnn/tflite/tensorrt, in FP32 and INT8.

Sign convention, the one place this is easy to get silently wrong: PyTorch's
`conv2d` is CROSS-CORRELATION, so composing two depthwise convs gives an
equivalent kernel equal to their FULL LINEAR CONVOLUTION (one kernel flipped),
not their correlation. A merge that gets this wrong still passes an
interior-only test; it fails on small inputs where the zero-padded boundary
dominates. `scripts/smoke_reparam_oriented.py` tests at 13x13 with k=11 for
exactly that reason.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def full_conv2d_depthwise(w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    """Equivalent single kernel of applying depthwise `w1` then depthwise `w2`.

    ``((x * w1) * w2) == x * (w1 FULL-CONV w2)`` for cross-correlation ``*``.
    w1: (C,1,h1,s1), w2: (C,1,h2,s2) -> (C,1,h1+h2-1,s1+s2-1)
    """
    c, _, h1, s1 = w1.shape
    _, _, h2, s2 = w2.shape
    w1p = F.pad(w1, (s2 - 1, s2 - 1, h2 - 1, h2 - 1))
    w1p = w1p.view(1, c, h1 + 2 * (h2 - 1), s1 + 2 * (s2 - 1))
    out = F.conv2d(w1p, torch.flip(w2, dims=(-2, -1)), groups=c)
    return out.view(c, 1, h1 + h2 - 1, s1 + s2 - 1)


def pad_to(w: torch.Tensor, k: int) -> torch.Tensor:
    """Centre-pad a (C,1,h,s) kernel to (C,1,k,k) -- RepLKNet's small-into-large."""
    h, s = w.shape[-2:]
    if h > k or s > k:
        raise ValueError(f"kernel {h}x{s} exceeds target {k}x{k}")
    return F.pad(w, ((k - s) // 2, (k - s + 1) // 2, (k - h) // 2, (k - h + 1) // 2))


def diagonal_mask(k: int, band: int, main: bool) -> torch.Tensor:
    """Diagonal band support. Per Freeman & Adelson (TPAMI 1991) a directional
    filter needs a directional SUPPORT, not merely a directional label -- which
    is precisely what AdaIR's own alpha/beta collapse to equality (AFLB3
    0.496/0.497) shows it failing to maintain."""
    m = torch.zeros(k, k)
    half = band // 2
    for i in range(k):
        j = i if main else (k - 1 - i)
        for dj in range(-half, half + 1):
            if 0 <= j + dj < k:
                m[i, j + dj] = 1.0
    return m


def delta_kernel_(weight: torch.Tensor) -> None:
    """In-place delta init for a depthwise kernel: centre tap 1, rest 0.

    Makes conv(x) = x, so a zero-initialised residual gate scales up a copy of
    the signal instead of a frozen random kernel. See the module docstring --
    this is the fix for the defect that killed B0V3-KD-K11.
    """
    with torch.no_grad():
        weight.zero_()
        kh, kw = weight.shape[-2:]
        weight[:, 0, kh // 2, kw // 2] = 1.0


class ReparamOrientedCore(nn.Module):
    """Multi-branch oriented depthwise bank collapsing to ONE depthwise conv."""

    def __init__(self, dim: int, k: int = 11, kp: int = 3) -> None:
        super().__init__()
        if k % 2 == 0 or kp % 2 == 0:
            raise ValueError(f"kernel sizes must be odd, got k={k}, kp={kp}")
        if kp > k:
            raise ValueError(f"kp={kp} must not exceed k={k}")
        self.dim, self.k, self.kp = dim, k, kp
        dw = dict(groups=dim, bias=False)

        # 0 deg: long horizontal then short vertical (anisotropic, separable)
        self.h_long = nn.Conv2d(dim, dim, (1, k), padding=(0, k // 2), **dw)
        self.h_short = nn.Conv2d(dim, dim, (kp, 1), padding=(kp // 2, 0), **dw)
        # 90 deg: long vertical then short horizontal
        self.v_long = nn.Conv2d(dim, dim, (k, 1), padding=(k // 2, 0), **dw)
        self.v_short = nn.Conv2d(dim, dim, (1, kp), padding=(0, kp // 2), **dw)
        # 45 / 135 deg: k x k, gradient-masked to a diagonal band
        self.d45 = nn.Conv2d(dim, dim, k, padding=k // 2, **dw)
        self.d135 = nn.Conv2d(dim, dim, k, padding=k // 2, **dw)
        self.register_buffer("m45", diagonal_mask(k, kp, True).view(1, 1, k, k))
        self.register_buffer("m135", diagonal_mask(k, kp, False).view(1, 1, k, k))
        self.iso = nn.Conv2d(dim, dim, kp, padding=kp // 2, **dw)

        # per-channel band coefficients; linear, therefore mergeable
        self.coef = nn.Parameter(torch.ones(5, dim))
        self.id_coef = nn.Parameter(torch.ones(dim))
        # Delta init, expressed through the bank: zero every branch and keep the
        # identity branch at 1, so the MERGED kernel is delta and core(x) = x.
        # Without this, a zero-init fuse scales up a frozen random bank -- the
        # defect that killed B0V3-KD-K11. See the module docstring.
        # Zero ONE factor of each separable pair, not both. The 0/90 branches
        # are compositions h_short(h_long(x)); if BOTH factors start at zero then
        # dL/d(h_long) is proportional to h_short = 0 and dL/d(h_short) is
        # proportional to h_long(x) = 0, so neither ever leaves zero and those two
        # orientations are dead for the whole run -- a "4-orientation bank" that
        # is really a 2-orientation bank. Zeroing the LONG factor alone still
        # makes the composition exactly zero at init (so the merged kernel is
        # delta and core(x) = x), while leaving the short factor at its default
        # init keeps the gradient path open. Verified in
        # scripts/verify_delta_init.py -- all seven branches must learn.
        with torch.no_grad():
            for c in (self.h_long, self.v_long,          # long factors only
                      self.d45, self.d135, self.iso):    # single convs
                c.weight.zero_()
        self._apply_mask()

    def _apply_mask(self) -> None:
        """Zero the off-band taps and keep them zero. The hook is load-bearing:
        without it the 45/135 branches drift into generic square kernels within
        a few steps and the orientation structure is decorative."""
        with torch.no_grad():
            self.d45.weight.mul_(self.m45)
            self.d135.weight.mul_(self.m135)
        self.d45.weight.register_hook(lambda g: g * self.m45)
        self.d135.weight.register_hook(lambda g: g * self.m135)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bands = (self.h_short(self.h_long(x)), self.v_short(self.v_long(x)),
                 self.d45(x), self.d135(x), self.iso(x))
        out = self.id_coef.view(1, -1, 1, 1) * x
        for i, b in enumerate(bands):
            out = out + self.coef[i].view(1, -1, 1, 1) * b
        return out

    @torch.no_grad()
    def merged(self) -> nn.Conv2d:
        """Collapse every branch into a single depthwise conv (exact)."""
        k, dim = self.k, self.dim
        ws = [
            pad_to(full_conv2d_depthwise(self.h_long.weight, self.h_short.weight), k),
            pad_to(full_conv2d_depthwise(self.v_long.weight, self.v_short.weight), k),
            pad_to(self.d45.weight * self.m45, k),
            pad_to(self.d135.weight * self.m135, k),
            pad_to(self.iso.weight, k),
        ]
        acc = sum(w * self.coef[i].view(-1, 1, 1, 1) for i, w in enumerate(ws))
        delta = torch.zeros(dim, 1, k, k, device=acc.device, dtype=acc.dtype)
        delta[:, 0, k // 2, k // 2] = 1.0
        acc = acc + delta * self.id_coef.view(-1, 1, 1, 1)

        conv = nn.Conv2d(dim, dim, k, padding=k // 2, groups=dim, bias=False)
        conv.weight.copy_(acc)
        return conv


class ReparamOrientedBlock(nn.Module):
    """Deployable oriented-mining block: mergeable bank + 1x1 channel fuse,
    zero-init residual (identity at init, the repo's idiom everywhere else).

    At deployment this is exactly TWO convs -- one depthwise k x k and one 1x1.
    The 1x1 cannot fold into the depthwise (different group structure).
    """

    def __init__(self, dim: int, k: int = 11, kp: int = 3) -> None:
        super().__init__()
        self.core = ReparamOrientedCore(dim, k, kp)
        self.fuse = nn.Conv2d(dim, dim, 1, bias=False)
        nn.init.zeros_(self.fuse.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fuse(self.core(x))

    @torch.no_grad()
    def merge(self) -> nn.Module:
        """Deployment form. Numerically equal to ``self`` at eval."""
        return MergedOrientedBlock(self.core.merged(), self.fuse)


class MergedOrientedBlock(nn.Module):
    """What ships: one depthwise conv, one 1x1, one add."""

    def __init__(self, conv: nn.Conv2d, fuse: nn.Conv2d) -> None:
        super().__init__()
        self.conv, self.fuse = conv, fuse

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fuse(self.conv(x))


class PlainLargeKernelBlock(nn.Module):
    """Matched CONTROL for ReparamOrientedBlock: one unconstrained depthwise
    k x k kernel, same 1x1 fuse, same zero-init residual, same deployed cost.

    It differs from the oriented block in exactly one way -- no oriented branch
    structure -- so the difference between them isolates orientation from
    receptive field. It needs no merge step: a single depthwise conv is already
    the deployment form, which is also the point (whatever the oriented bank
    merges into, this is the same shape).
    """

    def __init__(self, dim: int, k: int = 11) -> None:
        super().__init__()
        if k % 2 == 0:
            raise ValueError(f"kernel size must be odd, got {k}")
        self.dim, self.k = dim, k
        self.conv = nn.Conv2d(dim, dim, k, padding=k // 2, groups=dim, bias=False)
        self.fuse = nn.Conv2d(dim, dim, 1, bias=False)
        # Delta, NOT Kaiming: with fuse zero-initialised the conv receives no
        # gradient at step 0, so whatever it holds gets amplified unchanged.
        # Kaiming here meant amplifying a random 11x11 blur into the decoder,
        # which is what killed B0V3-KD-K11 (-3.026 dB on dehaze).
        delta_kernel_(self.conv.weight)
        nn.init.zeros_(self.fuse.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fuse(self.conv(x))

    @torch.no_grad()
    def merge(self) -> nn.Module:
        """Already in deployment form; returned for interface parity."""
        return MergedOrientedBlock(self.conv, self.fuse)
