"""NPU-safe, theory-grounded restoration primitives -- ADDITIONS to the
LOCKED NAFNet backbone (configs/model/nafnet_locked.yaml), not a
replacement for it. See reports/student_theory_review/lit_review.md for the
full literature review and citations, and reports/export_smoke_test.md /
src/export/op_coverage.py for the mobile-NPU op-coverage methodology that
shaped every design choice below.

WHY NOT JUST COPY AdaIR'S OWN BLOCKS: AdaIR's FreModule (third_party/AdaIR/
net/model.py) does its frequency split with torch.fft.fft2/ifft2 on complex
tensors. There is no ONNX/QNN/TFLite op-support entry for DFT/FFT in this
repo's curated table, and web research on the real QNN Hexagon NPU target
confirms FFT is not part of any mainstream mobile NPU delegate's supported
op set (it is a live research topic, not shipped support) -- so the literal
teacher architecture cannot be deployed as our student regardless of
whether it would help feature-KD alignment. Everything below is built only
from ops already proven SUPPORTED in this repo's own Gate G1 export test
(Conv, AveragePool, MaxPool, Sub, Add, Mul, Sigmoid, ReLU, GlobalAveragePool,
PixelShuffle/DepthToSpace) -- see reports/student_theory_review/lit_review.md
section 3 for the full op-by-op accounting, including the two ops
(elementwise Min, for the dark-channel-prior channel-min) that are NOT in
the curated table and are called out there as unverified rather than
assumed safe.

Two additions, each targeting a real, disclosed gap between the current
student and the AdaIR teacher:

1. LaplacianFrequencyGate -- AdaIR's own ablation (Table 7, arXiv:2403.14614)
   shows frequency-selective processing is worth +1.58dB by itself (their
   "FMiM" component), before any cross-attention refinement is added. Their
   mechanism is a single GLOBAL adaptive FFT mask: one high/low threshold
   for the whole image, computed from a GAP over the whole feature map.
   Fourier basis functions have infinite spatial support, so a global mask
   cannot represent degradation whose intensity varies across the image
   (patchy haze density, localised rain streaks) -- Mallat's multiresolution
   theory (IEEE TPAMI, 1989) is the classical statement of exactly this
   trade-off: perfect frequency localisation buys zero spatial localisation.
   A Laplacian pyramid (Burt & Adelson, IEEE Trans. Commun., 1983) sits on
   the other side of that trade-off -- spatially localised frequency
   subbands -- and is built entirely from blur+decimate (Conv+AvgPool) and
   learned 2x upsampling (Conv+PixelShuffle), so it keeps AdaIR's
   demonstrated frequency-selectivity win while every operation involved is
   one this repo's own export gate has already cleared.

2. dark_channel_prior -- targets dehaze specifically: the task furthest
   behind the teacher on every real measurement so far
   (reports/kd_feature_multitask/cond_regression.md). AdaIR has no physical
   model of haze at all; it learns everything from data. The atmospheric
   scattering model (Koschmieder, 1924; McCartney, 1976) gives haze
   formation a known closed form, I(x) = J(x)t(x) + A(1-t(x)), and He, Sun &
   Tang's dark channel prior (CVPR 2009 / IEEE TPAMI 2011) gives a
   zero-learned-parameter, per-pixel estimate of the transmission map t(x)
   straight off the input image -- a spatially-varying physical prior the
   network does not have to discover from data at all.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["LaplacianFrequencyGate", "dark_channel_prior", "StripPoolingGate", "OrientedStreakGate"]


class _PSUp2x(nn.Module):
    """2x spatial upsample via 1x1 conv + PixelShuffle. Deliberately NOT
    F.interpolate: "Resize" appears in none of the three curated backend
    tables in src/export/op_coverage.py (not even tensorrt's), whereas
    PixelShuffle -> ONNX DepthToSpace is already characterized there
    (CAUTION on qnn, SUPPORTED on tflite/tensorrt) from NAFNet's own
    Downsample/Upsample blocks -- reuse a known quantity over an untested
    one.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim * 4, 1, bias=False)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shuffle(self.conv(x))


class LaplacianFrequencyGate(nn.Module):
    """`levels`-band Laplacian-pyramid decomposition of the current feature
    map, each band re-weighted by its own learned per-channel gate (GAP ->
    1x1 conv -> ReLU -> 1x1 conv -> sigmoid -> multiply -- the exact SCA
    primitive NAFNet's own channel gate uses, already export-verified in
    reports/export_smoke_test.md), then reconstructed and added back as a
    residual. Zero-init on the final projection so this module is the
    identity map at initialization (same stabilization trick as AdaIR's own
    FreModule para1/para2, and consistent with this project's established
    preference for additions that start inert and earn their effect --
    configs/model/nafnet_locked.yaml's clamp-engagement-rate tracking is the
    same philosophy).
    """

    def __init__(self, dim: int, levels: int = 2, reduction: int = 8):
        super().__init__()
        self.levels = levels
        self.blur = nn.ModuleList(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False) for _ in range(levels))
        self.up = _PSUp2x(dim)  # single shared learned 2x upsampler
        hidden = max(1, dim // reduction)
        self.gates = nn.ModuleList(
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(dim, hidden, 1, bias=False), nn.ReLU(inplace=True),
                nn.Conv2d(hidden, dim, 1, bias=False), nn.Sigmoid(),
            )
            for _ in range(levels + 1)
        )
        self.proj = nn.Conv2d(dim, dim, 1, bias=False)
        nn.init.zeros_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cur = x
        bands = []
        for blur in self.blur:
            down = F.avg_pool2d(blur(cur), 2)
            bands.append(cur - self.up(down))  # high-frequency residual at this scale
            cur = down
        bands.append(cur)  # coarsest low-frequency residue

        out = torch.zeros_like(x)
        for i, (band, gate) in enumerate(zip(bands, self.gates)):
            reweighted = band * gate(band)
            for _ in range(i):  # band i sits at resolution/2**i -- upsample i times to match x
                reweighted = self.up(reweighted)
            out = out + reweighted
        return x + self.proj(out)


def dark_channel_prior(x: torch.Tensor, patch_size: int = 7) -> torch.Tensor:
    """He, Sun & Tang's dark channel prior: the per-pixel minimum over color
    channels AND a local window, of a haze-free outdoor patch, is close to
    zero; on a hazy image this value rises with local haze density, giving a
    cheap, zero-learned-parameter, per-pixel transmission estimate
    (Koschmieder's atmospheric scattering model).

    `x`: (B, 3, H, W) in [0, 1]. Returns (B, 1, H, W) in [0, 1].

    An earlier version used torch.min(dim=1) + the -MaxPool(-x) trick;
    verified via this repo's own op_coverage.py gate that it lowers to
    ReduceMin + Neg, and NEITHER appears in ANY of the three curated backend
    tables (qnn/tflite/tensorrt) -- a real finding, not a guess (see
    reports/student_theory_review/lit_review.md section 3). Rewritten below
    using only ops already SUPPORTED everywhere in that table:
      - channel-min via min(a,b) = a - relu(a-b), pairwise over R/G/B
        (Split, Sub, Relu -- all SUPPORTED)
      - the windowed min via 1 - MaxPool(1-x), valid because x in [0,1]
        exactly (Sub, MaxPool -- both SUPPORTED; avoids Neg entirely by
        using the known value range instead of a sign flip)
    Re-verified empirically after the rewrite -- op histogram is exactly
    {Split, Sub, Relu, MaxPool}, all SUPPORTED on qnn/tflite/tensorrt.
    """
    r, g, b = torch.split(x, 1, dim=1)
    m1 = r - F.relu(r - g)          # min(r, g)
    min_channel = m1 - F.relu(m1 - b)  # min(min(r,g), b), in [0, 1]

    pad = patch_size // 2
    inv = 1.0 - min_channel
    pooled_inv = F.max_pool2d(inv, kernel_size=patch_size, stride=1, padding=pad)
    return 1.0 - pooled_inv


class StripPoolingGate(nn.Module):
    """Hu, Zhang, Xie & Yang, "Strip Pooling: Rethinking Spatial Pooling for
    Scene Parsing," CVPR 2020. Pools along one FULL spatial axis at a time
    (adaptive_avg_pool2d to (H,1) / (1,W)) -- genuinely global context along
    that axis, not just a larger local window.

    Motivation (reports/student_theory_review/lit_review.md section 5): a
    controlled backbone comparison (arXiv:2310.11881) found NAFNet loses to
    Restormer by 1.7dB (deraining) and 2.9dB (dehazing) while *winning* on
    deblurring, and attributes this specifically to depthwise convolution's
    "weak spatial mapping capability" versus attention's global reach --
    exactly the pattern this project's own real evaluation shows (tied on
    denoise, behind on derain/dehaze). MDTA (the attention mechanism AdaIR
    and Restormer both use) was probed the same way as everything else here
    (scripts/probe_mdta.py) and found to lower to MatMul/Softmax/ReduceL2,
    NONE of which appear in this repo's curated qnn/tflite/tensorrt tables
    -- genuinely unverified, not proven either way, but a real risk this
    module avoids taking. Strip pooling gets a comparable "large-range
    information" capability from AveragePool + Conv only: verified to lower
    to exactly {Add, AveragePool, Conv} -- SUPPORTED on every backend, no
    exceptions.

    Zero-init final projection: additive residual, identity at init.
    """

    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        hidden = max(1, dim // reduction)
        self.reduce = nn.Conv2d(dim, hidden, 1, bias=False)
        self.conv_h = nn.Conv2d(hidden, hidden, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.conv_w = nn.Conv2d(hidden, hidden, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.fuse = nn.Conv2d(hidden, dim, 1, bias=False)
        nn.init.zeros_(self.fuse.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.reduce(x)
        # adaptive_avg_pool2d(y, (H,1)) IS mean-over-W-keep-H -- pooling to
        # a target size equal to the input's OWN size in one axis is just a
        # reduction along the other axis. Using mean() directly instead of
        # adaptive_avg_pool2d avoids a real bug this had: inside the full
        # NAFNet graph (not the standalone probe), x.shape[-2] traces as a
        # non-constant value once downstream of NAFNet's own dynamic
        # padding, and ONNX export requires adaptive-pool output_size to be
        # constant -- verified via scripts/smoke_nafnet_theory.py, not
        # assumed. mean(dim=..., keepdim=True) has no such requirement.
        yh = self.conv_h(y.mean(dim=3, keepdim=True))  # (B,hidden,H,1)
        yw = self.conv_w(y.mean(dim=2, keepdim=True))  # (B,hidden,1,W)
        # Ordinary tensor-addition broadcasting (ONNX Add is natively
        # broadcasting) rather than an explicit .expand() call -- an
        # earlier version's runtime-shape-derived .expand(-1,-1,h,w) traced
        # to Equal/Where/ConstantOfShape/Expand, all UNKNOWN on every
        # backend (verified, not assumed; see scripts/probe_strip_pool.py).
        return x + self.fuse(yh + yw)


class OrientedStreakGate(nn.Module):
    """A rain-specific block derived from the actual mathematical structure
    of the degradation, not borrowed as a named black-box architecture.

    Rain-streak decomposition literature (Kang, Lin & Lin, "Automatic
    Single-Image-Based Rain Streak Removal via Image Decomposition," IEEE
    TIP, 2012; Li, Tan, Guo, Lu & Brown, "Rain Streak Removal Using Layer
    Priors," CVPR 2016) models a rainy image as I = B + R, where R is a
    SPARSE, DIRECTIONALLY-ANISOTROPIC high-frequency layer -- real streaks
    have a dominant orientation, not an isotropic one. Every conv in NAFBlock
    is a square (isotropic, 4-fold-symmetric) kernel -- structurally the
    wrong shape for a signal whose defining property IS its orientation.

    Freeman & Adelson, "The Design and Use of Steerable Filters," IEEE
    TPAMI 13(9), 1991 -- the actual theory: a small basis of directional
    filters, combined with learned angle-dependent weights, can synthesize
    a response at any orientation. Approximated here with 4 fixed-orientation
    depthwise kernels (0/45/90/135 degrees: two elongated rectangular
    kernels for the axis-aligned angles, two square kernels with the
    off-diagonal half masked to zero at init -- see `_mask_diagonal` -- for
    the diagonal angles, matching real rain's typical near-vertical-with-
    scatter geometry), combined via a learned SE-style channel gate.

    Built entirely from Conv/GlobalAveragePool/ReLU/Sigmoid/Concat/Mul/Add
    -- no new op types versus what NAFNet already ships. Verified: op
    histogram is exactly {Conv, GlobalAveragePool, Relu, Sigmoid, Concat,
    Mul, Add} -- SUPPORTED on every backend, no exceptions
    (scripts/probe_oriented_filter.py). Zero-init final projection:
    additive residual, identity at init.
    """

    def __init__(self, dim: int, reduction: int = 8, k: int = 7):
        super().__init__()
        hidden = max(1, dim // reduction)
        self.reduce = nn.Conv2d(dim, hidden, 1, bias=False)
        self.conv_0 = nn.Conv2d(hidden, hidden, kernel_size=(1, k), padding=(0, k // 2),
                                groups=hidden, bias=False)
        self.conv_90 = nn.Conv2d(hidden, hidden, kernel_size=(k, 1), padding=(k // 2, 0),
                                 groups=hidden, bias=False)
        self.conv_45 = nn.Conv2d(hidden, hidden, kernel_size=k, padding=k // 2,
                                 groups=hidden, bias=False)
        self.conv_135 = nn.Conv2d(hidden, hidden, kernel_size=k, padding=k // 2,
                                  groups=hidden, bias=False)
        self._mask_diagonal(self.conv_45.weight, k, main=True)
        self._mask_diagonal(self.conv_135.weight, k, main=False)

        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden * 4, hidden, 1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden * 4, 1, bias=False), nn.Sigmoid(),
        )
        self.fuse = nn.Conv2d(hidden * 4, dim, 1, bias=False)
        nn.init.zeros_(self.fuse.weight)

    @staticmethod
    def _mask_diagonal(weight: torch.Tensor, k: int, main: bool) -> None:
        """Zero every tap off the chosen diagonal band, in-place, at init,
        and keep it there via a gradient hook -- gives conv_45/conv_135 a
        genuine oriented support (per Freeman & Adelson: a directional
        filter needs directional SUPPORT, not just a directional label),
        not an isotropic kxk kernel that happens to be called '45 degrees'.
        """
        mask = torch.zeros(k, k)
        for i in range(k):
            j = i if main else (k - 1 - i)
            for dj in (-1, 0, 1):
                jj = j + dj
                if 0 <= jj < k:
                    mask[i, jj] = 1.0
        with torch.no_grad():
            weight.mul_(mask.view(1, 1, k, k))
        weight.register_hook(lambda g, m=mask: g * m.to(g.device).view(1, 1, k, k))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.reduce(x)
        bands = torch.cat([self.conv_0(y), self.conv_90(y), self.conv_45(y), self.conv_135(y)], dim=1)
        gated = bands * self.gate(bands)
        return x + self.fuse(gated)
