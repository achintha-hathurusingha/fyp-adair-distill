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

__all__ = ["LaplacianFrequencyGate", "dark_channel_prior"]


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
