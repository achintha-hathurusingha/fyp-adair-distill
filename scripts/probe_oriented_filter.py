"""OrientedStreakGate -- a rain-specific block derived from the actual
mathematical structure of the degradation, not borrowed as a named
black-box architecture.

Rain-streak decomposition literature (Kang, Lin & Lin, "Automatic
Single-Image-Based Rain Streak Removal via Image Decomposition," IEEE TIP
2012; Li, Tan, Guo, Lu & Brown, "Rain Streak Removal Using Layer Priors,"
CVPR 2016) models a rainy image as I = B + R, where R is a SPARSE,
DIRECTIONALLY-ANISOTROPIC high-frequency layer: real streaks have a
dominant orientation (near-vertical, with some scatter), not an isotropic
one. NAFNet's 3x3 depthwise convs are orientation-agnostic by construction
(a square kernel has 4-fold symmetry, no preferred angle) -- structurally
the wrong shape for a signal whose defining property IS its orientation.

Freeman & Adelson, "The Design and Use of Steerable Filters," IEEE TPAMI
13(9), 1991 -- gives the actual theory: a small FIXED basis of directional
derivative filters can be linearly combined (with angle-dependent,
learnable weights) to synthesize a filter response at any orientation,
without needing a separate learned kernel per angle. Implemented here with
a small bank of elongated depthwise conv kernels at fixed orientations
(0/45/90/135 degrees -- a first-order approximation to a full steerable
basis, cheap enough for a mobile student), each LEARNED (not fixed
Gaussian-derivative coefficients, since the student has capacity budget to
spare here and gets to adapt band centers to the real training
distribution), combined via a learned per-channel gate. Zero-init residual,
identity at init, same idiom as every other addition in theory_blocks.py.

All that's below is elongated Conv2d (kernel_size=(1,k) / (k,1) / diagonal
via a full kxk kernel masked to a diagonal band) + the SE-style channel
gate -- no new op types vs. what NAFNet already ships, so no separate
op-coverage risk category is expected. Verified below anyway rather than
assumed.
"""
import sys
sys.path.insert(0, ".")
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.export.op_coverage import op_histogram, render_markdown


class OrientedStreakGate(nn.Module):
    def __init__(self, dim: int, reduction: int = 8, k: int = 7):
        super().__init__()
        hidden = max(1, dim // reduction)
        self.reduce = nn.Conv2d(dim, hidden, 1, bias=False)
        # 0 deg (horizontal run, catches near-vertical streak edges) and
        # 90 deg (vertical run) via elongated rectangular kernels; 45/135
        # deg via a square kernel with the off-basis half masked to zero,
        # fixed at init (not learned to move off the diagonal) so it starts
        # as a genuine diagonal filter, not a generic square one.
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
        """Zero every tap off the chosen diagonal band, in-place, once at
        init -- gives conv_45/conv_135 a genuine oriented starting point
        (per Freeman & Adelson: a directional filter needs a directional
        support, not just directional labeling) rather than an isotropic
        kxk kernel that happens to be called '45 degrees'.
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


torch.manual_seed(0)
m = OrientedStreakGate(32).eval()
x = torch.randn(2, 32, 48, 48, requires_grad=True)
y = m(x)
assert torch.allclose(y, x), "not identity at init"
y.sum().backward()
assert x.grad is not None and torch.isfinite(x.grad).all()
# confirm the diagonal masks actually survive a step (gradient doesn't reintroduce off-diagonal taps)
opt = torch.optim.SGD(m.parameters(), lr=1.0)
opt.zero_grad(); m(x).sum().backward(); opt.step()
k = 7
mask_main = torch.zeros(k, k)
for i in range(k):
    for dj in (-1, 0, 1):
        jj = i + dj
        if 0 <= jj < k:
            mask_main[i, jj] = 1.0
off_diag = m.conv_45.weight[0, 0] * (1 - mask_main)
assert torch.allclose(off_diag, torch.zeros_like(off_diag)), "off-diagonal taps became non-zero after a step"
print(f"OrientedStreakGate: identity at init OK, grad finite, diagonal mask holds after a step, "
     f"params={sum(p.numel() for p in m.parameters())}")

m_probe = OrientedStreakGate(32).eval()
dummy = torch.randn(1, 32, 64, 64)
onnx_path = "runs/oriented_probe.onnx"
torch.onnx.export(m_probe, dummy, onnx_path, opset_version=17, input_names=["x"], output_names=["y"])
hist = op_histogram(onnx_path)
print("\n--- OrientedStreakGate ONNX op histogram ---")
for op, count in sorted(hist.items()):
    print(f"  {op}: {count}")
print("\n" + render_markdown(onnx_path))
