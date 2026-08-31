"""Frequency-domain diagnostic: released AdaIR vs Himeth's repaired
checkpoint, same image, same AFLB positions.

Reproduces TEST18's diagnostic panels (log FFT magnitude / mask / raw_low /
raw_high) but runs them on BOTH models so the effect of the mask repair is
visible rather than described.

  RELEASED   adair3d.ckpt, upstream mask: h_ = (h // 128 * rate).int()
             floors to 0 for every feature map under 256px -> empty mask,
             FFT and IFFT cancel, module degenerates to torch.abs().

  REPAIRED   C_full_soft_real/final.pt, soft mask over NORMALISED radius:
             M = sigmoid((alpha - u)/tau) * sigmoid((beta - v)/tau)
             resolution-independent and differentiable.

CPU only -- the GPU is training.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, "/home/minura/FYP/Workspace/Himeth/AdaIR/AdaIR")
sys.path.insert(0, "/home/minura/FYP/Workspace/Himeth/AdaIR/finetune")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from net.model import AdaIR, FreModule                      # noqa: E402
from freq_fix import load_adair_state                        # noqa: E402

DEV = "cpu"
SIZE = 256
H = "/home/minura/FYP/Workspace/Himeth"
RELEASED = f"{H}/AdaIR/weights/adair3d.ckpt"
REPAIRED = f"{H}/AdaIR/runs/finetune/C_full_soft_real/final.pt"
OUT = "/home/minura/fyp-adair-distill/reports/student_v3"
os.makedirs(OUT, exist_ok=True)

IMAGES = {
    "Haze": "/home/minura/fyp-adair-distill/data/dehaze/RESIDE/SOTS/outdoor/input/0001_0.8_0.2.jpg",
    "Rain": f"{H}/data/rain100L/rain100L_test/Rain100L/rainy/rain-050.png",
}
NOISE_SRC = "/home/minura/fyp-adair-distill/data/dehaze/RESIDE/SOTS/outdoor/target/0001.png"


def load_t(p):
    a = np.asarray(Image.open(p).convert("RGB"))
    h, w = a.shape[:2]
    y, x = max(0, (h - SIZE) // 2), max(0, (w - SIZE) // 2)
    a = a[y:y + SIZE, x:x + SIZE]
    if a.shape[0] != SIZE or a.shape[1] != SIZE:
        a = np.asarray(Image.fromarray(a).resize((SIZE, SIZE)))
    return torch.from_numpy(a.astype(np.float32) / 255.).permute(2, 0, 1)[None]


def make_noisy():
    t = load_t(NOISE_SRC)
    g = torch.Generator().manual_seed(0)
    return (t + torch.randn(t.shape, generator=g) * (25 / 255.)).clamp(0, 1)


CAP: dict = {}


def instrument(net, mode, tau=0.05):
    """Replace FreModule.fft with a capturing version of the requested mask."""
    def make(name):
        def fft(self, x, n=128):
            conv = self.conv1(x)
            h, w = conv.shape[-2:]
            rate = self.rate_conv(F.adaptive_avg_pool2d(conv, 1)).sigmoid()
            a, b = rate[:, 0:1], rate[:, 1:2]
            if mode == "upstream":
                mask = torch.zeros_like(conv)
                for i in range(mask.shape[0]):
                    h_ = int((h // n * a[i, 0, 0, 0]).int())
                    w_ = int((w // n * b[i, 0, 0, 0]).int())
                    if h_ > 0 and w_ > 0:
                        mask[i, :, h // 2 - h_:h // 2 + h_, w // 2 - w_:w // 2 + w_] = 1
            else:
                iy = torch.arange(h, dtype=conv.dtype); ix = torch.arange(w, dtype=conv.dtype)
                u = (iy - h // 2).abs() / max(h // 2, 1)
                v = (ix - w // 2).abs() / max(w // 2, 1)
                mask = (torch.sigmoid((a - u.view(1, 1, h, 1)) / tau) *
                        torch.sigmoid((b - v.view(1, 1, 1, w)) / tau))
            fft_ = self.shift(torch.fft.fft2(conv, norm="forward", dim=(-2, -1)))
            high = torch.abs(torch.fft.ifft2(self.unshift(fft_ * (1 - mask)), norm="forward", dim=(-2, -1)))
            low = torch.abs(torch.fft.ifft2(self.unshift(fft_ * mask), norm="forward", dim=(-2, -1)))
            CAP[(mode, name)] = {
                "fft": torch.log1p(torch.abs(fft_)[0].mean(0)).numpy(),
                "mask": mask[0].mean(0).numpy(),
                "low": low[0].mean(0).numpy(),
                "high": high[0].mean(0).numpy(),
                "alpha": float(a.mean()), "beta": float(b.mean()),
                "active": float((mask > 0.5).float().mean()),
            }
            return high, low
        return fft

    import types
    for attr, nm in (("fre1", "AFLB1"), ("fre2", "AFLB2"), ("fre3", "AFLB3")):
        m = getattr(net, attr)
        m.fft = types.MethodType(make(nm), m)


def build(path, mode):
    net = AdaIR(decoder=True)
    net.load_state_dict(load_adair_state(path), strict=True)
    net.eval()
    instrument(net, mode)
    return net


print("loading models (CPU)...", flush=True)
rel = build(RELEASED, "upstream")
rep = build(REPAIRED, "soft")

cases = {k: load_t(v) for k, v in IMAGES.items() if os.path.exists(v)}
cases["Noise"] = make_noisy()
print("cases:", list(cases), flush=True)

for deg, x in cases.items():
    CAP.clear()
    with torch.no_grad():
        rel(x); rep(x)

    fig, axes = plt.subplots(3, 8, figsize=(20.5, 8.2), dpi=155)
    cols = [("fft", "log FFT magnitude", "viridis"), ("mask", "mask", "viridis"),
            ("low", "raw_low", "viridis"), ("high", "raw_high", "viridis")]

    for r, aflb in enumerate(["AFLB1", "AFLB2", "AFLB3"]):
        for blk, (mode, tag) in enumerate([("upstream", "RELEASED"), ("soft", "REPAIRED")]):
            d = CAP.get((mode, aflb))
            for c, (key, title, cm) in enumerate(cols):
                ax = axes[r, blk * 4 + c]
                im = ax.imshow(d[key], cmap=cm)
                ax.set_xticks([]); ax.set_yticks([])
                if r == 0:
                    ax.set_title(title, fontsize=9, color="#10141C")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(labelsize=6)
            if blk == 0:
                axes[r, 0].set_ylabel(aflb, fontsize=11, fontweight="bold", color="#1F6F7A")
        # per-row annotation of what the mask actually is
        du, ds = CAP[("upstream", aflb)], CAP[("soft", aflb)]
        axes[r, 1].set_xlabel(f"active {du['active']:.6f}", fontsize=7.5, color="#A6423A")
        axes[r, 5].set_xlabel(f"active {ds['active']:.3f}   α={ds['alpha']:.3f} β={ds['beta']:.3f}",
                              fontsize=7.5, color="#2E7D5B")

    # block headers + divider
    fig.text(0.255, 0.965, "RELEASED  adair3d.ckpt  —  mask floors to zero",
             ha="center", fontsize=12.5, fontweight="bold", color="#A6423A")
    fig.text(0.745, 0.965, "REPAIRED  C_full_soft_real  —  soft mask, resolution-independent",
             ha="center", fontsize=12.5, fontweight="bold", color="#2E7D5B")
    fig.add_artist(plt.Line2D([0.503, 0.503], [0.03, 0.945], color="#DDE2E8", lw=1.6))
    fig.suptitle(f"{deg} — what the frequency module actually computes, before and after the repair",
                 fontsize=13.5, fontweight="bold", y=0.995, color="#10141C")
    fig.tight_layout(rect=[0.01, 0.02, 0.99, 0.945])
    p = f"{OUT}/freq_compare_{deg.lower()}.png"
    fig.savefig(p, facecolor="white", bbox_inches="tight"); plt.close(fig)
    print(f"  {deg}: released active={CAP[('upstream','AFLB3')]['active']:.6f}  "
          f"repaired active={CAP[('soft','AFLB3')]['active']:.3f}  -> {p}", flush=True)

print("done")
