"""Can frequency-domain restoration be REALISED as a small spatial kernel?

The idea under test (not mine -- the user's, and it is a better one than
the spectral-descriptor route I tested in spatial_converter.py):

    Use FFT at analysis time to learn WHAT frequency response actually
    restores each degradation. Then map that response into a SPATIAL
    operator and ship only that. By the convolution theorem, multiplying
    by H(u,v) in frequency is exactly convolving by h = F^-1(H) in space,
    so the frequency behaviour can in principle be carried by convolutions
    alone -- no FFT at inference.

The catch is quantitative, not conceptual. Multiplication in frequency
corresponds to CIRCULAR convolution with a kernel of FULL support (as
large as the image). It collapses to a practical k x k convolution only if
H is smooth enough that h decays fast. So the whole idea reduces to one
measurable question, which is what this script answers:

    HOW MUCH OF THE OPTIMAL FILTER'S ENERGY FITS IN A k x k WINDOW?

Method, per degradation:
  1. Optimal linear frequency-domain restoration filter, estimated over
     many paired (degraded, clean) crops -- the standard cross-spectral
     (Wiener) estimator:
            H(u,v) = E[ F_clean * conj(F_deg) ] / E[ |F_deg|^2 ]
     This is the BEST any linear frequency-domain filter can do for that
     degradation, so it upper-bounds what such a block could learn.
  2. h = real( ifft2( H ) ), centred, i.e. the exact spatial equivalent.
  3. Energy concentration: fraction of sum(h^2) inside a centred k x k
     window, for a range of k. This is the deployability curve.
  4. Actual restoration quality of the TRUNCATED k x k kernel vs the full
     filter, in PSNR on held-out crops -- because energy concentration is
     a proxy and the thing we actually care about is output quality.

Honest framing of what a positive result would and would not mean. A
compact kernel shows a LINEAR frequency-domain operation is realisable
spatially -- it does not show that the nonlinear, content-adaptive
frequency behaviour a network might learn is. It bounds the linear case,
which is the case the convolution theorem actually covers.

Same-scene synthetic degradations so degradation, not corpus, is the only
variable (same design as spectral_samescene.py).
"""
from __future__ import annotations

import sys, os, glob, random
sys.path.insert(0, ".")
sys.path.insert(0, "/home/minura/FYP/Workspace/Himeth/scripts/distill")

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from degradations import add_noise, add_haze, add_rain  # noqa: E402

DATA = "/home/minura/fyp-adair-distill/data"
N_TRAIN, N_TEST, PATCH, SEED = 300, 80, 128, 0
KS = [3, 5, 7, 9, 11, 15, 21, 31, 45, 63]
random.seed(SEED); np.random.seed(SEED)


def center_crop(a, n=PATCH):
    h, w = a.shape[:2]
    if h < n or w < n:
        s = max(n / h, n / w)
        a = np.asarray(Image.fromarray(a.astype(np.uint8)).resize(
            (max(n, int(w * s + 1)), max(n, int(h * s + 1)))), dtype=np.uint8)
        h, w = a.shape[:2]
    y, x = (h - n) // 2, (w - n) // 2
    return a[y:y + n, x:x + n]


def degrade(clean, task, r):
    if task == "denoise":
        return np.asarray(add_noise(clean, r, sigma=float(r.choice([15, 25, 50]))))
    if task == "dehaze":
        return np.asarray(add_haze(clean, r))
    return np.asarray(add_rain(clean, r))


def psnr(a, b):
    m = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if m <= 1e-12 else 10 * np.log10(255.0 ** 2 / m)


pool = sorted(glob.glob(f"{DATA}/Train/Denoise/*"))
random.shuffle(pool)
tr_files, te_files = pool[:N_TRAIN], pool[N_TRAIN:N_TRAIN + N_TEST]
print(f"scene-disjoint: {len(tr_files)} train / {len(te_files)} test crops\n", flush=True)

TASKS = ["denoise", "dehaze", "derain"]
results = {}

for task in TASKS:
    # ---- 1. optimal linear frequency filter (cross-spectral estimator) ----
    num = np.zeros((PATCH, PATCH), dtype=np.complex128)
    den = np.zeros((PATCH, PATCH), dtype=np.float64)
    for p in tr_files:
        try:
            clean = center_crop(np.asarray(Image.open(p).convert("RGB")))
        except Exception:
            continue
        r = np.random.default_rng(abs(hash(os.path.basename(p))) % (2**31))
        deg = degrade(clean, task, r)
        if deg.shape != clean.shape:
            continue
        c = clean.astype(np.float64).mean(2); d = deg.astype(np.float64).mean(2)
        c -= c.mean(); d -= d.mean()
        Fc, Fd = np.fft.fft2(c), np.fft.fft2(d)
        num += Fc * np.conj(Fd)
        den += (np.abs(Fd) ** 2)
    H = num / (den + 1e-8)

    # ---- 2. exact spatial equivalent, centred ----
    h = np.real(np.fft.ifft2(H))
    h = np.fft.fftshift(h)                      # DC at centre -> kernel centred
    total_energy = float((h ** 2).sum())

    # ---- 3. energy concentration vs kernel size ----
    c0 = PATCH // 2
    conc = {}
    for k in KS:
        r_ = k // 2
        win = h[c0 - r_:c0 + r_ + 1, c0 - r_:c0 + r_ + 1]
        conc[k] = float((win ** 2).sum() / total_energy)

    # ---- 4. actual PSNR of truncated kernel vs full filter, held out ----
    def apply_full(dimg):
        out = np.zeros_like(dimg, dtype=np.float64)
        for ch in range(3):
            x = dimg[..., ch].astype(np.float64)
            mu = x.mean()
            out[..., ch] = np.real(np.fft.ifft2(np.fft.fft2(x - mu) * H)) + mu
        return np.clip(out, 0, 255)

    def apply_trunc(dimg, k):
        r_ = k // 2
        ker = np.zeros((PATCH, PATCH))
        ker[c0 - r_:c0 + r_ + 1, c0 - r_:c0 + r_ + 1] = \
            h[c0 - r_:c0 + r_ + 1, c0 - r_:c0 + r_ + 1]
        Hk = np.fft.fft2(np.fft.ifftshift(ker))   # truncated kernel -> its own response
        out = np.zeros_like(dimg, dtype=np.float64)
        for ch in range(3):
            x = dimg[..., ch].astype(np.float64)
            mu = x.mean()
            out[..., ch] = np.real(np.fft.ifft2(np.fft.fft2(x - mu) * Hk)) + mu
        return np.clip(out, 0, 255)

    p_deg, p_full, p_k = [], [], {k: [] for k in KS}
    for p in te_files:
        try:
            clean = center_crop(np.asarray(Image.open(p).convert("RGB")))
        except Exception:
            continue
        r = np.random.default_rng(abs(hash("te" + os.path.basename(p))) % (2**31))
        deg = degrade(clean, task, r)
        if deg.shape != clean.shape:
            continue
        p_deg.append(psnr(clean, deg))
        p_full.append(psnr(clean, apply_full(deg)))
        for k in KS:
            p_k[k].append(psnr(clean, apply_trunc(deg, k)))

    results[task] = {
        "conc": conc, "h": h,
        "psnr_deg": float(np.mean(p_deg)),
        "psnr_full": float(np.mean(p_full)),
        "psnr_k": {k: float(np.mean(v)) for k, v in p_k.items()},
    }

    print(f"=== {task} ===")
    print(f"  degraded input          {results[task]['psnr_deg']:6.2f} dB")
    print(f"  full frequency filter   {results[task]['psnr_full']:6.2f} dB "
          f"({results[task]['psnr_full']-results[task]['psnr_deg']:+.2f})")
    print(f"  {'k':>3}  {'energy in kxk':>13}  {'PSNR':>7}  {'vs full':>8}")
    for k in KS:
        print(f"  {k:3d}  {conc[k]*100:12.1f}%  {results[task]['psnr_k'][k]:7.2f}  "
              f"{results[task]['psnr_k'][k]-results[task]['psnr_full']:+8.2f}")
    print(flush=True)

# ------------------------------------------------------------------ figure
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
colors = {"denoise": "#9467bd", "dehaze": "#ff7f0e", "derain": "#2ca02c"}

ax = axes[0]
for t in TASKS:
    ax.plot(range(len(KS)), [results[t]["conc"][k] * 100 for k in KS], "o-", color=colors[t], label=t, lw=2)
ax.axhline(95, ls="--", c="#888", lw=1)
ax.text(len(KS)-1, 95.6, "95%", ha="right", fontsize=9, color="#666")
ax.set_xticks(range(len(KS))); ax.set_xticklabels(KS)
ax.set_xlabel("spatial kernel size k"); ax.set_ylabel("% of filter energy inside k x k")
ax.set_title("Is the optimal filter compact in space?", fontsize=11, weight="bold")
ax.legend(); ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False)

ax = axes[1]
for t in TASKS:
    full = results[t]["psnr_full"]
    ax.plot(range(len(KS)), [results[t]["psnr_k"][k] - full for k in KS], "o-", color=colors[t], label=t, lw=2)
ax.axhline(0, ls="--", c="#333", lw=1)
ax.set_xticks(range(len(KS))); ax.set_xticklabels(KS)
ax.set_xlabel("spatial kernel size k"); ax.set_ylabel("PSNR vs full frequency filter (dB)")
ax.set_title("What does truncation actually cost?", fontsize=11, weight="bold")
ax.legend(); ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False)

ax = axes[2]
t = "derain"
r_ = 15
k_show = results[t]["h"][PATCH // 2 - r_:PATCH // 2 + r_ + 1, PATCH // 2 - r_:PATCH // 2 + r_ + 1]
im = ax.imshow(k_show, cmap="RdBu_r", vmin=-np.abs(k_show).max(), vmax=np.abs(k_show).max())
ax.set_title(f"optimal spatial kernel ({t}, centre 31x31)", fontsize=11, weight="bold")
ax.set_xticks([]); ax.set_yticks([])
plt.colorbar(im, ax=ax, fraction=0.046)

fig.suptitle("Frequency-domain restoration realised as a spatial kernel (convolution theorem)",
             fontsize=12, weight="bold")
fig.tight_layout()
out = "reports/student_v3/freq_to_spatial.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=150, facecolor="white")
print(f"wrote {out}")
