"""Is degradation type recoverable from a cheap spectral signature?

Motivation. AdaIR's Fig.1 asserts that "different degradation types impact
image content on different frequency subbands," and builds its whole
architecture on that. We have already shown the ARCHITECTURE does not
exploit frequency (the mask is degenerate; repairing it is worth ~0.00dB).
That says nothing about whether the OBSERVATION is true -- two separate
claims. This script tests the observation on OUR data, and then asks the
question that actually matters for deployment.

Two settings, and the distinction is the point:

  RESIDUAL  profile of (degraded - clean). This is what cleanly
            characterises a degradation -- but it needs the clean image,
            which does not exist at inference. Useful as an upper bound
            and for understanding; useless as a deployed mechanism.

  BLIND     profile of the degraded image ALONE. This is the only setting
            a deployed all-in-one model actually has. If separability
            survives here, it is a usable degradation detector; if it
            collapses, then "degradations differ spectrally" is true but
            operationally empty for a blind model -- which would be a
            substantive finding about AdaIR's premise, not just its code.

Method. Radially-averaged log power spectrum: 2D FFT -> fftshift -> |F|^2
-> average over annuli of increasing radius -> log. That is the same
construction AdaIR's Fig.1 right panel plots (log amplitude vs normalised
radius). Then a linear probe (logistic regression, scene-grouped CV so a
scene never spans train/test) measures how separable the three classes are.

Scene-grouped splitting matters: denoise/derain/dehaze here come from
different source corpora, so an ungrouped probe could separate DATASETS
rather than DEGRADATIONS -- the exact confound teacher-experiments/test02
hit and test03 was built to remove. We cannot fully remove it (the corpora
genuinely differ), so the CONTROL below is the load-bearing part:

  CONTROL   the same probe on CLEAN images from the same three corpora.
            If clean images are already separable, the probe is reading
            dataset identity, not degradation, and any "degradation
            separability" number is inflated by exactly that much.

CPU-only, small sample: B0V3 is training on the GPU and must not be
disturbed.
"""
from __future__ import annotations

import sys
sys.path.insert(0, ".")

import glob
import os
import random

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = "/home/minura/fyp-adair-distill/data"
N_PER_TASK = 120
PATCH = 256          # fixed size so radial bins are comparable across images
N_BINS = 48
SEED = 0

rng = random.Random(SEED)
np.random.seed(SEED)


def load_rgb(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float64)


def center_crop(a, n=PATCH):
    h, w = a.shape[:2]
    if h < n or w < n:
        s = max(n / h, n / w)
        a = np.asarray(Image.fromarray(a.astype(np.uint8)).resize(
            (max(n, int(w * s + 1)), max(n, int(h * s + 1)))), dtype=np.float64)
        h, w = a.shape[:2]
    y, x = (h - n) // 2, (w - n) // 2
    return a[y:y + n, x:x + n]


def radial_log_spectrum(img, n_bins=N_BINS):
    """Radially-averaged log power spectrum of a grayscale image.
    Returns (n_bins,) -- log amplitude vs normalised radius, i.e. exactly
    the curve AdaIR's Fig.1 right panel plots."""
    g = img.mean(axis=2)
    g = g - g.mean()                       # drop DC so it doesn't dominate
    F = np.fft.fftshift(np.fft.fft2(g))
    P = np.abs(F) ** 2
    h, w = P.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r = r / r.max()                        # normalised radius in [0,1]
    idx = np.clip((r * n_bins).astype(int), 0, n_bins - 1)
    out = np.zeros(n_bins)
    for b in range(n_bins):
        m = idx == b
        out[b] = P[m].mean() if m.any() else 0.0
    return np.log(out + 1e-12)


# ---------------------------------------------------------------- data
def sample_pairs():
    """(task, scene_id, degraded, clean) triples from the REAL corpora."""
    items = []

    # derain: Train/Derain/{input,target}/rain-XXX.png
    rain = sorted(glob.glob(f"{DATA}/Train/Derain/input/*.png"))
    rng.shuffle(rain)
    for p in rain[:N_PER_TASK]:
        t = p.replace("/input/", "/target/")
        if os.path.exists(t):
            items.append(("derain", os.path.basename(p), p, t))

    # dehaze: synthetic/{scene}_{beta}_{A}.jpg -> clear/{scene}.jpg
    haze = []
    for part in (1, 2, 3, 4):
        haze += glob.glob(f"{DATA}/Train/Dehaze/synthetic/part{part}/*.jpg")
    rng.shuffle(haze)
    seen = set()
    for p in haze:
        scene = os.path.basename(p).split("_")[0]
        if scene in seen:
            continue
        c = f"{DATA}/Train/Dehaze/clear/{scene}.jpg"
        if os.path.exists(c):
            seen.add(scene)
            items.append(("dehaze", scene, p, c))
        if len(seen) >= N_PER_TASK:
            break

    # denoise: clean corpus + synthetic Gaussian noise, same sigmas the
    # student trains on. Degraded is generated here, per-image seeded.
    clean = sorted(glob.glob(f"{DATA}/Train/Denoise/*"))
    rng.shuffle(clean)
    for p in clean[:N_PER_TASK]:
        items.append(("denoise", os.path.basename(p), None, p))
    return items


print("loading and computing spectra (CPU only -- B0V3 has the GPU)...", flush=True)
rows = []
for task, scene, dpath, cpath in sample_pairs():
    try:
        clean = center_crop(load_rgb(cpath))
        if task == "denoise":
            s = np.random.RandomState(abs(hash(scene)) % (2**31))
            sigma = s.choice([15, 25, 50])
            degraded = np.clip(clean + s.standard_normal(clean.shape) * sigma, 0, 255)
        else:
            degraded = center_crop(load_rgb(dpath))
    except Exception as e:
        continue
    if degraded.shape != clean.shape:
        continue
    rows.append({
        "task": task, "scene": scene,
        "blind": radial_log_spectrum(degraded),
        "residual": radial_log_spectrum(degraded - clean + 128.0),
        "clean": radial_log_spectrum(clean),
    })

tasks = sorted({r["task"] for r in rows})
print(f"{len(rows)} samples: " + ", ".join(f"{t}={sum(r['task']==t for r in rows)}" for t in tasks))


# ------------------------------------------------------- separability
def probe(key, label):
    X = np.stack([r[key] for r in rows])
    y = np.array([tasks.index(r["task"]) for r in rows])
    groups = np.array([f"{r['task']}/{r['scene']}" for r in rows])
    Xs = StandardScaler().fit_transform(X)
    accs = []
    for tr, te in GroupKFold(n_splits=5).split(Xs, y, groups):
        clf = LogisticRegression(max_iter=4000)
        clf.fit(Xs[tr], y[tr])
        accs.append((clf.predict(Xs[te]) == y[te]).mean())
    acc = float(np.mean(accs))
    print(f"  {label:52s} {acc*100:6.1f}%   (chance {100/len(tasks):.1f}%)")
    return acc


print("\n3-way degradation classification from the radial spectrum alone:")
a_res = probe("residual", "RESIDUAL (degraded-clean) -- needs clean, not deployable")
a_bli = probe("blind", "BLIND (degraded only) -- the deployable setting")
a_cln = probe("clean", "CONTROL: CLEAN images only -- pure dataset identity")

print("\nInterpretation:")
print(f"  dataset-identity floor      = {a_cln*100:.1f}%  (what the probe gets for free)")
print(f"  blind degradation signal    = {a_bli*100:.1f}%")
print(f"  headroom over that floor    = {(a_bli-a_cln)*100:+.1f} pp")
if a_cln > 0.8:
    print("  WARNING: clean images are already highly separable -- the corpora differ,")
    print("  so the blind number is inflated by dataset identity and CANNOT be read")
    print("  as pure degradation separability.")

# ------------------------------------------------------------- figure
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
colors = {"denoise": "#9467bd", "derain": "#2ca02c", "dehaze": "#ff7f0e"}
xs = np.linspace(0, 1, N_BINS)
for ax, key, title in zip(
        axes, ["blind", "residual", "clean"],
        [f"BLIND: degraded image only  ({a_bli*100:.0f}%)",
         f"RESIDUAL: degraded - clean  ({a_res*100:.0f}%)",
         f"CONTROL: clean only  ({a_cln*100:.0f}%)"]):
    for t in tasks:
        M = np.stack([r[key] for r in rows if r["task"] == t])
        m, sd = M.mean(0), M.std(0)
        ax.plot(xs, m, color=colors.get(t, None), label=t, lw=2)
        ax.fill_between(xs, m - sd, m + sd, color=colors.get(t, None), alpha=0.15)
    ax.set_title(title, fontsize=11, weight="bold")
    ax.set_xlabel("normalised radius (0 = DC, 1 = Nyquist)")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("log amplitude")
axes[0].legend(fontsize=9)
fig.suptitle("Radial log power spectrum by degradation -- AdaIR Fig.1 reproduced on OUR data, "
             "with the deployability and dataset-identity controls it omits",
             fontsize=12, weight="bold")
fig.tight_layout()
out = "reports/student_v3/spectral_signature.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=150, facecolor="white")
print(f"\nwrote {out}")
