"""Same-scene control: is degradation type spectrally identifiable when
dataset identity is removed entirely?

The corpus-based run (scripts/spectral_signature.py) found 88.1% blind
3-way accuracy -- but its CLEAN control already scored 65.8%, i.e. the
three corpora are themselves distinguishable, so most of that 88.1% could
be the probe reading "which dataset is this" rather than "which
degradation is this." Same confound teacher-experiments/test02 hit; test03
fixed it by synthesising every degradation from the SAME clean scenes.

This does that. One clean pool, three degradations applied to each scene,
scene-grouped CV. Now a scene contributes one sample to every class, so
dataset identity carries ZERO information and chance is exactly 33.3%.
Anything above chance is degradation signal.

Caveat recorded rather than hidden: rain here is SYNTHETIC (add_rain from
scripts/distill/degradations.py) rather than real Rain100L streaks, and
haze is synthesised by the atmospheric-scattering model rather than being
real RESIDE renders. That is the price of removing the confound -- the
identifiability question is answered cleanly, but on synthetic
degradations whose spectra may be cleaner than nature's.
"""
from __future__ import annotations

import sys, os, glob, random
sys.path.insert(0, ".")
sys.path.insert(0, "/home/minura/FYP/Workspace/Himeth/scripts/distill")

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from degradations import add_noise, add_haze, add_rain  # noqa: E402

DATA = "/home/minura/fyp-adair-distill/data"
N_SCENES = 150
PATCH = 256
N_BINS = 48
SEED = 0
rng = random.Random(SEED)


def center_crop(a, n=PATCH):
    h, w = a.shape[:2]
    if h < n or w < n:
        s = max(n / h, n / w)
        a = np.asarray(Image.fromarray(a.astype(np.uint8)).resize(
            (max(n, int(w * s + 1)), max(n, int(h * s + 1)))), dtype=np.uint8)
        h, w = a.shape[:2]
    y, x = (h - n) // 2, (w - n) // 2
    return a[y:y + n, x:x + n]


def radial_log_spectrum(img, n_bins=N_BINS):
    g = img.astype(np.float64).mean(axis=2)
    g = g - g.mean()
    P = np.abs(np.fft.fftshift(np.fft.fft2(g))) ** 2
    h, w = P.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((yy - h // 2) ** 2 + (xx - w // 2) ** 2)
    r /= r.max()
    idx = np.clip((r * n_bins).astype(int), 0, n_bins - 1)
    return np.log(np.array([P[idx == b].mean() if (idx == b).any() else 0.0
                            for b in range(n_bins)]) + 1e-12)


# One clean pool -- BSD400+WED (the denoise corpus), no degradation baked in.
pool = sorted(glob.glob(f"{DATA}/Train/Denoise/*"))
rng.shuffle(pool)
pool = pool[:N_SCENES]
print(f"same-scene control: {len(pool)} clean scenes x 3 synthetic degradations", flush=True)

rows = []
for i, p in enumerate(pool):
    try:
        clean = center_crop(np.asarray(Image.open(p).convert("RGB")))
    except Exception:
        continue
    scene = os.path.basename(p)
    r = np.random.default_rng(abs(hash(scene)) % (2**31))
    variants = {
        "denoise": add_noise(clean, r, sigma=float(r.choice([15, 25, 50]))),
        "dehaze": add_haze(clean, r),
        "derain": add_rain(clean, r),
    }
    for task, deg in variants.items():
        deg = np.asarray(deg)
        if deg.shape != clean.shape:
            continue
        rows.append({"task": task, "scene": scene,
                     "blind": radial_log_spectrum(deg),
                     "clean": radial_log_spectrum(clean)})

tasks = sorted({r["task"] for r in rows})
per = {t: sum(x["task"] == t for x in rows) for t in tasks}
print(f"{len(rows)} samples: {per}")


def probe(key, label):
    X = np.stack([r[key] for r in rows])
    y = np.array([tasks.index(r["task"]) for r in rows])
    groups = np.array([r["scene"] for r in rows])   # scene never spans folds
    Xs = StandardScaler().fit_transform(X)
    accs = []
    for tr, te in GroupKFold(n_splits=5).split(Xs, y, groups):
        clf = LogisticRegression(max_iter=4000).fit(Xs[tr], y[tr])
        accs.append((clf.predict(Xs[te]) == y[te]).mean())
    a = float(np.mean(accs))
    print(f"  {label:56s} {a*100:6.1f}%   (chance {100/len(tasks):.1f}%)")
    return a


print("\n3-way degradation classification, SAME SCENES (no dataset identity):")
a_blind = probe("blind", "BLIND (degraded only) -- deployable")
a_clean = probe("clean", "CONTROL: clean only -- must now be ~chance by construction")

print("\nInterpretation:")
print(f"  clean control  = {a_clean*100:.1f}%  (should be ~{100/len(tasks):.0f}% -- every scene "
      f"appears in all 3 classes, so clean carries no class information)")
print(f"  blind signal   = {a_blind*100:.1f}%")
print(f"  => degradation is {'RECOVERABLE' if a_blind > 0.75 else 'WEAKLY recoverable' if a_blind > 0.5 else 'NOT recoverable'} "
      f"blind, from a {N_BINS}-d radial spectrum alone")

fig, ax = plt.subplots(figsize=(7.5, 5))
colors = {"denoise": "#9467bd", "derain": "#2ca02c", "dehaze": "#ff7f0e"}
xs = np.linspace(0, 1, N_BINS)
for t in tasks:
    M = np.stack([r["blind"] for r in rows if r["task"] == t])
    m, sd = M.mean(0), M.std(0)
    ax.plot(xs, m, color=colors[t], label=t, lw=2)
    ax.fill_between(xs, m - sd, m + sd, color=colors[t], alpha=0.15)
Mc = np.stack([r["clean"] for r in rows])
ax.plot(xs, Mc.mean(0), color="#666", ls="--", lw=1.5, label="clean (same scenes)")
ax.set_xlabel("normalised radius (0 = DC, 1 = Nyquist)")
ax.set_ylabel("log amplitude")
ax.set_title(f"Same-scene radial spectra — dataset identity removed\n"
             f"blind 3-way accuracy {a_blind*100:.1f}% (chance 33.3%, clean control {a_clean*100:.1f}%)",
             fontsize=11, weight="bold")
ax.legend(fontsize=9)
ax.grid(alpha=0.25)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
out = "reports/student_v3/spectral_samescene.png"
fig.savefig(out, dpi=150, facecolor="white")
print(f"\nwrote {out}")
