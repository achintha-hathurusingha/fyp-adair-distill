"""Can an NPU-SAFE spatial feature match the FFT radial spectrum at
identifying degradation type?

Why this matters. spectral_samescene.py established that degradation type
is 93.6% recoverable blind from a 48-d radial log spectrum, with the
clean control at exactly chance -- so the signal is real and not dataset
identity. But that feature needs torch.fft, which this project has BANNED
after verifying it is absent from every mobile-NPU backend table. A
detector we cannot deploy is not a mechanism, it is a diagnostic.

The proxy. A Laplacian pyramid's per-band energies are a spatial-domain
approximation of a radially-binned spectrum: each band isolates a scale
(an annulus in frequency), and its energy is that annulus' power. Built
from blur + decimate + subtract + mean-of-squares, i.e. Conv / AvgPool /
Sub / Mul -- every one already verified SUPPORTED on qnn/tflite/tensorrt
(reports/student_v3/design.md, scripts/smoke_student_v3.py).

If the proxy matches the FFT feature, we get a deployable degradation
detector, which is precisely the inference-time selective mechanism
StudentV3 currently lacks (its only content-adaptive part is one SE gate
inside OrientedStreakGate).

NOTE ON THE LAPLACIAN MACHINERY. This is NOT a walk-back of deleting
LaplacianFrequencyGate from v3. That module was deleted because its
justification -- AdaIR's +1.58dB "frequency mining" claim -- was refuted.
Nothing here restores that claim. This is a different and separately
measured use of the same primitive: identification, not restoration.
"""
from __future__ import annotations

import sys, os, glob, random
sys.path.insert(0, ".")
sys.path.insert(0, "/home/minura/FYP/Workspace/Himeth/scripts/distill")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from degradations import add_noise, add_haze, add_rain  # noqa: E402

DATA = "/home/minura/fyp-adair-distill/data"
N_SCENES, PATCH, N_BINS, LEVELS, SEED = 150, 256, 48, 6, 0
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
    """FFT reference feature (NOT deployable -- needs torch.fft)."""
    g = img.astype(np.float64).mean(axis=2); g -= g.mean()
    P = np.abs(np.fft.fftshift(np.fft.fft2(g))) ** 2
    h, w = P.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((yy - h // 2) ** 2 + (xx - w // 2) ** 2); r /= r.max()
    idx = np.clip((r * n_bins).astype(int), 0, n_bins - 1)
    return np.log(np.array([P[idx == b].mean() if (idx == b).any() else 0.0
                            for b in range(n_bins)]) + 1e-12)


def laplacian_band_energies(img, levels=LEVELS):
    """NPU-SAFE proxy. Ops used: AveragePool (blur+decimate), Sub, Mul,
    ReduceMean -- all SUPPORTED on qnn/tflite/tensorrt. No FFT.

    Per band we record log mean-square (energy) AND log mean-abs, since
    heavy-tailed structured degradations (rain streaks) and Gaussian noise
    differ in how those two relate -- a cheap shape cue beyond raw energy.
    """
    x = torch.from_numpy(img.astype(np.float32).mean(axis=2))[None, None]
    feats, cur = [], x
    for _ in range(levels):
        blur = F.avg_pool2d(cur, 3, stride=1, padding=1)   # blur, keep size
        down = F.avg_pool2d(blur, 2)                        # decimate
        up = F.interpolate(down, size=cur.shape[-2:], mode="nearest")
        band = cur - up                                     # band-pass residual
        feats.append(float(torch.log(band.pow(2).mean() + 1e-8)))
        feats.append(float(torch.log(band.abs().mean() + 1e-8)))
        cur = down
        if min(cur.shape[-2:]) < 4:
            break
    while len(feats) < levels * 2:                          # pad if we stopped early
        feats.append(0.0)
    return np.array(feats[:levels * 2])


pool = sorted(glob.glob(f"{DATA}/Train/Denoise/*")); rng.shuffle(pool); pool = pool[:N_SCENES]
print(f"same-scene: {len(pool)} clean scenes x 3 synthetic degradations", flush=True)

rows = []
for p in pool:
    try:
        clean = center_crop(np.asarray(Image.open(p).convert("RGB")))
    except Exception:
        continue
    scene = os.path.basename(p)
    r = np.random.default_rng(abs(hash(scene)) % (2**31))
    for task, deg in {
        "denoise": add_noise(clean, r, sigma=float(r.choice([15, 25, 50]))),
        "dehaze": add_haze(clean, r),
        "derain": add_rain(clean, r),
    }.items():
        deg = np.asarray(deg)
        if deg.shape != clean.shape:
            continue
        rows.append({"task": task, "scene": scene,
                     "fft": radial_log_spectrum(deg),
                     "lap": laplacian_band_energies(deg)})

tasks = sorted({r["task"] for r in rows})
print(f"{len(rows)} samples: " + ", ".join(f"{t}={sum(x['task']==t for x in rows)}" for t in tasks))


def probe(key, label):
    X = np.stack([r[key] for r in rows])
    y = np.array([tasks.index(r["task"]) for r in rows])
    g = np.array([r["scene"] for r in rows])
    Xs = StandardScaler().fit_transform(X)
    accs = []
    for tr, te in GroupKFold(n_splits=5).split(Xs, y, g):
        clf = LogisticRegression(max_iter=4000).fit(Xs[tr], y[tr])
        accs.append((clf.predict(Xs[te]) == y[te]).mean())
    a = float(np.mean(accs))
    print(f"  {label:50s} dim={X.shape[1]:3d}  {a*100:6.1f}%")
    return a


print("\n3-way degradation ID, same scenes (chance 33.3%):")
a_fft = probe("fft", "FFT radial spectrum  (BANNED on NPU)")
a_lap = probe("lap", "Laplacian band energies  (NPU-SAFE)")

print("\nVerdict:")
print(f"  FFT reference      {a_fft*100:.1f}%")
print(f"  NPU-safe proxy     {a_lap*100:.1f}%   ({(a_lap-a_fft)*100:+.1f} pp vs FFT)")
if a_lap >= a_fft - 0.03:
    print("  => the spatial proxy MATCHES the FFT feature. A deployable degradation")
    print("     detector exists, using only ops already verified on all 3 backends.")
else:
    print("  => the spatial proxy is materially worse; FFT is doing something the")
    print("     Laplacian bands do not capture. Reported as measured.")
