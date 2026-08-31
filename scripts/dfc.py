"""Degradation Frequency Curve (Huang et al., arXiv:2605.17506) on OUR data.

DFC, Eq.4-5:
    R~_b = sum(E_r^b) / sum(E_y^b)      residual energy / degraded energy, band b
    R_b  = R~_b / sum_j R~_j            normalised to a probability-like curve

with E computed under Gaussian band masks over the radial frequency axis,
residual r = y - x (degraded minus clean).

The claim worth testing: the RATIO form should divide image content out, which
is the confound that forced our own spectral work into same-scene synthesis
(we measured a 65.8% dataset-identity floor from clean images alone). If that
holds, DFC should separate degradations tightly even across different scenes.

CPU only -- the GPU is training.
"""
import sys, os, glob
sys.path.insert(0, ".")
import numpy as np, yaml
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = yaml.safe_load(open("configs/paths.local.yaml"))["data_root"]
OUT = "reports/dfc"; os.makedirs(OUT, exist_ok=True)
B, SIZE, SIG = 24, 256, 0.035          # bands, crop, Gaussian band width
rng = np.random.default_rng(0)

def load(p, n=SIZE):
    a = np.asarray(Image.open(p).convert("RGB"), np.float64) / 255.
    h, w = a.shape[:2]
    y, x = max(0,(h-n)//2), max(0,(w-n)//2)
    a = a[y:y+n, x:x+n]
    return a if a.shape[:2] == (n, n) else np.asarray(
        Image.fromarray((a*255).astype(np.uint8)).resize((n,n)), np.float64)/255.

# ---- Gaussian band masks over normalised radial frequency ------------------
fy = np.fft.fftshift(np.fft.fftfreq(SIZE))[:, None]
fx = np.fft.fftshift(np.fft.fftfreq(SIZE))[None, :]
rho = np.sqrt(fy**2 + fx**2); rho /= rho.max()
mu = np.linspace(0.0, 1.0, B)
MASKS = np.stack([np.exp(-(rho-m)**2 / (2*SIG**2)) for m in mu])   # (B,H,W)

def band_energy(img):
    """Total |F|^2 per Gaussian band, summed over colour channels."""
    E = np.zeros(B)
    for c in range(3):
        P = np.abs(np.fft.fftshift(np.fft.fft2(img[:, :, c])))**2
        E += (MASKS * P[None]).sum(axis=(1, 2))
    return E

def dfc(clean, degraded, eps=1e-12):
    r = degraded - clean                       # what the degradation ADDED
    Rt = band_energy(r) / (band_energy(degraded) + eps)
    return Rt / (Rt.sum() + eps), Rt           # normalised, and raw ratio

def radial_spec(img):
    """Our previous representation: radial log power of the DEGRADED image."""
    E = band_energy(img)
    return np.log1p(E) / (np.log1p(E).sum() + 1e-12)

# ---- real held-out pairs, one list per degradation -------------------------
cases = {}
r_in = sorted(glob.glob(f"{DATA}/test/derain/demo/input/*"))[:40]
cases["rain"] = [(load(p.replace("/input/","/target/")), load(p)) for p in r_in]
h_in = sorted(glob.glob(f"{DATA}/test/dehaze/demo/input/*"))[:40]
h_gt = sorted(glob.glob(f"{DATA}/test/dehaze/demo/target/*"))[:40]
cases["haze"] = [(load(b), load(a)) for a, b in zip(h_in, h_gt)]
n_gt = sorted(glob.glob(f"{DATA}/test/denoise/bsd68/*"))[:40]
cases["noise"] = [(lambda c: (c, np.clip(c + rng.standard_normal(c.shape)*(25/255.), 0, 1)))(load(p))
                  for p in n_gt]
print({k: len(v) for k, v in cases.items()}, flush=True)

curves, raws, specs, labels = {}, {}, [], []
for i, (name, pairs) in enumerate(cases.items()):
    cs, rs = [], []
    for clean, deg in pairs:
        d, raw = dfc(clean, deg)
        cs.append(d); rs.append(raw)
        specs.append(radial_spec(deg)); labels.append(i)
    curves[name] = np.array(cs); raws[name] = np.array(rs)
    print(f"  {name}: DFC computed for {len(cs)} pairs", flush=True)

specs = np.array(specs); labels = np.array(labels)
allc = np.concatenate([curves[k] for k in cases]); alll = np.concatenate(
    [np.full(len(curves[k]), i) for i, k in enumerate(cases)])

# ---- separability: DFC vs our previous raw-spectrum representation ---------
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def acc(X, y):
    cv = StratifiedKFold(5, shuffle=True, random_state=0); s = []
    for tr, te in cv.split(X, y):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        m.fit(X[tr], y[tr]); s.append(m.score(X[te], y[te]))
    return float(np.mean(s)), float(np.std(s))

a_dfc, s_dfc = acc(allc, alll)
a_spec, s_spec = acc(specs, labels)
print(f"\n  degradation ID from DFC          : {a_dfc*100:5.1f}% +/- {s_dfc*100:.1f}")
print(f"  degradation ID from raw spectrum : {a_spec*100:5.1f}% +/- {s_spec*100:.1f}")
print(f"  chance                           :  33.3%")

# content-invariance: how tight is each degradation's curve across scenes?
print("\n  within-degradation spread (mean std across bands, lower = more content-invariant)")
for k in cases:
    print(f"    {k:6s}  DFC {curves[k].std(0).mean():.4f}   raw-spectrum "
          f"{specs[labels==list(cases).index(k)].std(0).mean():.4f}")

np.savez(f"{OUT}/dfc_data.npz", mu=mu, **{f"dfc_{k}": curves[k] for k in cases},
         **{f"raw_{k}": raws[k] for k in cases})

# ---------------------------------------------------------------- figure
COL = {"rain": "#1F6F7A", "haze": "#A6423A", "noise": "#8A7E5C"}
fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.6), dpi=170)

for k in cases:
    m, s = curves[k].mean(0), curves[k].std(0)
    ax[0].plot(mu, m, color=COL[k], lw=2, label=k)
    ax[0].fill_between(mu, m-s, m+s, color=COL[k], alpha=0.18, lw=0)
ax[0].set_title("Degradation Frequency Curve\nR_b = band residual energy / band degraded energy",
                fontsize=10.5, fontweight="bold")
ax[0].set_xlabel("normalised radial frequency"); ax[0].set_ylabel("normalised DFC")
ax[0].legend(fontsize=9); ax[0].grid(alpha=0.25)

for k in cases:
    m = raws[k].mean(0)
    ax[1].semilogy(mu, np.maximum(m, 1e-8), color=COL[k], lw=2, label=k)
ax[1].set_title("Un-normalised ratio  E_residual / E_degraded\n(how much of each band the degradation owns)",
                fontsize=10.5, fontweight="bold")
ax[1].set_xlabel("normalised radial frequency"); ax[1].set_ylabel("energy ratio (log)")
ax[1].legend(fontsize=9); ax[1].grid(alpha=0.25, which="both")

bars = ax[2].bar(["DFC", "raw spectrum", "chance"], [a_dfc*100, a_spec*100, 33.3],
                 color=["#2E7D5B", "#4A5A6B", "#DDE2E8"], width=0.55)
ax[2].errorbar([0, 1], [a_dfc*100, a_spec*100], yerr=[s_dfc*100, s_spec*100],
               fmt="none", ecolor="#10141C", capsize=4, lw=1.2)
for b, v in zip(bars, [a_dfc*100, a_spec*100, 33.3]):
    ax[2].text(b.get_x()+b.get_width()/2, v+1.5, f"{v:.1f}%", ha="center",
               fontsize=10, fontweight="bold")
ax[2].set_ylim(0, 108); ax[2].set_ylabel("degradation ID accuracy (%)")
ax[2].set_title("Does the ratio form separate better?\n5-fold CV, real held-out pairs",
                fontsize=10.5, fontweight="bold")
ax[2].grid(axis="y", alpha=0.25)

fig.suptitle("Frequency representation of clean vs degraded images — DFC on our own data "
             "(real Rain100L / SOTS pairs, BSD68+sigma25)", fontsize=12, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(f"{OUT}/dfc_representation.png", facecolor="white", bbox_inches="tight")
print(f"\nwrote {OUT}/dfc_representation.png")
