"""Can a LEARNED, NPU-safe spatial module emulate the FFT radial spectrum?

The idea being tested. AdaIR's frequency path is dead, so there is nothing
spectral to distil from it. But the spectral SIGNAL itself is real
(scripts/spectral_samescene.py: 93.6% blind degradation ID, clean control
at exactly chance). So instead of distilling from a broken network, use a
FIXED FFT OPERATOR as the target -- it has no weights, no checkpoint and
no training, so it cannot be dead the way AdaIR's module was -- and train a
spatial converter to emulate it. At deployment the FFT is discarded and
only the converter ships. That is literally "replace spectral ops the NPU
cannot execute", backed by a real signal.

Feasibility is already bounded: a HAND-DESIGNED, untrained Laplacian
band-energy feature reached 88.0% vs the FFT's 92.4%
(scripts/spectral_spatial_proxy.py). A learned converter should beat a
hand-designed one. This measures whether it does.

Four conditions, all on the SAME same-scene data (every scene appears in
all three classes, so dataset identity carries zero information and chance
is exactly 33.3%):

  FFT           classify from the true 48-d radial spectrum      = ceiling
  HAND          classify from untrained Laplacian energies       = prior work here
  CONVERTER     classify from the converter's PREDICTED spectrum = the idea
  DIRECT        same backbone trained straight on the 3-way label
                -- the control that asks whether the spectral detour buys
                anything at all, since we DO have labels at training time.
                If DIRECT >= CONVERTER, predicting the spectrum is not
                earning its keep for this particular use.

NPU safety: the converter uses Conv2d / AvgPool2d / ReLU / Sub /
GlobalAvgPool / Linear only. No FFT. No F.interpolate -- that lowers to
ONNX `Resize`, which appears in NONE of the three curated backend tables
(same reason LaplacianFrequencyGate used PixelShuffle instead). The pyramid
is therefore built WITHOUT upsampling: band = cur - blur(cur) at the
current scale, then decimate for the next level.
"""
from __future__ import annotations

import sys, os, glob, random
sys.path.insert(0, ".")
sys.path.insert(0, "/home/minura/FYP/Workspace/Himeth/scripts/distill")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from degradations import add_noise, add_haze, add_rain  # noqa: E402

DATA = "/home/minura/fyp-adair-distill/data"
N_SCENES, PATCH, N_BINS, LEVELS = 400, 128, 48, 5
EPOCHS, BS, SEED = 30, 32, 0

torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
dev = "cuda" if torch.cuda.is_available() else "cpu"
TASKS = ["dehaze", "denoise", "derain"]


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
    g = img.astype(np.float64).mean(axis=2); g -= g.mean()
    P = np.abs(np.fft.fftshift(np.fft.fft2(g))) ** 2
    h, w = P.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((yy - h // 2) ** 2 + (xx - w // 2) ** 2); r /= r.max()
    idx = np.clip((r * n_bins).astype(int), 0, n_bins - 1)
    return np.log(np.array([P[idx == b].mean() if (idx == b).any() else 0.0
                            for b in range(n_bins)]) + 1e-12)


def hand_laplacian(img, levels=LEVELS):
    x = torch.from_numpy(img.astype(np.float32).mean(axis=2))[None, None]
    f, cur = [], x
    for _ in range(levels):
        blur = F.avg_pool2d(cur, 3, stride=1, padding=1)
        band = cur - blur
        f += [float(torch.log(band.pow(2).mean() + 1e-8)),
              float(torch.log(band.abs().mean() + 1e-8))]
        cur = F.avg_pool2d(blur, 2)
        if min(cur.shape[-2:]) < 4:
            break
    while len(f) < levels * 2:
        f.append(0.0)
    return np.array(f[:levels * 2])


# ----------------------------------------------------------------- model
class SpatialConverter(nn.Module):
    """NPU-safe learned pyramid. No FFT, no Resize."""

    def __init__(self, levels=LEVELS, width=24, out_dim=N_BINS):
        super().__init__()
        self.levels = levels
        self.blurs = nn.ModuleList(
            nn.Conv2d(1, 1, 3, padding=1, bias=False) for _ in range(levels))
        for b in self.blurs:                       # init as a real blur
            with torch.no_grad():
                b.weight.fill_(1.0 / 9.0)
        self.encs = nn.ModuleList(
            nn.Sequential(nn.Conv2d(1, width, 3, padding=1), nn.ReLU(inplace=True),
                          nn.Conv2d(width, width, 1), nn.ReLU(inplace=True))
            for _ in range(levels))
        self.head = nn.Sequential(
            nn.Linear(levels * width * 2, 256), nn.ReLU(inplace=True),
            nn.Linear(256, out_dim))

    def features(self, x):
        feats, cur = [], x
        for blur, enc in zip(self.blurs, self.encs):
            b = blur(cur)
            band = cur - b                          # high-pass at this scale
            e = enc(band)
            # two pooled statistics per band: mean and mean-of-squares
            feats.append(e.mean(dim=(2, 3)))
            feats.append(e.pow(2).mean(dim=(2, 3)).clamp_min(1e-8).log())
            cur = F.avg_pool2d(b, 2)                # decimate, no upsampling
        return torch.cat(feats, dim=1)

    def forward(self, x):
        return self.head(self.features(x))


class DirectClassifier(nn.Module):
    """Same backbone, trained straight on the label -- the control."""

    def __init__(self, levels=LEVELS, width=24, n_cls=3):
        super().__init__()
        self.body = SpatialConverter(levels, width, out_dim=n_cls)

    def forward(self, x):
        return self.body(x)


# ------------------------------------------------------------------ data
pool = sorted(glob.glob(f"{DATA}/Train/Denoise/*"))
random.shuffle(pool); pool = pool[:N_SCENES]
print(f"building {len(pool)} scenes x 3 degradations (same-scene)...", flush=True)

imgs, specs, hands, labels, scenes = [], [], [], [], []
for p in pool:
    try:
        clean = center_crop(np.asarray(Image.open(p).convert("RGB")))
    except Exception:
        continue
    sc = os.path.basename(p)
    r = np.random.default_rng(abs(hash(sc)) % (2**31))
    for t, deg in {
        "denoise": add_noise(clean, r, sigma=float(r.choice([15, 25, 50]))),
        "dehaze": add_haze(clean, r),
        "derain": add_rain(clean, r),
    }.items():
        deg = np.asarray(deg)
        if deg.shape != clean.shape:
            continue
        imgs.append(deg.astype(np.float32).mean(axis=2) / 255.0)
        specs.append(radial_log_spectrum(deg))
        hands.append(hand_laplacian(deg))
        labels.append(TASKS.index(t))
        scenes.append(sc)

X = torch.from_numpy(np.stack(imgs))[:, None]
Y = torch.from_numpy(np.stack(specs)).float()
H = np.stack(hands)
L = np.array(labels)
G = np.array(scenes)
# normalise the spectrum target (per-bin) so MSE is not dominated by low bins
Ym, Ys = Y.mean(0, keepdim=True), Y.std(0, keepdim=True) + 1e-6
Yn = (Y - Ym) / Ys
print(f"{len(X)} samples, image {tuple(X.shape[1:])}, spectrum dim {Y.shape[1]}")

uniq = np.array(sorted(set(G)))
rs = np.random.RandomState(SEED); rs.shuffle(uniq)
cut = int(0.75 * len(uniq))
tr_sc, te_sc = set(uniq[:cut]), set(uniq[cut:])
tr = np.array([i for i, s in enumerate(G) if s in tr_sc])
te = np.array([i for i, s in enumerate(G) if s in te_sc])
print(f"scene-disjoint split: {len(tr)} train / {len(te)} test "
      f"({len(tr_sc)}/{len(te_sc)} scenes)")


def train(model, target, loss_fn, tag):
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    tgt = target.to(dev)
    xb_all = X.to(dev)
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(tr), device=dev)
        idx_tr = torch.as_tensor(tr, device=dev)[perm]
        tot = 0.0
        for i in range(0, len(idx_tr), BS):
            b = idx_tr[i:i + BS]
            opt.zero_grad()
            loss = loss_fn(model(xb_all[b]), tgt[b])
            loss.backward(); opt.step()
            tot += float(loss) * len(b)
        if ep % 10 == 9:
            print(f"    {tag} ep{ep+1:3d} loss {tot/len(idx_tr):.4f}", flush=True)
    model.eval()
    return model


def probe_from(feats_tr, feats_te):
    sc = StandardScaler().fit(feats_tr)
    clf = LogisticRegression(max_iter=4000).fit(sc.transform(feats_tr), L[tr])
    return float((clf.predict(sc.transform(feats_te)) == L[te]).mean())


print("\n--- training the spatial converter (target = FFT spectrum) ---", flush=True)
conv = train(SpatialConverter(), Yn, nn.MSELoss(), "conv")
with torch.no_grad():
    pred = conv(X.to(dev)).cpu().numpy()
# how well does it reproduce the spectrum?
true_n = Yn.numpy()
mse = float(np.mean((pred[te] - true_n[te]) ** 2))
corr = float(np.mean([np.corrcoef(pred[i], true_n[i])[0, 1] for i in te]))
print(f"  spectrum reconstruction on held-out scenes: MSE {mse:.4f}, mean per-sample r {corr:.3f}")

print("\n--- training the DIRECT label classifier (control) ---", flush=True)
direct = train(DirectClassifier(), torch.from_numpy(L).long(), nn.CrossEntropyLoss(), "direct")
with torch.no_grad():
    acc_direct = float((direct(X[te].to(dev)).argmax(1).cpu().numpy() == L[te]).mean())

acc_fft = probe_from(Y.numpy()[tr], Y.numpy()[te])
acc_hand = probe_from(H[tr], H[te])
acc_conv = probe_from(pred[tr], pred[te])

print("\n" + "=" * 62)
print("3-way degradation ID, scene-disjoint test set (chance 33.3%)")
print("=" * 62)
print(f"  FFT true spectrum      (NOT deployable)      {acc_fft*100:6.1f}%")
print(f"  HAND Laplacian         (NPU-safe, untrained) {acc_hand*100:6.1f}%")
print(f"  CONVERTER pred spectrum(NPU-safe, learned)   {acc_conv*100:6.1f}%")
print(f"  DIRECT label head      (NPU-safe, learned)   {acc_direct*100:6.1f}%")
print("\nReading:")
print(f"  learned vs hand-designed : {(acc_conv-acc_hand)*100:+.1f} pp")
print(f"  learned vs FFT ceiling   : {(acc_conv-acc_fft)*100:+.1f} pp")
print(f"  spectral detour vs direct: {(acc_conv-acc_direct)*100:+.1f} pp "
      f"({'detour justified' if acc_conv > acc_direct else 'DIRECT is as good or better -- the spectral target is not earning its keep for classification'})")
