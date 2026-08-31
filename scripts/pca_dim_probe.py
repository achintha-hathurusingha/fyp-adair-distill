"""S0.3 -- How much does PCA-16 throw away?

Plan: probe degradation-ID accuracy from decoder features at 16/32/64/128/full
dims, leave-scene-out.
  Kill: 16 dims already saturates -> "wider feature set" is unjustified; say so
  and keep PCA-16.

WHAT IS BEING PROBED
--------------------
The teacher's PCA-16 `e_D` code is a TEST19 artifact and lives outside this
repo (`teacher-experiments/` is empty here), so this measures the thing S2.1
would actually build on: the STUDENT's own decoder features, from the current
best arm (B0V3-KD-FEAT @81k). The decision S0.3 gates is whether a *wider*
post-decoder representation is justified, and that is a question about the
dimensionality of the student's features, not about reproducing the teacher's
specific basis.

METHOD
------
SAME-SCENE design, which is non-negotiable here. Every scene contributes all
three degradations, so degradation is the only variable. The project has
already measured what happens without it: a corpus-based version was inflated
by a **65.8% dataset-identity floor**, while the same-scene version gives 93.6%
with the clean control exactly at chance. Corpus-based numbers would answer a
different, easier question (which dataset is this?).

  * features: GAP over each hooked tensor (middle_blks + the 4 decoder stages),
    matching what `DegradationHead` actually consumes (AdaptiveAvgPool2d -> Linear)
  * split: 5-fold LEAVE-SCENE-OUT. Train and test scenes are disjoint, so a
    probe cannot win by memorising scenes.
  * PCA is fitted on the TRAIN fold only, then applied to test. Fitting it on
    everything would leak, and at 128+ dims the leak would look exactly like
    "more dims help".
  * probe: multinomial logistic regression, 3-way. Chance = 33.3%.

CONTROLS
--------
  * CLEAN control: identical clean images labelled with the degradation that
    *would* have been applied. Must sit at chance -- if it does not, the probe
    is reading scene or fold structure, not degradation.
  * SHUFFLED-LABEL control: same features, permuted labels. Must sit at chance.

SANITY GATE
-----------
Before any of the above, the loaded model must actually restore (PSNR on a
degraded batch well above the degraded input). If preprocessing were wrong --
wrong scale, wrong weight set -- the features would be garbage and the probe
would still happily report a plausible-looking accuracy. The gate makes that
failure loud instead of silent. It also picks between the `model` and `ema`
weight sets by measured restoration quality rather than assumption.
"""
from __future__ import annotations

import glob
import json
import os
import random
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "/home/minura/FYP/Workspace/Himeth/scripts/distill")

import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from degradations import add_haze, add_noise, add_rain  # noqa: E402

from src.train.train import build_model  # noqa: E402

CKPT = "/tmp/b0v3kd_81k.pth"
DATA = "/home/minura/fyp-adair-distill/data"
N_SCENES, PATCH, SEED = 400, 128, 0
DIMS = [2, 4, 8, 16, 32, 64, 128, None]     # None = full (no PCA)
TASKS = ["denoise", "derain", "dehaze"]
N_FOLDS = 5

torch.set_grad_enabled(False)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


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


def to_tensor(img_uint8):
    return torch.from_numpy(
        np.ascontiguousarray(img_uint8.transpose(2, 0, 1))).float().unsqueeze(0) / 255.0


def psnr_t(a, b):
    m = float(((a - b) ** 2).mean())
    return 99.0 if m <= 1e-12 else 10 * np.log10(1.0 / m)


# --------------------------------------------------------------- model + hooks
def load_model():
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = build_model(cfg)
    return model, ck


def restoration_psnr(model, files, n=12):
    """Mean PSNR gain over degraded input -- the sanity gate."""
    gains = []
    for p in files[:n]:
        clean = center_crop(np.asarray(Image.open(p).convert("RGB")))
        r = np.random.default_rng(abs(hash("gate" + os.path.basename(p))) % (2 ** 31))
        deg = degrade(clean, "denoise", r)
        ct, dt = to_tensor(clean), to_tensor(deg)
        out = model(dt).clamp(0, 1)
        gains.append(psnr_t(out, ct) - psnr_t(dt, ct))
    return float(np.mean(gains))


class FeatureTap:
    """GAP over middle_blks and every decoder stage."""

    def __init__(self, model):
        self.buf = {}
        self.handles = []
        self.names = []

        def mk(name):
            def hook(_m, _i, o):
                t = o[0] if isinstance(o, (tuple, list)) else o
                self.buf[name] = t.mean(dim=(2, 3)).squeeze(0).cpu().numpy()
            return hook

        self.handles.append(model.middle_blks.register_forward_hook(mk("middle")))
        self.names.append("middle")
        for i, dec in enumerate(model.decoders):
            self.handles.append(dec.register_forward_hook(mk(f"dec{i}")))
            self.names.append(f"dec{i}")

    def close(self):
        for h in self.handles:
            h.remove()


# --------------------------------------------------------------------- probing
def probe(X, y, scenes, dims, seed=0):
    """5-fold leave-scene-out accuracy at a given PCA dim.

    PCA and the scaler are fitted on the TRAIN fold only -- fitting on all data
    would leak, and the leak grows with dimension, which is exactly the axis
    under test."""
    uniq = np.unique(scenes)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    folds = np.array_split(perm, N_FOLDS)
    accs = []
    for f in folds:
        test_scenes = set(uniq[f])
        te = np.array([s in test_scenes for s in scenes])
        tr = ~te
        Xtr, Xte = X[tr], X[te]
        sc = StandardScaler().fit(Xtr)
        Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        if dims is not None and dims < Xtr.shape[1]:
            p = PCA(n_components=dims, random_state=seed).fit(Xtr)
            Xtr, Xte = p.transform(Xtr), p.transform(Xte)
        # sklearn >=1.7 removed `multi_class`; lbfgs is multinomial by default
        clf = LogisticRegression(max_iter=3000)
        clf.fit(Xtr, y[tr])
        accs.append(float((clf.predict(Xte) == y[te]).mean()))
    return float(np.mean(accs)), float(np.std(accs))


def main():
    print("=" * 78)
    print("S0.3 -- how much does PCA-16 throw away?")
    print("=" * 78)

    pool = sorted(glob.glob(f"{DATA}/Train/Denoise/*"))
    random.shuffle(pool)
    files = pool[:N_SCENES]

    print("\n[0] load model + SANITY GATE")
    model, ck = load_model()
    best = None
    for key in ("ema", "model"):
        if key not in ck:
            continue
        sd = ck[key]
        sd = sd.get("shadow", sd) if isinstance(sd, dict) and "shadow" in sd else sd
        try:
            missing, unexpected = model.load_state_dict(sd, strict=False)
        except Exception as e:  # noqa: BLE001
            print(f"    {key}: could not load ({type(e).__name__})")
            continue
        model.eval()
        g = restoration_psnr(model, files)
        print(f"    weights '{key}': missing={len(missing)} unexpected={len(unexpected)}"
              f"  denoise PSNR gain = {g:+.2f} dB")
        if best is None or g > best[1]:
            best = (key, g, {k: v.clone() for k, v in model.state_dict().items()})
    assert best is not None, "no loadable weight set"
    print(f"    -> using '{best[0]}' (gain {best[1]:+.2f} dB)")
    assert best[1] > 3.0, (
        f"SANITY GATE FAILED: model restores only {best[1]:+.2f} dB. "
        "Preprocessing or weights are wrong; features would be meaningless.")
    model.load_state_dict(best[2])
    model.eval()
    print(f"    params: {sum(p.numel() for p in model.parameters()):,}")

    CACHE = "/tmp/s0_3_feats.npz"
    if os.path.exists(CACHE):
        print(f"\n[1] loading cached features from {CACHE}")
        z = np.load(CACHE)
        names = [str(n) for n in z["names"]]
        feats = {n: z[f"f_{n}"] for n in names}
        feats_clean = {n: z[f"c_{n}"] for n in names}
        labels, labels_clean = z["labels"], z["labels_clean"]
        scenes, scenes_clean = z["scenes"], z["scenes_clean"]
        return finish(names, feats, feats_clean, labels, labels_clean,
                      scenes, scenes_clean, best)

    print(f"\n[1] extract same-scene features ({len(files)} scenes x "
          f"{len(TASKS)} degradations + clean control)")
    tap = FeatureTap(model)
    feats = {n: [] for n in tap.names}
    feats_clean = {n: [] for n in tap.names}
    labels, labels_clean, scenes, scenes_clean = [], [], [], []

    for si, p in enumerate(files):
        try:
            clean = center_crop(np.asarray(Image.open(p).convert("RGB")))
        except Exception:
            continue
        for ti, task in enumerate(TASKS):
            r = np.random.default_rng(
                abs(hash(task + os.path.basename(p))) % (2 ** 31))
            deg = degrade(clean, task, r)
            if deg.shape != clean.shape:
                continue
            model(to_tensor(deg))
            for n in tap.names:
                feats[n].append(tap.buf[n].copy())
            labels.append(ti)
            scenes.append(si)
            # CLEAN control: identical clean image, labelled as if degraded
            model(to_tensor(clean))
            for n in tap.names:
                feats_clean[n].append(tap.buf[n].copy())
            labels_clean.append(ti)
            scenes_clean.append(si)
        if (si + 1) % 100 == 0:
            print(f"    {si + 1}/{len(files)} scenes", flush=True)
    tap.close()

    for n in tap.names:
        feats[n] = np.asarray(feats[n])
        feats_clean[n] = np.asarray(feats_clean[n])
    labels = np.asarray(labels)
    labels_clean = np.asarray(labels_clean)
    scenes = np.asarray(scenes)
    scenes_clean = np.asarray(scenes_clean)

    np.savez(CACHE, names=np.array(tap.names),
             labels=labels, labels_clean=labels_clean,
             scenes=scenes, scenes_clean=scenes_clean,
             **{f"f_{n}": feats[n] for n in tap.names},
             **{f"c_{n}": feats_clean[n] for n in tap.names})
    print(f"    cached features -> {CACHE}")

    return finish(tap.names, feats, feats_clean, labels, labels_clean,
                  scenes, scenes_clean, best)


def finish(names, feats, feats_clean, labels, labels_clean,
           scenes, scenes_clean, best):
    concat = np.concatenate([feats[n] for n in names], axis=1)
    concat_clean = np.concatenate([feats_clean[n] for n in names], axis=1)
    print(f"    samples {len(labels)}, per-stage dims "
          f"{ {n: feats[n].shape[1] for n in names} }, concat {concat.shape[1]}")

    sources = {n: (feats[n], feats_clean[n]) for n in names}
    sources["concat"] = (concat, concat_clean)

    print("\n[2] degradation-ID accuracy vs PCA dimension "
          "(5-fold leave-scene-out, chance = 33.3%)")
    results = {}
    for name, (X, _) in sources.items():
        row = {}
        print(f"\n    --- {name} (dim {X.shape[1]}) ---")
        print(f"      {'dims':>6}{'acc':>9}{'sd':>8}{'vs 16d':>9}")
        acc16 = None
        for d in DIMS:
            if d is not None and d > X.shape[1]:
                continue
            m, s = probe(X, labels, scenes, d)
            row[str(d) if d else "full"] = {"acc": m, "sd": s}
            if d == 16:
                acc16 = m
            delta = f"{(m - acc16) * 100:+8.2f}" if acc16 is not None else "       -"
            print(f"      {str(d) if d else 'full':>6}{m * 100:>8.2f}%{s * 100:>7.2f}{delta}")
        results[name] = row

    print("\n[3] CONTROLS (all must sit at chance = 33.3%)")
    Xc, Xcc = sources["concat"]
    m, s = probe(Xcc, labels_clean, scenes_clean, 16)
    print(f"      clean control, 16 dims      {m * 100:6.2f}% (sd {s * 100:.2f})")
    ctrl_clean = m
    m2, s2 = probe(Xcc, labels_clean, scenes_clean, None)
    print(f"      clean control, full dims    {m2 * 100:6.2f}% (sd {s2 * 100:.2f})")
    rng = np.random.default_rng(0)
    y_shuf = rng.permutation(labels)
    m3, s3 = probe(Xc, y_shuf, scenes, 16)
    print(f"      shuffled labels, 16 dims    {m3 * 100:6.2f}% (sd {s3 * 100:.2f})")

    print("\n" + "=" * 78)
    print("VERDICT (kill: 16 dims already saturates -> keep PCA-16)")
    print("=" * 78)
    for name in results:
        r = results[name]
        if "16" not in r:
            continue
        a16 = r["16"]["acc"]
        best_k = max(r, key=lambda k: r[k]["acc"])
        gain = (r[best_k]["acc"] - a16) * 100
        print(f"  {name:<8} 16d = {a16 * 100:6.2f}%   best = {r[best_k]['acc'] * 100:6.2f}% "
              f"at {best_k:>4}d   gain over 16d = {gain:+.2f} pp")
    best_over = max(
        ((r[k]["acc"] - r["16"]["acc"]) * 100)
        for r in results.values() if "16" in r for k in r)
    print(f"\n  Largest gain of ANY dim over 16 dims, any source: {best_over:+.2f} pp")
    print(f"  Controls: clean {ctrl_clean * 100:.2f}%, shuffled {m3 * 100:.2f}% "
          f"(chance 33.33%)")
    if best_over < 2.0:
        print("\n  KILL: 16 dims saturates (<+2pp available anywhere).")
        print("        A wider post-decoder feature set is NOT justified by")
        print("        degradation-ID capacity. Keep PCA-16.")
    else:
        print(f"\n  PASS: {best_over:+.2f} pp is available beyond 16 dims.")

    os.makedirs("reports/reparam_gate", exist_ok=True)
    out = {"results": results,
           "controls": {"clean_16": ctrl_clean, "clean_full": m2, "shuffled_16": m3},
           "n_scenes": int(len(np.unique(scenes))), "n_samples": int(len(labels)),
           "weights": best[0], "sanity_gain_db": best[1]}
    with open("reports/reparam_gate/s0_3_pca_dims.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote reports/reparam_gate/s0_3_pca_dims.json")


if __name__ == "__main__":
    main()
