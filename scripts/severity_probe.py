"""S2.1 headroom test: is degradation SEVERITY decodable, and does it saturate?

S0.3 killed the "wider feature set" premise on degradation TYPE and, in doing so,
broke S2.1's kill criterion: PCA-16 on decoder features already scores 99.67% on
3-way type ID, so the +2pp bar S2.1 asks for is unreachable -- only +0.33 pp
exists. Type is a saturated, trivially easy probe.

Severity is the axis with plausible headroom, and it is the one a conditioning
signal would actually have to carry: knowing "this is rain" is worth little if
the model cannot tell light drizzle from a downpour. So:

    Does the student's decoder encode HOW MUCH degradation, not just which kind,
    and does that saturate at 16 dims the way type did?

  * saturates at 16 too -> the whole "richer representation" branch is dead, and
    S2.1 can be killed for ~10 min of CPU instead of ~3h plus GPU follow-on.
  * needs more dims -> S2.1 is worth running, redefined on severity.

Severity is drawn CONTINUOUSLY (not from the 3 training sigmas) so the probe
measures a graded quantity rather than re-running type ID under another name:
    denoise  sigma   in [5, 55]      (training used the discrete {15,25,50})
    dehaze   beta    in [0.6, 2.6]   (generator default draws U(1.0, 2.2))
    derain   density in [0.005, 0.045] (generator default 0.020)
Nuisance parameters are FIXED where the generator would otherwise randomise them
(haze airlight A, rain angle/length), so severity is the only thing varying.

Same discipline as S0.3: same-scene, 5-fold leave-scene-out, PCA and scaler fitted
on the train fold only, and a shuffled-target control that must land at R^2 <= 0.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "/home/minura/FYP/Workspace/Himeth/scripts/distill")

import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from degradations import add_haze, add_noise, add_rain  # noqa: E402

_s = importlib.util.spec_from_file_location("pdp", "scripts/pca_dim_probe.py")
pdp = importlib.util.module_from_spec(_s)
_s.loader.exec_module(pdp)

DIMS = [2, 4, 8, 16, 32, 64, 128, None]
N_FOLDS = 5
RANGES = {"denoise": (5.0, 55.0), "dehaze": (0.6, 2.6), "derain": (0.005, 0.045)}
CACHE = "/tmp/severity_feats.npz"

torch.set_grad_enabled(False)


def degrade_at(clean, task, sev, rng):
    """Apply `task` at a KNOWN severity, with nuisance parameters pinned."""
    if task == "denoise":
        return np.asarray(add_noise(clean, rng, sigma=float(sev)))
    if task == "dehaze":
        return np.asarray(add_haze(clean, rng, beta=float(sev), A=0.85))
    return np.asarray(add_rain(clean, rng, density=float(sev), angle=0.0,
                               length=20, brightness=200))


def probe_reg(X, y, scenes, dims, seed=0):
    """5-fold leave-scene-out R^2. PCA/scaler fitted on the train fold only."""
    uniq = np.unique(scenes)
    folds = np.array_split(np.random.default_rng(seed).permutation(len(uniq)),
                           N_FOLDS)
    scores = []
    for f in folds:
        te = np.isin(scenes, uniq[f])
        tr = ~te
        Xtr, Xte = X[tr], X[te]
        sc = StandardScaler().fit(Xtr)
        Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        if dims is not None and dims < Xtr.shape[1]:
            p = PCA(n_components=dims, random_state=seed).fit(Xtr)
            Xtr, Xte = p.transform(Xtr), p.transform(Xte)
        m = Ridge(alpha=1.0).fit(Xtr, y[tr])
        pred = m.predict(Xte)
        ss_res = float(((y[te] - pred) ** 2).sum())
        ss_tot = float(((y[te] - y[tr].mean()) ** 2).sum())
        scores.append(1.0 - ss_res / max(ss_tot, 1e-12))
    return float(np.mean(scores)), float(np.std(scores))


def main():
    print("=" * 76)
    print("S2.1 headroom -- is degradation SEVERITY decodable, and does it saturate?")
    print("=" * 76)

    model, ck = pdp.load_model()
    sd = ck["model"]
    model.load_state_dict(sd)
    model.eval()
    import glob
    import random
    random.seed(pdp.SEED)
    pool = sorted(glob.glob(f"{pdp.DATA}/Train/Denoise/*"))
    random.shuffle(pool)
    files = pool[:pdp.N_SCENES]
    gain = pdp.restoration_psnr(model, files)
    print(f"\n[0] sanity gate: model restores {gain:+.2f} dB")
    assert gain > 3.0, "model is not restoring; features would be meaningless"

    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        names = [str(n) for n in z["names"]]
        feats = {t: {n: z[f"f_{t}_{n}"] for n in names} for t in RANGES}
        sev = {t: z[f"s_{t}"] for t in RANGES}
        scenes = z["scenes"]
        print(f"\n[1] loaded cached features from {CACHE}")
    else:
        print(f"\n[1] extracting features ({len(files)} scenes x {len(RANGES)} tasks)")
        tap = pdp.FeatureTap(model)
        names = list(tap.names)
        feats = {t: {n: [] for n in names} for t in RANGES}
        sev = {t: [] for t in RANGES}
        scenes = []
        for si, p in enumerate(files):
            try:
                clean = pdp.center_crop(np.asarray(Image.open(p).convert("RGB")))
            except Exception:
                continue
            scenes.append(si)
            for task, (lo, hi) in RANGES.items():
                r = np.random.default_rng(
                    abs(hash(task + os.path.basename(p))) % (2 ** 31))
                s = float(r.uniform(lo, hi))
                deg = degrade_at(clean, task, s, r)
                model(pdp.to_tensor(deg))
                for n in names:
                    feats[task][n].append(tap.buf[n].copy())
                sev[task].append(s)
            if (si + 1) % 100 == 0:
                print(f"    {si + 1}/{len(files)}", flush=True)
        tap.close()
        for t in RANGES:
            for n in names:
                feats[t][n] = np.asarray(feats[t][n])
            sev[t] = np.asarray(sev[t])
        scenes = np.asarray(scenes)
        np.savez(CACHE, names=np.array(names), scenes=scenes,
                 **{f"f_{t}_{n}": feats[t][n] for t in RANGES for n in names},
                 **{f"s_{t}": sev[t] for t in RANGES})
        print(f"    cached -> {CACHE}")

    # --- sanity: severity must actually change the image -------------------
    print("\n[2] sanity: does the severity parameter do anything?")
    for task, (lo, hi) in RANGES.items():
        clean = pdp.center_crop(np.asarray(Image.open(files[0]).convert("RGB")))
        ps = []
        for s in (lo, (lo + hi) / 2, hi):
            r = np.random.default_rng(7)
            d = degrade_at(clean, task, s, r)
            m = np.mean((d.astype(float) - clean.astype(float)) ** 2)
            ps.append(10 * np.log10(255.0 ** 2 / max(m, 1e-9)))
        mono = ps[0] > ps[1] > ps[2]
        print(f"    {task:<8} input PSNR at (lo, mid, hi) = "
              f"{ps[0]:6.2f} {ps[1]:6.2f} {ps[2]:6.2f}   "
              f"{'monotone OK' if mono else 'NOT MONOTONE'}")
        assert mono, f"{task}: severity parameter is not monotone in damage"

    # --- the probe ---------------------------------------------------------
    print("\n[3] severity R^2 vs PCA dimension (5-fold leave-scene-out)")
    results = {}
    for task in RANGES:
        cat = np.concatenate([feats[task][n] for n in names], axis=1)
        y = sev[task]
        y = (y - y.mean()) / y.std()
        row = {}
        print(f"\n    --- {task} (concat dim {cat.shape[1]}) ---")
        print(f"      {'dims':>6}{'R^2':>9}{'sd':>8}{'vs 16d':>9}")
        r16 = None
        for d in DIMS:
            if d is not None and d > cat.shape[1]:
                continue
            m, s = probe_reg(cat, y, scenes, d)
            row[str(d) if d else "full"] = {"r2": m, "sd": s}
            if d == 16:
                r16 = m
            delta = f"{m - r16:+9.3f}" if r16 is not None else "        -"
            print(f"      {str(d) if d else 'full':>6}{m:>9.3f}{s:>8.3f}{delta}")
        results[task] = row

    print("\n[4] CONTROL: shuffled severity targets (must be R^2 <= 0)")
    for task in RANGES:
        cat = np.concatenate([feats[task][n] for n in names], axis=1)
        y = sev[task]
        y = (y - y.mean()) / y.std()
        ysh = np.random.default_rng(0).permutation(y)
        m, s = probe_reg(cat, ysh, scenes, 16)
        print(f"      {task:<8} shuffled R^2 = {m:+.3f} (sd {s:.3f})")

    print("\n" + "=" * 76)
    print("VERDICT")
    print("=" * 76)
    gains = []
    for task, row in results.items():
        if "16" not in row:
            continue
        r16 = row["16"]["r2"]
        bk = max(row, key=lambda k: row[k]["r2"])
        g = row[bk]["r2"] - r16
        gains.append(g)
        print(f"  {task:<8} 16d R^2 = {r16:6.3f}   best = {row[bk]['r2']:6.3f} "
              f"at {bk:>4}d   gain = {g:+.3f}")
    best = max(gains) if gains else 0.0
    print(f"\n  Largest R^2 gain of any dim over 16, any task: {best:+.3f}")
    if best < 0.05:
        print("\n  KILL S2.1: severity saturates at 16 dims too. Neither type nor")
        print("  magnitude has headroom above PCA-16, so a richer post-decoder")
        print("  representation is unjustified on both axes. Close the branch.")
    else:
        print(f"\n  S2.1 HAS HEADROOM on severity ({best:+.3f} R^2 above 16 dims).")
        print("  Redefine S2.1's criterion on severity and run it.")

    with open("reports/reparam_gate/s2_1_severity_probe.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nwrote reports/reparam_gate/s2_1_severity_probe.json")


if __name__ == "__main__":
    main()
