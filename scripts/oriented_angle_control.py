"""S0.1b -- does the S0.1 conclusion survive rain that is not near-vertical?

S0.1 (global filter, synthetic rain) found:
  * orientation matters ONLY for rain (+0.385 dB over isotropic; noise/haze ~0)
  * but AXIS-ALIGNED low rank (rank2/rank3) matched or beat the 4-orientation
    band bank at fewer parameters

That second conclusion is CONFOUNDED. `degradations.add_rain` draws
`angle = rng.uniform(-15, 15)`, so the synthetic rain is near-vertical BY
CONSTRUCTION, and the fitted optimal kernel's principal axis came out at 1-3 deg.
Axis-aligned separability is trivially sufficient for an axis-aligned degradation.
Acting on that would delete the 45/135 branches on the basis of an artifact of
the data generator.

There is a second, independent reason S0.1 understates orientation: a SINGLE
global linear filter has to serve every rain angle in the corpus at once, so
angle diversity pushes the oracle filter toward isotropy. A network with an
oriented basis can select orientation per image; a global filter cannot.

Two experiments here, both cheap:

  C. CONTROLLED ANGLE. Generate rain at fixed angles 0/22.5/45/67.5/90 deg and
     refit every family per angle. This isolates the confound completely: if
     axis-aligned rank-2 only wins at 0/90 and collapses at 45, then the
     diagonal branches are load-bearing for angle robustness and S0.1's
     "just use low rank" reading is wrong.

  B. REAL RAIN. The repo ships 200 real RainTrainL pairs. Measure the actual
     streak-angle distribution from the residual (rainy - clean) via the
     structure tensor, and refit the families on real pairs. Tells us which
     regime the deployment data is actually in.

Same fitting machinery as S0.1 (exact linear-least-squares oracles on the
R_dd / R_cd sufficient statistics), imported rather than reimplemented.
"""
from __future__ import annotations

import glob
import importlib.util
import json
import os
import random
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "/home/minura/FYP/Workspace/Himeth/scripts/distill")

import numpy as np
from PIL import Image

from degradations import add_rain  # noqa: E402

_spec = importlib.util.spec_from_file_location("oc", "scripts/oriented_ceiling.py")
oc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(oc)

PATCH = oc.PATCH
DATA = oc.DATA
BAND = oc.BAND
KS = [7, 11, 15]
ANGLES = [0.0, 22.5, 45.0, 67.5, 90.0]
N_TRAIN, N_TEST = 300, 80


def accumulate(pairs):
    """R_dd / R_cd sufficient statistics from (clean, degraded) pairs."""
    S_dd = np.zeros((PATCH, PATCH))
    S_cd = np.zeros((PATCH, PATCH), dtype=np.complex128)
    for clean, deg in pairs:
        c = clean.astype(np.float64).mean(2)
        d = deg.astype(np.float64).mean(2)
        c -= c.mean()
        d -= d.mean()
        Fc, Fd = np.fft.fft2(c), np.fft.fft2(d)
        S_dd += np.abs(Fd) ** 2
        S_cd += Fc * np.conj(Fd)
    return (np.real(np.fft.ifft2(S_dd)),
            np.real(np.fft.ifft2(np.conj(S_cd))),
            S_cd / (S_dd + 1e-8))


def families_at_k(R_dd, R_cd, k):
    offs = oc.offsets_for(k)
    di = np.array([o[0] for o in offs])
    dj = np.array([o[1] for o in offs])
    m_cross = (np.abs(di) <= BAND) | (np.abs(dj) <= BAND)
    m_diag = (np.abs(di - dj) <= BAND) | (np.abs(di + dj) <= BAND)
    groups_iso = [int(a * a + b * b) for a, b in offs]

    out = {}
    out["full"] = oc.fit_support(R_dd, R_cd, offs, np.ones(len(offs), bool))
    out["iso"] = oc.fit_groups(R_dd, R_cd, offs, groups_iso)
    out["cross"] = oc.fit_support(R_dd, R_cd, offs, m_cross)
    out["diag"] = oc.fit_support(R_dd, R_cd, offs, m_diag)
    out["oriented4"] = oc.fit_support(R_dd, R_cd, offs, m_cross | m_diag)
    for r in (1, 2, 3):
        out[f"rank{r}"] = oc.fit_lowrank(R_dd, R_cd, offs, k, r)
    return offs, out


def evaluate(theta, offs, test):
    resp = oc.kernel_to_response(theta, offs)
    return float(np.mean([oc.psnr(c, oc.apply_response(d, resp)) for c, d in test]))


def streak_angle(residual):
    """Dominant orientation + coherence of a rain layer, via the structure
    tensor. Returns (angle in [0,180), coherence in [0,1]). Coherence near 1
    means a strongly oriented layer; near 0 means no preferred direction.
    The reported angle is the RIDGE direction (perpendicular to the gradient)."""
    from scipy.ndimage import gaussian_filter, sobel

    r = residual.astype(np.float64)
    gx, gy = sobel(r, axis=1), sobel(r, axis=0)
    jxx = gaussian_filter(gx * gx, 3.0)
    jyy = gaussian_filter(gy * gy, 3.0)
    jxy = gaussian_filter(gx * gy, 3.0)
    jxx_, jyy_, jxy_ = jxx.mean(), jyy.mean(), jxy.mean()
    # gradient orientation; ridge is perpendicular
    theta_g = 0.5 * np.arctan2(2 * jxy_, jxx_ - jyy_)
    ridge = (np.degrees(theta_g) + 90.0) % 180.0
    tr = jxx_ + jyy_
    disc = np.sqrt((jxx_ - jyy_) ** 2 + 4 * jxy_ ** 2)
    coh = float(disc / (tr + 1e-12))
    return float(ridge), coh


# ============================================================ experiment C
def experiment_c(tr_files, te_files):
    print("=" * 78)
    print("C. CONTROLLED RAIN ANGLE -- does the family ranking depend on angle?")
    print("=" * 78)
    print("  Rain generated at a FIXED angle; every family refitted per angle.")
    print("  0/90 deg are axis-aligned; 45 deg is the case axis-aligned low rank")
    print("  cannot represent cheaply. If rank2 only wins at 0/90, the diagonal")
    print("  branches are load-bearing.\n")

    rows = []
    for ang in ANGLES:
        tr_pairs, te_pairs = [], []
        for files, bucket, tag in ((tr_files, tr_pairs, ""), (te_files, te_pairs, "te")):
            for p in files:
                try:
                    clean = oc.center_crop(np.asarray(Image.open(p).convert("RGB")))
                except Exception:
                    continue
                r = np.random.default_rng(
                    abs(hash(tag + os.path.basename(p))) % (2 ** 31))
                deg = np.asarray(add_rain(clean, r, angle=ang))
                if deg.shape == clean.shape:
                    bucket.append((clean, deg))
        R_dd, R_cd, H = accumulate(tr_pairs)
        p_deg = float(np.mean([oc.psnr(c, d) for c, d in te_pairs]))
        p_freq = float(np.mean([oc.psnr(c, oc.apply_response(d, H)) for c, d in te_pairs]))

        for k in KS:
            offs, fams = families_at_k(R_dd, R_cd, k)
            sc = {n: evaluate(th, offs, te_pairs) for n, (th, _) in fams.items()}
            pr = {n: np_ for n, (_, np_) in fams.items()}
            ar, kang = oc.anisotropy(fams["full"][0], offs)
            rows.append({"angle": ang, "k": k, "psnr_deg": p_deg,
                         "psnr_freq": p_freq, "scores": sc, "params": pr,
                         "aniso": ar, "kernel_angle": kang})
            if k == 11:
                print(f"  --- rain angle {ang:>5.1f} deg,  k={k} ---"
                      f"   degraded {p_deg:.3f} -> linear ceiling {p_freq:.3f}")
                print(f"      optimal kernel: anisotropy {ar:.2f}x at {kang:.0f} deg")
                base = sc["iso"]
                for n in ["full", "oriented4", "cross", "diag", "iso",
                          "rank1", "rank2", "rank3"]:
                    print(f"      {n:<10}{pr[n]:>6}{sc[n]:>9.3f}"
                          f"{sc[n] - base:>+9.3f} vs iso"
                          f"{sc[n] - sc['full']:>+9.3f} vs full")
                print(flush=True)

    print("\n  SUMMARY across angles (k=11), dB relative to the `full` k x k oracle")
    print(f"  {'angle':>7}{'full':>9}{'or4':>9}{'cross':>9}{'diag':>9}"
          f"{'iso':>9}{'rank2':>9}{'rank3':>9}{'aniso':>8}{'kang':>7}")
    for r in rows:
        if r["k"] != 11:
            continue
        s, f = r["scores"], r["scores"]["full"]
        print(f"  {r['angle']:>7.1f}{f:>9.3f}"
              f"{s['oriented4'] - f:>+9.3f}{s['cross'] - f:>+9.3f}"
              f"{s['diag'] - f:>+9.3f}{s['iso'] - f:>+9.3f}"
              f"{s['rank2'] - f:>+9.3f}{s['rank3'] - f:>+9.3f}"
              f"{r['aniso']:>8.2f}{r['kernel_angle']:>7.0f}")
    return rows


# ============================================================ experiment B
def experiment_b():
    print("\n" + "=" * 78)
    print("B. REAL RAIN (RainTrainL pairs shipped in the repo)")
    print("=" * 78)
    inp = sorted(glob.glob(f"{DATA}/Train/Derain/input/*"))
    tgt = sorted(glob.glob(f"{DATA}/Train/Derain/target/*"))
    names_i = {os.path.basename(p): p for p in inp}
    names_t = {os.path.basename(p): p for p in tgt}
    common = sorted(set(names_i) & set(names_t))
    print(f"  {len(common)} paired real rain images")

    pairs = []
    angles, cohs = [], []
    for n in common:
        try:
            deg = oc.center_crop(np.asarray(Image.open(names_i[n]).convert("RGB")))
            clean = oc.center_crop(np.asarray(Image.open(names_t[n]).convert("RGB")))
        except Exception:
            continue
        if deg.shape != clean.shape:
            continue
        pairs.append((clean, deg))
        resid = deg.astype(np.float64).mean(2) - clean.astype(np.float64).mean(2)
        a, c = streak_angle(resid)
        angles.append(a)
        cohs.append(c)

    angles = np.array(angles)
    print(f"  usable pairs: {len(pairs)}")
    print("\n  Measured streak orientation (structure tensor on the residual):")
    print(f"    mean coherence {np.mean(cohs):.3f}  "
          f"(1 = strongly oriented, 0 = no preferred direction)")
    # circular spread on a 180-deg axis
    z = np.exp(2j * np.radians(angles))
    Rbar = float(np.abs(z.mean()))
    circ_mean = float((np.degrees(np.angle(z.mean())) / 2) % 180)
    circ_std = float(np.degrees(np.sqrt(-2 * np.log(max(Rbar, 1e-12)))) / 2)
    print(f"    circular mean {circ_mean:.1f} deg, circular sd {circ_std:.1f} deg, "
          f"R = {Rbar:.3f}")
    hist, edges = np.histogram(angles, bins=12, range=(0, 180))
    print("    angle histogram (deg):")
    for h, e0, e1 in zip(hist, edges[:-1], edges[1:]):
        print(f"      {e0:>5.0f}-{e1:<5.0f} {'#' * int(40 * h / max(hist.max(), 1))} {h}")
    print("    (synthetic add_rain draws angle ~ U(-15, 15) by construction)")

    random.Random(0).shuffle(pairs)
    n_tr = int(len(pairs) * 0.75)
    tr_pairs, te_pairs = pairs[:n_tr], pairs[n_tr:]
    print(f"\n  fit on {len(tr_pairs)} pairs, held out {len(te_pairs)}")
    R_dd, R_cd, H = accumulate(tr_pairs)
    p_deg = float(np.mean([oc.psnr(c, d) for c, d in te_pairs]))
    p_freq = float(np.mean([oc.psnr(c, oc.apply_response(d, H)) for c, d in te_pairs]))
    print(f"    degraded input                 {p_deg:6.3f} dB")
    print(f"    unconstrained frequency filter {p_freq:6.3f} dB ({p_freq - p_deg:+.3f})")

    out = []
    for k in KS:
        offs, fams = families_at_k(R_dd, R_cd, k)
        sc = {n: evaluate(th, offs, te_pairs) for n, (th, _) in fams.items()}
        pr = {n: np_ for n, (_, np_) in fams.items()}
        ar, kang = oc.anisotropy(fams["full"][0], offs)
        out.append({"k": k, "scores": sc, "params": pr,
                    "aniso": ar, "kernel_angle": kang,
                    "psnr_deg": p_deg, "psnr_freq": p_freq})
        print(f"\n    --- k = {k} ---  optimal kernel anisotropy "
              f"{ar:.2f}x at {kang:.0f} deg")
        base = sc["iso"]
        for n in ["full", "oriented4", "cross", "diag", "iso",
                  "rank1", "rank2", "rank3"]:
            print(f"      {n:<10}{pr[n]:>6}{sc[n]:>9.3f}{sc[n] - base:>+9.3f} vs iso"
                  f"{sc[n] - sc['full']:>+9.3f} vs full")
        print(f"      KEY: oriented4 - iso = {sc['oriented4'] - base:+.3f} dB",
              flush=True)
    return {"angles": angles.tolist(), "coherence": cohs,
            "circ_mean": circ_mean, "circ_sd": circ_std, "rows": out}


def main():
    random.seed(oc.SEED)
    np.random.seed(oc.SEED)
    pool = sorted(glob.glob(f"{DATA}/Train/Denoise/*"))
    random.shuffle(pool)
    tr_files, te_files = pool[:N_TRAIN], pool[N_TRAIN:N_TRAIN + N_TEST]

    rows_c = experiment_c(tr_files, te_files)
    res_b = experiment_b()

    print("\n" + "=" * 78)
    print("READING")
    print("=" * 78)
    r45 = [r for r in rows_c if r["k"] == 11 and r["angle"] == 45.0][0]
    r0 = [r for r in rows_c if r["k"] == 11 and r["angle"] == 0.0][0]
    for tag, r in (("rain at  0 deg", r0), ("rain at 45 deg", r45)):
        s, f = r["scores"], r["scores"]["full"]
        print(f"  {tag}: rank2 {s['rank2'] - f:+.3f} vs full | "
              f"cross {s['cross'] - f:+.3f} | diag {s['diag'] - f:+.3f} | "
              f"oriented4 {s['oriented4'] - f:+.3f}")
    print(f"\n  real rain: circular sd of streak angle = {res_b['circ_sd']:.1f} deg "
          f"(synthetic is U(-15,15) -> sd ~8.7 deg)")

    os.makedirs("reports/reparam_gate", exist_ok=True)
    with open("reports/reparam_gate/s0_1b_angle_control.json", "w") as f:
        json.dump({"controlled_angle": rows_c, "real_rain": res_b}, f, indent=2)
    print("\nwrote reports/reparam_gate/s0_1b_angle_control.json")


if __name__ == "__main__":
    main()
