"""S0.1 -- Oracle ceiling for separable ORIENTED spatial kernels.

Question the plan asks: does the Phase-3 block need orientation machinery, or
should it just use a plain large kernel?
  Kill: oriented ceiling < +0.15 dB over isotropic -> drop the orientation
  machinery.

WHAT THIS IS AND IS NOT MEASURING
---------------------------------
The S3.1 block merges into ONE k x k depthwise conv (measured in S0.2:
`runs/reparam_oriented_merged.onnx` is a single Conv node). So at DEPLOYMENT the
oriented bank and a plain large kernel have *identical* expressive power -- an
arbitrary k x k kernel. Orientation is therefore an INDUCTIVE BIAS / training-time
parameterization, not extra capacity.

That makes the useful ceiling question not "how much can orientation add on top
of k x k" (answer: nothing, by construction) but:

    Does the optimal restoration filter for each degradation ACTUALLY HAVE
    oriented structure -- i.e. how much does an isotropy constraint cost,
    and how much of that cost do oriented bands recover?

If an isotropic kernel recovers essentially all of what an unconstrained k x k
kernel achieves, then orientation is not a property of the problem, the block's
oriented branches are decoration, and a plain large kernel is the right design.
If the isotropy constraint is expensive and oriented bands recover it, the
orientation structure is tracking something real. That is the decision S0.1 gates.

METHOD
------
Reuses `freq_to_spatial.py`'s data pipeline, degradations, crops and test split
exactly, so numbers are comparable to the existing convolution-theorem result
(full linear filter worth only +0.68 dB dehaze / +0.87 dB derain -- everything
below lives inside that envelope).

The optimum within any LINEAR family of kernels is available in closed form, so
these are true oracles, not SGD fits that might have found a local minimum.
With cross-correlation output  out[x] = sum_m theta[m] d[x+m],  Parseval gives

    A[m,n] = R_dd[m-n]      (autocorrelation of the degraded image)
    b[m]   = R_cd[m]        (cross-correlation of clean with degraded)

so one accumulation pass over training pairs yields R_dd and R_cd, and EVERY
kernel family is then a small linear solve on the same sufficient statistics.
Grouped families (isotropic) sum A and b over their groups.

FAMILIES (all supported inside k x k; b = 1, i.e. band width 3, matching the
S0.2 stub's kp=3)
  full       all k^2 taps                              <- ceiling for ANY k x k op
  iso        radially symmetric, f(di^2 + dj^2)        <- the isotropy constraint
  cross      |di|<=b or |dj|<=b                        <- 0 / 90 deg bands only
  diag       |di-dj|<=b or |di+dj|<=b                  <- 45 / 135 deg bands only
  oriented4  cross OR diag                             <- the S3.1 block's support
  random     random taps, count matched to oriented4   <- CONTROL
  rank1/2/3  low-rank (ALS), separable but NOT oriented <- Rigamonti's claim

`random` is the control that matters: if oriented4 only ties a random support
with the same number of free taps, the orientation structure is doing nothing
beyond having that many taps, and the geometry story is dead.

Also reported: the anisotropy of the fitted optimal kernel itself (second-moment
eigenvalue ratio). That is a MECHANISM check -- it says whether the optimal
kernel is oriented -- independent of any PSNR delta.
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
from PIL import Image

from degradations import add_haze, add_noise, add_rain  # noqa: E402

DATA = "/home/minura/fyp-adair-distill/data"
N_TRAIN, N_TEST, PATCH, SEED = 300, 80, 128, 0
KS = [3, 5, 7, 9, 11, 15, 21]
BAND = 1                      # half-width -> band width 3, matches S0.2 kp=3
N_RANDOM_DRAWS = 5
TASKS = ["denoise", "dehaze", "derain"]

random.seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------- data utils
# (identical to freq_to_spatial.py so the two are directly comparable)
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


# ------------------------------------------------------------ kernel algebra
def kernel_to_response(theta, offsets, n=PATCH):
    """FFT response of the cross-correlation  out[x] = sum_m theta[m] d[x+m].

    K[x] = theta_m at x = -m (mod n), so that fft2(K)[u] = sum_m theta_m
    e^{+2pi i u m / n}, which is the multiplier on F_d. The sign convention is
    the one place this whole script could be silently wrong, so `self_test`
    below verifies it by recovering a known kernel.
    """
    k = np.zeros((n, n))
    for t, (di, dj) in zip(theta, offsets):
        k[(-di) % n, (-dj) % n] += t
    return np.fft.fft2(k)


def apply_response(img, resp):
    """Apply a frequency response per channel, mean-preserved (same convention
    as freq_to_spatial.py's apply_trunc)."""
    out = np.zeros(img.shape, dtype=np.float64)
    for ch in range(img.shape[2]):
        x = img[..., ch].astype(np.float64)
        mu = x.mean()
        out[..., ch] = np.real(np.fft.ifft2(np.fft.fft2(x - mu) * resp)) + mu
    return np.clip(out, 0, 255)


def offsets_for(k):
    r = k // 2
    return [(di, dj) for di in range(-r, r + 1) for dj in range(-r, r + 1)]


def build_normal_equations(R_dd, R_cd, offsets):
    """A[m,n] = R_dd[m-n], b[m] = R_cd[m], indices taken modulo PATCH."""
    p = len(offsets)
    off = np.array(offsets)
    di = off[:, 0][:, None] - off[:, 0][None, :]
    dj = off[:, 1][:, None] - off[:, 1][None, :]
    A = R_dd[di % PATCH, dj % PATCH]
    b = R_cd[off[:, 0] % PATCH, off[:, 1] % PATCH]
    return A, b, p


def solve_ls(A, b):
    """Ridge-stabilised solve. The ridge is ~1e-9 of the scale of A, far below
    any effect we would report, and only guards a singular grouped basis."""
    ridge = 1e-9 * (np.trace(A) / max(len(b), 1) + 1e-30)
    return np.linalg.lstsq(A + ridge * np.eye(len(b)), b, rcond=None)[0]


def fit_support(R_dd, R_cd, offsets, mask):
    """Oracle kernel restricted to a subset of taps. Returns full-length theta."""
    idx = [i for i, m in enumerate(mask) if m]
    sub = [offsets[i] for i in idx]
    A, b, _ = build_normal_equations(R_dd, R_cd, sub)
    th_sub = solve_ls(A, b)
    theta = np.zeros(len(offsets))
    theta[idx] = th_sub
    return theta, len(idx)


def fit_groups(R_dd, R_cd, offsets, groups):
    """Oracle kernel with taps TIED within groups (used for isotropy).
    Basis element g is the indicator of group g, so A_gh = sum_{m in g, n in h}
    R_dd[m-n] and b_g = sum_{m in g} R_cd[m]."""
    A_full, b_full, _ = build_normal_equations(R_dd, R_cd, offsets)
    keys = sorted(set(groups))
    G = np.zeros((len(offsets), len(keys)))
    for i, g in enumerate(groups):
        G[i, keys.index(g)] = 1.0
    A = G.T @ A_full @ G
    b = G.T @ b_full
    coef = solve_ls(A, b)
    return G @ coef, len(keys)


def fit_lowrank(R_dd, R_cd, offsets, k, rank, iters=60, restarts=3, seed=0):
    """Oracle kernel constrained to rank <= `rank` (a sum of `rank` separable
    outer products -- axis-aligned separability, no orientation). Bilinear, so
    solved by alternating least squares with restarts. Reported as the direct
    test of Rigamonti's 'linear combinations of far fewer separable filters'."""
    A, b, _ = build_normal_equations(R_dd, R_cd, offsets)
    rng = np.random.default_rng(seed)
    best_theta, best_obj = None, np.inf

    def objective(theta):
        return float(theta @ A @ theta - 2 * b @ theta)

    for _ in range(restarts):
        U = rng.standard_normal((k, rank)) * 0.1
        V = rng.standard_normal((k, rank)) * 0.1
        for it in range(iters):
            # theta_{ij} = sum_r U[i,r] V[j,r]; fix one factor, solve the other
            for fix_v in (True, False):
                M = np.zeros((k * k, k * rank))
                for i in range(k):
                    for j in range(k):
                        for r in range(rank):
                            if fix_v:
                                M[i * k + j, i * rank + r] = V[j, r]
                            else:
                                M[i * k + j, j * rank + r] = U[i, r]
                AM = M.T @ A @ M
                bM = M.T @ b
                p = solve_ls(AM, bM)
                if fix_v:
                    U = p.reshape(k, rank)
                else:
                    V = p.reshape(k, rank)
        theta = (U @ V.T).reshape(-1)
        o = objective(theta)
        if o < best_obj:
            best_obj, best_theta = o, theta
    return best_theta, k * rank * 2 - rank


def anisotropy(theta, offsets):
    """Second-moment eigenvalue ratio of |kernel| -- a MECHANISM measure of
    whether the optimal kernel is actually oriented, independent of PSNR.
    1.0 = isotropic; larger = more elongated. Also returns the principal angle."""
    w = np.abs(np.asarray(theta))
    if w.sum() <= 0:
        return 1.0, 0.0
    off = np.array(offsets, dtype=float)
    w = w / w.sum()
    mu = (w[:, None] * off).sum(0)
    d = off - mu
    C = (w[:, None, None] * (d[:, :, None] * d[:, None, :])).sum(0)
    ev = np.linalg.eigvalsh(C)
    ev = np.clip(ev, 1e-12, None)
    vecs = np.linalg.eigh(C)[1]
    ang = np.degrees(np.arctan2(vecs[1, -1], vecs[0, -1])) % 180
    return float(ev[-1] / ev[0]), float(ang)


# ------------------------------------------------------------------ self test
def self_test():
    """Verify the cross-correlation / FFT sign convention by recovering a known
    kernel exactly. Without this the whole script could be fitting a flipped
    kernel and still produce plausible-looking PSNR numbers."""
    rng = np.random.default_rng(0)
    k = 5
    offs = offsets_for(k)
    true = rng.standard_normal(len(offs)) * 0.2
    resp = kernel_to_response(true, offs)

    S_dd = np.zeros((PATCH, PATCH))
    S_cd = np.zeros((PATCH, PATCH), dtype=np.complex128)
    for _ in range(24):
        d = rng.standard_normal((PATCH, PATCH))
        d -= d.mean()
        Fd = np.fft.fft2(d)
        Fc = Fd * resp                      # clean := exactly the kernel applied
        S_dd += np.abs(Fd) ** 2
        S_cd += Fc * np.conj(Fd)
    R_dd = np.real(np.fft.ifft2(S_dd))
    R_cd = np.real(np.fft.ifft2(np.conj(S_cd)))

    A, b, _ = build_normal_equations(R_dd, R_cd, offs)
    est = solve_ls(A, b)
    err = float(np.abs(est - true).max())
    print(f"  self-test: recovered a known 5x5 kernel, max|err| = {err:.3e}")
    assert err < 1e-6, "sign/scale convention is wrong -- everything below is void"


# ----------------------------------------------------------------------- main
def main():
    print("=" * 78)
    print("S0.1 -- oracle ceiling for separable oriented kernels")
    print("=" * 78)
    print("\n[0] mechanism self-test")
    self_test()

    pool = sorted(glob.glob(f"{DATA}/Train/Denoise/*"))
    random.shuffle(pool)
    tr_files, te_files = pool[:N_TRAIN], pool[N_TRAIN:N_TRAIN + N_TEST]
    print(f"\nscene-disjoint: {len(tr_files)} train / {len(te_files)} test crops")
    print(f"band half-width b={BAND} (band width {2*BAND+1}, matches S0.2 kp=3)")

    results = {}
    for task in TASKS:
        print(f"\n{'=' * 78}\n=== {task} ===\n{'=' * 78}", flush=True)

        # ---- sufficient statistics over training pairs --------------------
        S_dd = np.zeros((PATCH, PATCH))
        S_cd = np.zeros((PATCH, PATCH), dtype=np.complex128)
        n_used = 0
        for p in tr_files:
            try:
                clean = center_crop(np.asarray(Image.open(p).convert("RGB")))
            except Exception:
                continue
            r = np.random.default_rng(abs(hash(os.path.basename(p))) % (2 ** 31))
            deg = degrade(clean, task, r)
            if deg.shape != clean.shape:
                continue
            c = clean.astype(np.float64).mean(2)
            d = deg.astype(np.float64).mean(2)
            c -= c.mean()
            d -= d.mean()
            Fc, Fd = np.fft.fft2(c), np.fft.fft2(d)
            S_dd += np.abs(Fd) ** 2
            S_cd += Fc * np.conj(Fd)
            n_used += 1
        R_dd = np.real(np.fft.ifft2(S_dd))
        R_cd = np.real(np.fft.ifft2(np.conj(S_cd)))
        H_full = S_cd / (S_dd + 1e-8)     # unconstrained frequency filter
        print(f"  accumulated over {n_used} training crops")

        # ---- held-out set, precomputed ------------------------------------
        test = []
        for p in te_files:
            try:
                clean = center_crop(np.asarray(Image.open(p).convert("RGB")))
            except Exception:
                continue
            r = np.random.default_rng(abs(hash("te" + os.path.basename(p))) % (2 ** 31))
            deg = degrade(clean, task, r)
            if deg.shape != clean.shape:
                continue
            test.append((clean, deg))

        p_deg = float(np.mean([psnr(c, d) for c, d in test]))
        p_freq = float(np.mean([psnr(c, apply_response(d, H_full)) for c, d in test]))
        print(f"  degraded input                 {p_deg:6.3f} dB")
        print(f"  unconstrained frequency filter {p_freq:6.3f} dB ({p_freq - p_deg:+.3f})"
              "   <- absolute linear ceiling")

        def evaluate(theta, offsets):
            resp = kernel_to_response(theta, offsets)
            return float(np.mean([psnr(c, apply_response(d, resp)) for c, d in test]))

        task_rows = []
        for k in KS:
            offs = offsets_for(k)
            r_ = k // 2
            di = np.array([o[0] for o in offs])
            dj = np.array([o[1] for o in offs])

            m_full = np.ones(len(offs), bool)
            m_cross = (np.abs(di) <= BAND) | (np.abs(dj) <= BAND)
            m_diag = (np.abs(di - dj) <= BAND) | (np.abs(di + dj) <= BAND)
            m_or4 = m_cross | m_diag
            groups_iso = [int(a * a + b_ * b_) for a, b_ in offs]

            fams = {}
            th, np_ = fit_support(R_dd, R_cd, offs, m_full)
            fams["full"] = (th, np_)
            th, np_ = fit_groups(R_dd, R_cd, offs, groups_iso)
            fams["iso"] = (th, np_)
            th, np_ = fit_support(R_dd, R_cd, offs, m_cross)
            fams["cross"] = (th, np_)
            th, np_ = fit_support(R_dd, R_cd, offs, m_diag)
            fams["diag"] = (th, np_)
            th, np_ = fit_support(R_dd, R_cd, offs, m_or4)
            fams["oriented4"] = (th, np_)

            # CONTROL: random supports with the SAME number of free taps
            n_taps = int(m_or4.sum())
            rand_scores = []
            for s in range(N_RANDOM_DRAWS):
                rng = np.random.default_rng(1000 * k + s)
                sel = rng.choice(len(offs), size=n_taps, replace=False)
                mr = np.zeros(len(offs), bool)
                mr[sel] = True
                th_r, _ = fit_support(R_dd, R_cd, offs, mr)
                rand_scores.append(evaluate(th_r, offs))

            for rank in (1, 2, 3):
                if rank <= k:
                    th, np_ = fit_lowrank(R_dd, R_cd, offs, k, rank)
                    fams[f"rank{rank}"] = (th, np_)

            row = {"k": k, "n_taps_oriented4": n_taps,
                   "random_mean": float(np.mean(rand_scores)),
                   "random_min": float(np.min(rand_scores)),
                   "random_max": float(np.max(rand_scores))}
            for name, (th, np_) in fams.items():
                row[name] = {"psnr": evaluate(th, offs), "params": np_}
            ar, ang = anisotropy(fams["full"][0], offs)
            row["aniso_ratio"], row["aniso_angle"] = ar, ang
            task_rows.append(row)

            print(f"\n  --- k = {k} ---   (optimal k x k kernel anisotropy "
                  f"{ar:.2f}x at {ang:.0f} deg)")
            print(f"    {'family':<11}{'params':>7}{'PSNR':>9}{'vs iso':>9}"
                  f"{'vs full':>9}")
            base_iso = row["iso"]["psnr"]
            base_full = row["full"]["psnr"]
            for name in ["full", "oriented4", "cross", "diag", "iso",
                         "rank1", "rank2", "rank3"]:
                if name not in row:
                    continue
                v = row[name]
                print(f"    {name:<11}{v['params']:>7}{v['psnr']:>9.3f}"
                      f"{v['psnr'] - base_iso:>+9.3f}{v['psnr'] - base_full:>+9.3f}")
            print(f"    {'random(ctl)':<11}{n_taps:>7}{row['random_mean']:>9.3f}"
                  f"{row['random_mean'] - base_iso:>+9.3f}"
                  f"{row['random_mean'] - base_full:>+9.3f}"
                  f"   [{row['random_min']:.3f}, {row['random_max']:.3f}]")
            print(f"    KEY: oriented4 - iso = {row['oriented4']['psnr'] - base_iso:+.3f} dB"
                  f"   |  oriented4 - random = "
                  f"{row['oriented4']['psnr'] - row['random_mean']:+.3f} dB", flush=True)

        results[task] = {"psnr_deg": p_deg, "psnr_freq": p_freq, "rows": task_rows}

    # ------------------------------------------------------------- verdict
    print(f"\n{'=' * 78}\nVERDICT (kill: oriented ceiling < +0.15 dB over isotropic)\n{'=' * 78}")
    print(f"\n{'task':<10}{'k':>4}{'iso':>9}{'oriented4':>11}{'full':>9}"
          f"{'or4-iso':>10}{'or4-rand':>10}{'aniso':>8}")
    best = {}
    for task in TASKS:
        for row in results[task]["rows"]:
            d_iso = row["oriented4"]["psnr"] - row["iso"]["psnr"]
            d_rnd = row["oriented4"]["psnr"] - row["random_mean"]
            print(f"{task:<10}{row['k']:>4}{row['iso']['psnr']:>9.3f}"
                  f"{row['oriented4']['psnr']:>11.3f}{row['full']['psnr']:>9.3f}"
                  f"{d_iso:>+10.3f}{d_rnd:>+10.3f}{row['aniso_ratio']:>8.2f}")
            if task not in best or d_iso > best[task][1]:
                best[task] = (row["k"], d_iso, d_rnd, row["aniso_ratio"])
    print("\nBest oriented-over-isotropic gain per task (over all k):")
    passed = []
    for task in TASKS:
        k, d_iso, d_rnd, ar = best[task]
        ok = d_iso >= 0.15
        passed.append(ok)
        print(f"  {task:<9} k={k:<3} oriented4 - iso = {d_iso:+.3f} dB"
              f"   (vs random control {d_rnd:+.3f}, kernel anisotropy {ar:.2f}x)"
              f"   {'PASS' if ok else 'below +0.15'}")
    print(f"\n  {'PASS' if any(passed) else 'KILL'}: orientation "
          f"{'is worth its keep on at least one task' if any(passed) else 'buys < +0.15 dB anywhere -- use a plain large kernel'}")

    os.makedirs("reports/reparam_gate", exist_ok=True)
    with open("reports/reparam_gate/s0_1_oriented_ceiling.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nwrote reports/reparam_gate/s0_1_oriented_ceiling.json")


if __name__ == "__main__":
    main()
