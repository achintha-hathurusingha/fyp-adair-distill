# S0.1 — Oracle ceiling for separable oriented kernels

**Status: PASS**, on rain only. Ran 2026-08-31 on devon.
Scripts: `scripts/oriented_ceiling.py`, `scripts/oriented_angle_control.py`.
Data: `reports/reparam_gate/s0_1_oriented_ceiling.json`, `s0_1b_angle_control.json`.
Console: `reports/reparam_gate/s0_1_console.txt`, `s0_1b_console.txt`.

**Kill criterion (plan): oriented ceiling < +0.15 dB over isotropic → drop the
orientation machinery.** Not met on rain (+0.385 dB synthetic, +0.278 dB real),
comfortably met — i.e. killed — on denoise (+0.009) and dehaze (+0.001).

---

## What was actually measured, and why not the obvious thing

S0.2 established that the S3.1 block merges into **one k×k depthwise conv**
(`runs/reparam_oriented_merged.onnx` is a single Conv node). So at deployment the
oriented bank and a plain large kernel have *identical* expressive power. Asking
"how much does orientation add on top of k×k" is therefore vacuous — the answer
is zero by construction. Orientation is an **inductive bias**, not capacity.

The question that does decide the design is:

> Does the optimal restoration filter for each degradation actually *have*
> oriented structure — how much does an isotropy constraint cost, and do
> oriented bands recover it?

Reuses `freq_to_spatial.py`'s pipeline, degradations, crops and test split
exactly, so these numbers sit alongside the existing convolution-theorem result.
Within any *linear* family the optimum is closed-form, so these are true oracles,
not SGD fits that might have found a local minimum. One accumulation pass over
training pairs gives the sufficient statistics `R_dd`, `R_cd`; every family is
then a small linear solve on them.

**Mechanism self-test:** the cross-correlation/FFT sign convention is verified by
recovering a known 5×5 kernel, max error **4.7e-10**. Without it the whole script
could have been fitting a flipped kernel and still produced plausible PSNR.

**Resolution:** differences below ~0.05 dB are not read as real here (`full` and
`oriented4` swap places by ±0.01 dB on held-out data purely through
regularisation). The comparisons below are all ≥ 0.2 dB.

---

## Result 1 — orientation matters for rain, and only for rain

Best oriented-over-isotropic gain per task, over all k ∈ {3…21}:

| task | best k | oriented4 − iso | kernel anisotropy | verdict |
|---|---|---|---|---|
| denoise | 11 | **+0.009 dB** | 1.01× | killed |
| dehaze | 5 | **+0.001 dB** | 1.36× | killed |
| **derain** | 15 | **+0.385 dB** | 3.12× | **PASS** |

For denoise, an isotropic kernel with **10 parameters** exactly matches the
unconstrained 49-parameter k×k oracle. For dehaze *every* family is identical to
3 decimal places at every k — the optimal linear operator for haze is essentially
a scalar gain, which every family contains, because synthetic haze is a global
affine map, not a convolution.

This reproduces the DFC geometry finding — *the entire gain is on rain, the only
anisotropic degradation* — independently, in the restoration domain rather than
the classification domain.

**Mechanism corroboration, independent of PSNR.** The second-moment anisotropy of
the fitted optimal kernel grows with support for rain and not for the others:

| k | denoise | dehaze | derain |
|---|---|---|---|
| 7 | 1.02× | 1.70× | 1.24× |
| 11 | 1.01× | 1.69× | 2.09× |
| 15 | 1.05× | 1.38× | 3.12× |
| 21 | 1.04× | 1.23× | **3.84×** |

The rain kernel's principal angle sits at 1–3°, i.e. elongated *perpendicular* to
the near-vertical streaks — the filter averages across streaks, which is what a
rain-removal filter should do. Dehaze's 1.2–1.7× anisotropy carries **zero** PSNR
consequence, which is a useful reminder that an anisotropy statistic on its own
proves nothing.

### The random-support control

This is the control that decides whether the geometry matters or only the tap
count. `random` draws the same number of free taps at random positions (5 draws):

| k | oriented4 taps | oriented4 − full | random − full |
|---|---|---|---|
| 7 | 49 / 49 | 0.000 | 0.000 |
| 9 | 73 / 81 | −0.000 | −0.004 |
| 11 | 97 / 121 | −0.001 | **−0.862** |
| 15 | 145 / 225 | −0.005 | **−0.857** |
| 21 | 217 / 441 | −0.013 | **−2.048** |

At k ≤ 9 the oriented support covers essentially the whole window (49/49 at k=7),
so it is not a constraint and random ties it — as it must. From k = 11 the
oriented support is a genuine proper subspace, and there the geometry does real
work: oriented4 stays within 0.013 dB of the unconstrained oracle while a random
support of the *same size* loses 0.86–2.05 dB. The orientation structure is not
"having that many taps."

---

## Result 2 — the first reading of the low-rank column was a data artifact

On the standard synthetic rain, `rank3` matched `full` (−0.001 dB) at **28%** of
the parameters, and beat `oriented4` (123 vs 217 params at k=21). Taken at face
value that says: drop the orientation branches, use an axis-aligned separable
kernel — Rigamonti's separability without any orientation machinery.

That reading is wrong, and the reason is in the data generator.
`degradations.add_rain` draws `angle = rng.uniform(-15, 15)` — **the synthetic
rain is near-vertical by construction**, which is exactly why the fitted kernel's
principal angle came out at 1–3° and why axis-aligned separability sufficed.
Axis-aligned low rank is trivially adequate for an axis-aligned degradation.

Refitting every family on rain at **controlled** angles (k=11, dB relative to the
`full` k×k oracle):

| rain angle | full (dB) | oriented4 | cross | diag | iso | rank2 | rank3 | fitted kernel angle |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0° | 27.410 | −0.005 | −0.068 | −0.177 | −0.520 | −0.002 | +0.000 | 0° |
| 22.5° | 28.064 | −0.078 | −0.138 | −0.234 | −0.830 | −0.207 | −0.105 | 22° |
| **45°** | 27.878 | **−0.015** | −0.455 | −0.297 | −0.548 | **−0.458** | **−0.288** | 45° |
| 67.5° | 28.025 | −0.086 | −0.161 | −0.245 | −0.805 | −0.223 | −0.108 | 68° |
| 90° | 27.230 | −0.005 | −0.052 | −0.139 | −0.416 | −0.000 | +0.001 | 88° |

- The fitted kernel's principal angle **tracks the rain angle exactly**
  (0→0, 22.5→22, 45→45, 67.5→68, 90→88). The oracle is finding the real geometry.
- **`oriented4` is within 0.09 dB of the unconstrained oracle at every angle.**
  It is the only family that is.
- `rank2`/`rank3` are excellent at 0°/90° and **collapse at 45°** (−0.458 /
  −0.288). A 45° ridge is full-rank in the axis-aligned basis; low rank cannot
  represent it cheaply.
- `cross` collapses at 45° (−0.455) and `diag` is weakest at 0°/90°. `oriented4`
  beats *both* of its halves at *every* angle — the union is doing work, not just
  whichever half happens to match.

So orientation is required, and the four-orientation bank is the only structure
tested that is robust to the streak angle. The apparent "low rank is enough"
result was an artifact of a ±15° generator prior.

---

## Result 3 — what the real rain data actually looks like

The repo ships 200 real RainTrainL pairs. Streak orientation measured from the
residual (rainy − clean) via the structure tensor:

- mean coherence **0.910** (strongly oriented — the streaks really are directional)
- circular mean **93.4°**, circular sd **13.2°**
- histogram spans 60–120°; nothing outside that

So real RainTrainL is **also near-vertical**, slightly wider than the synthetic
±15° but in the same regime. Fitting on 150 pairs, held out 50:

| k | full | oriented4 | cross | diag | iso | rank2 | rank3 | oriented4 − iso |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | 25.995 | 25.995 | 25.975 | 25.968 | 25.721 | 26.004 | 25.994 | **+0.274** |
| 11 | 25.983 | 25.986 | 25.955 | 25.960 | 25.707 | 25.992 | 25.989 | **+0.278** |
| 15 | 25.974 | 25.983 | 25.951 | 25.955 | 25.702 | 25.984 | 25.987 | **+0.281** |

The kill criterion is passed on real data too (+0.278 vs +0.15). But because real
RainTrainL is near-vertical, `rank2` (42 params) ties `oriented4` (97) here —
exactly as the controlled-angle experiment predicts for 90° rain.

One incidental finding worth keeping: on real rain the **k×k spatial kernels beat
the unconstrained frequency filter on held-out data** (25.99 vs 25.92). The
unconstrained filter has 128×128 complex free parameters and overfits 150 pairs;
the small spatial support acts as a regulariser. The spatial route is not merely
a lossy approximation of the frequency one — here it generalises better.

---

## Decision this licenses for S3.1

1. **Keep the orientation machinery, and keep all four orientations.** It is the
   only family within 0.09 dB of the oracle at every angle, and the 45°/135°
   branches are precisely what fails when they are absent.
2. **Drop it from the denoise path if the block is ever made task-conditional.**
   Orientation is worth +0.009 dB on noise and +0.001 dB on haze. Whatever the
   block buys on those tasks, it will not be through orientation.
3. **k = 11 is the sensible size.** derain oriented4 gains +0.384 dB over
   isotropic at k=11 vs +0.385 at k=15 and +0.365 at k=7 — saturated by 11, and
   k=11 is where the oriented support first becomes a real constraint (97 of 121
   taps). Consistent with the existing 7×7–11×11 convolution-theorem result.
4. **A design caution.** On the corpora we currently have — synthetic ±15° and
   real RainTrainL at 93°±13° — a cheap axis-aligned `rank2` kernel would score
   the same as the full oriented bank. The bank's advantage only appears on
   off-axis rain, which neither corpus contains. **If S3.3 shows the oriented
   block beating a plain-kernel control on our current test sets, that gain is
   not coming from the mechanism measured here**, and should be treated with
   suspicion until explained. The place the orientation machinery should pay is
   **S4.3 (composite / unseen degradations)** — that is where to put a
   rotated-rain condition.

## Limits

- **Linear oracles only.** This bounds the linear case, which is the case the
  convolution theorem covers. It says nothing about the nonlinear, content-adaptive
  behaviour a network might learn — as `freq_to_spatial.py`'s docstring already
  states.
- **Global filters.** A single filter must serve every angle in the corpus at
  once, which pushes the oracle toward isotropy and *understates* what an oriented
  basis is worth to a model that can select orientation per image. Experiment C
  exists to work around this, by fitting per angle.
- **Small envelope.** The full linear ceiling is +0.85 dB (synthetic rain) and
  only **+0.22 dB** on real rain. Everything here lives inside that. This is the
  same low ceiling recorded as risk 1 in the plan, and it is the reason S3.3's
  kill criterion (+0.07 dB over a matched control) is the number that actually
  decides Phase 3.
- Grayscale filter estimation, per-channel application — the existing convention,
  kept for comparability.
