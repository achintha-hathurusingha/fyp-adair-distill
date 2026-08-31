# S2.1 headroom test — severity, unlike type, is NOT saturated

**Status: S2.1 is revived, with a new criterion.** Ran 2026-08-31 on devon.
`scripts/severity_probe.py` → `reports/reparam_gate/s2_1_severity_probe.json`,
console `s2_1_severity_console.txt`.

S0.3 killed the "wider feature set" premise on degradation **type** and, in
doing so, broke S2.1's criterion: PCA-16 already scores 99.67% on 3-way type ID,
so the +2 pp bar was unreachable — only +0.33 pp exists. The plan therefore said
to test severity headroom before committing to S2.1 at all. This is that test.

## Method

Same discipline as S0.3 — same-scene, 5-fold leave-scene-out, PCA and scaler
fitted on the train fold only, sanity gate on the model first (+11.50 dB).
Severity drawn **continuously** so the probe measures a graded quantity rather
than re-running type ID under another name, with nuisance parameters pinned so
severity is the only thing varying:

| task | parameter | range | generator default |
|---|---|---|---|
| denoise | `sigma` | 5 – 55 | training used discrete {15, 25, 50} |
| dehaze | `beta` | 0.6 – 2.6 | `U(1.0, 2.2)`, airlight `A` pinned to 0.85 |
| derain | `density` | 0.005 – 0.045 | 0.020, angle/length pinned |

**Sanity check on the parameter itself** — degraded-input PSNR must fall
monotonically with severity, else the knob is not doing what the label says:

| task | low | mid | high |
|---|---|---|---|
| denoise | 34.12 | 19.00 | 14.27 ✅ |
| dehaze | 18.25 | 11.89 | 9.61 ✅ |
| derain | 28.48 | 25.31 | 24.82 ✅ |

## Result — severity R² vs PCA dimension

| dims | denoise | dehaze | derain |
|---:|---:|---:|---:|
| 2 | 0.931 | **0.012** | — |
| 4 | 0.947 | 0.281 | — |
| 8 | 0.960 | 0.495 | 0.232 |
| **16** | **0.979** | **0.646** | **0.470** |
| 32 | 0.991 | 0.691 | 0.571 |
| 64 | 0.994 | **0.742** | **0.584** |
| 128 | 0.994 | 0.714 | 0.576 |
| full (496) | 0.992 | 0.526 | **−0.256** |

Gain over 16 dims: denoise **+0.015** (saturated), dehaze **+0.096**,
derain **+0.114**. Fold sd is 0.002–0.12.

**Control:** shuffled severity targets give R² of −0.056 / −0.067 / −0.060 —
correctly at or below zero on all three.

## What this says

**Type is free; magnitude is expensive.** The contrast with S0.3 is the whole
finding:

| | 2 dims | 16 dims |
|---|---|---|
| degradation **type** (3-way) | **99.67%** | 99.67% |
| **severity**, dehaze | **R² 0.012** | R² 0.646 |

Two dimensions identify *which* degradation essentially perfectly, and tell you
**nothing** about how much of it there is. So the student's decoder does encode
degradation identity almost for free, and encodes magnitude poorly — worst on
**derain (0.470)**, which is also the only task with a real teacher gap (3.482 dB).
Derain being worst on both axes is worth noting, but with n=3 and no consistent
ordering between the other two tasks (denoise has better severity encoding than
dehaze but a *larger* gap), this is a suggestion, not a relationship.

**PCA is doing real regularisation work.** At full 496 dims R² *collapses* —
dehaze 0.742→0.526, derain 0.584→**−0.256** — with ~400 samples. "Just use all
the features" is not an available option, which is itself an argument for a
compact, structured representation rather than a wider one.

## Consequence for the plan

**S2.1 is revived.** Its old criterion (`< +2pp over PCA-16` on type ID) stays
withdrawn as unsatisfiable. Replacement, concrete and falsifiable:

> **S2.1 kill:** a DFC-style representation at **16 dims** fails to beat PCA-16's
> severity R² by **≥ +0.10** on derain or dehaze — i.e. fails to reach at 16 dims
> what PCA needs 64 dims for (derain 0.470 → 0.584, dehaze 0.646 → 0.742).

+0.10 is roughly 2 fold-sd, and the target is exactly the headroom dimension
alone buys, so a representation that only matches "more PCA dims" gets no credit.

Note this also gives S2.2 (conditioning) a better signal to condition *on*:
conditioning on a degradation code that already saturates at 2 dims was never
going to carry information the network lacks, which may be part of why v1 FiLM
regressed. A severity-carrying code is a materially different proposition.

## Limits

- GAP features only, matching `DegradationHead`. A DFC band-wise statistic is
  deliberately not GAP, so S2.1's representation is not a subset of what was
  probed here — this measures the *baseline* it must beat, not its ceiling.
- Synthetic severity on same-scene crops; the student trained on real derain and
  dehaze pairs, so these inputs are mildly off-distribution. The price of the
  same-scene control.
- Linear (Ridge) probe, single arm (B0V3-KD-FEAT @81k), single seed.
