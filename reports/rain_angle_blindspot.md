# A 45° blind spot in the current student — baseline for S3.3

Measured 2026-08-31 on **B0V3-KD-FEAT @90k** (the no-block control A), before
any S3.3 arm finished. `scripts/score_rain_angles.py`,
`scripts/build_rain_angles.py`, data in `reports/rain_angle_profile.json`.

## The finding

Rain synthesised at controlled angles onto the 100 Rain100L test *targets*
(images no arm trained on; content upright, only streak angle varies):

| | native | 0° | 22.5° | **45°** | 67.5° | 90° |
|---|---|---|---|---|---|---|
| degraded input | 25.521 | 29.496 | 29.793 | 29.383 | 30.404 | 29.781 |
| rain layer MAE | 5.69 | 1.68 | 1.73 | 1.77 | 1.55 | 1.49 |
| **restored** | 35.159 | 36.955 | 37.074 | **32.111** | 36.115 | 34.943 |
| **gain** | +9.64 | **+7.46** | **+7.28** | **+2.73** | **+5.71** | **+5.16** |

**The model recovers less than half as much at 45° as at 0°/22.5°.** It is a
sharp outlier, not part of a monotone trend — 67.5° and 90° both recover far
more than 45° despite being further from the training distribution's vertical.

## The control that makes it a result

The obvious alternative explanation is that 45° rain is simply harder input.
It is not:

- degraded-input PSNR is **flat**: 29.383 at 45° vs 29.4–30.4 elsewhere. The
  45° input is only 0.4 dB below the *easiest* angle while its output is 4–5 dB
  below.
- rain-layer MAE is **flat**: 1.77 at 45° vs 1.49–1.77 elsewhere, i.e. 45° has
  essentially the same amount of rain painted on.

Degradation strength is matched across angles, so the dip is a property of the
**model**, not of the data.

## Why this matters for S3.3

1. **There is a large, specific deficit to attack** — ~4.5 dB of missing gain at
   the diagonal — rather than the marginal effect the plan's +0.07 dB criterion
   was braced for. If the oriented block's 45°/135° branches do anything, this
   is where it will show.
2. **It is the S0.1 prediction, realised in a trained network.** S0.1 found
   axis-aligned low-rank kernels match the oracle at 0°/90° and collapse at 45°
   (−0.458 dB), while the 4-orientation bank holds (−0.015). The trained
   student shows the same signature: axis-aligned angles fine, diagonal broken.
3. **It sharpens S3.3's prediction into something falsifiable.** Oriented minus
   plain-k11 should be ≈0 at native/0°/90° and **positive at 45°**. A flat
   profile closes the spatial-orientation route.

## What it is NOT evidence of, yet

- **Architecture vs training distribution is not yet separated.** Both corpora
  are near-vertical (synthetic `add_rain` is U(−15,15); real Rain100L measures
  93°±13°), so the student never saw diagonal rain. The dip could be a pure
  generalisation gap that *any* architecture would show, and that a plain
  large kernel would close just as well. **That is exactly what the plain-k11
  control arm exists to determine** — and it is why S3.3 without that control
  could not have attributed anything.
- Note StudentV3 *already* contains `OrientedStreakGate` with 45°/135°
  diagonal-masked branches, and still craters at 45°. That is a caution against
  assuming the new block will fix it: having diagonal support is evidently not
  sufficient. The difference the S3.1 block brings is that its branches combine
  **linearly** and merge, where `OrientedStreakGate` puts a Sigmoid channel gate
  between bands and fuse — but whether that matters for accuracy is untested.
- Single arm, single seed, synthetic rain.

## Status

S3.3 sequence launched 23:14: `B0V3-KD-K11` (plain 11×11 control) → `B0V3-KD-ORI`
→ `B0V3-KD-ORI-MID`, ~12h each, all identical to B0V3-KD-FEAT apart from the
block. Score with `scripts/score_rain_angles.py` as each finishes.
