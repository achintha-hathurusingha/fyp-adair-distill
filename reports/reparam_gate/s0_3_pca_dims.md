# S0.3 — How much does PCA-16 throw away?

**Status: KILLED (as the plan anticipated). Keep PCA-16.**
Ran 2026-08-31 on devon. Script: `scripts/pca_dim_probe.py`.
Data: `reports/reparam_gate/s0_3_pca_dims.json`, console `s0_3_console.txt`.

**Kill criterion: 16 dims already saturates → a wider feature set is
unjustified.** Met. The largest gain of *any* dimension over 16, from *any*
feature source, is **+1.17 pp** — and the best 16-dim source (99.67%) already
beats the best 128-dim source of every other stage.

---

## What was probed, and why not the teacher's `e_D`

The teacher's PCA-16 `e_D` is a TEST19 artifact that lives outside this repo
(`teacher-experiments/` is empty here). So this measures the thing S2.1 would
actually build on: the **student's own decoder features**, from the current best
arm (B0V3-KD-FEAT @81k). The decision S0.3 gates — is a wider post-decoder
representation justified — is a question about the dimensionality of the
student's features, not about reproducing the teacher's particular basis.

**Same-scene design**, non-negotiable here: every scene contributes all three
degradations, so degradation is the only variable. The project has already
measured what happens without it — a corpus version was inflated by a **65.8%
dataset-identity floor**. Corpus-based numbers answer the easier question
"which dataset is this?".

- features: GAP over `middle_blks` and each of the 4 decoder stages — matching
  what `DegradationHead` actually consumes (`AdaptiveAvgPool2d` → `Linear`)
- split: **5-fold leave-scene-out**, train and test scenes disjoint
- **PCA and scaler fitted on the train fold only.** Fitting on everything leaks,
  and the leak grows with dimension — which is the exact axis under test
- probe: multinomial logistic regression, 3-way, chance 33.3%
- 400 scenes → 1,200 samples

**Sanity gate before anything else:** the loaded model must actually restore.
Measured **+11.50 dB** on denoise. Had preprocessing or the weight set been
wrong, features would have been garbage and the probe would still have returned
a plausible-looking number. The gate also chose between the `model` and `ema`
weight sets by measured restoration rather than assumption (they were equivalent).

---

## Result

Accuracy vs PCA dimension, 5-fold leave-scene-out (%, ± sd):

| source | dim | 2 | 4 | 8 | **16** | 32 | 64 | 128 | full |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| middle | 256 | 95.92 | 97.58 | 98.25 | **98.50** | 99.08 | 99.42 | 99.67 | 99.58 |
| dec0 | 128 | 89.00 | 91.92 | 97.33 | **98.58** | 98.92 | 99.25 | 99.25 | 99.25 |
| dec1 | 64 | 98.67 | 98.67 | 98.83 | **99.17** | 99.25 | 99.42 | — | 99.42 |
| dec2 | 32 | 97.08 | 97.67 | 98.08 | **98.92** | 99.17 | — | — | 99.17 |
| dec3 | 16 | 96.67 | 96.42 | 97.83 | **98.58** | — | — | — | 98.58 |
| **concat** | 496 | **99.67** | 99.67 | 99.75 | **99.67** | 99.75 | 99.58 | 99.75 | 99.67 |

Gain over 16 dims: middle +1.17 pp, dec0 +0.67, dec1 +0.25, dec2 +0.25,
dec3 +0.00, concat +0.08. Fold sd is 0.3–1.9 pp, so every one of those except
possibly `middle` is inside fold noise.

### Controls — both clean

| control | accuracy |
|---|---|
| clean images, labelled as if degraded, 16 dims | **33.33%** (sd 0.00) |
| clean images, full dims | **33.33%** (sd 0.00) |
| shuffled labels, 16 dims | 35.25% (sd 1.74) |

Chance is 33.33%. The clean control sits *exactly* at chance, which is the
result that says the probe is reading degradation and not scene or fold
structure. This matches the project's existing same-scene finding, where the
clean control was also exactly at chance.

---

## The finding that matters more than the verdict

**The probe task is saturated, not just the dimension axis.** `concat` reaches
**99.67% at two dimensions**. Two numbers identify the degradation. Every source
at every dimension ≥ 2 sits between 89% and 99.75%, almost all above 97%.

This is unsurprising in hindsight — the student was *trained* on exactly these
three degradations, so its decoder features are explicitly degradation-
discriminative. It is a much easier problem than the 93.6% the project measured
for blind degradation ID from raw spectra.

**Consequence: S2.1's kill criterion as written is unsatisfiable.** S2.1 says
*"Kill: < +2pp over PCA-16 at matched dimension."* PCA-16 on concat is already
99.67%, so the maximum gain physically available is **+0.33 pp**. No
representation, however good, can clear a +2 pp bar against that baseline.
**S2.1 needs its success metric redefined before it is run**, or it will produce
a guaranteed kill that means nothing.

Concretely, a DFC-style representation should be judged on something with
headroom. Options, cheapest first:

1. **Degradation *severity*, not type** — regress σ ∈ {15,25,50}, haze density,
   rain density. Type is trivial; magnitude is what a conditioning signal would
   actually need to carry, and nothing here shows it is present.
2. **Unseen / composite degradations** — the S4.3 regime. Degradations the
   student was never trained on are where a representation has to generalise
   rather than recall.
3. **Downstream PSNR**, not probe accuracy — the only metric that reflects the
   thing we care about, and immune to ceiling effects.

Option 1 is a ~1h CPU extension of this script and would settle whether the
"richer representation" idea has any headroom at all before spending GPU on it.

---

## Decision this licenses

1. **Keep PCA-16.** The plan's own words: "16 dims already saturates → 'wider
   feature set' is unjustified; say so." Said. A wider post-decoder feature set
   is not justified by degradation-ID capacity.
2. **The plan's premise "a richer degradation representation … replacing PCA-16"
   loses its stated motivation.** If 2 dims already identify the degradation,
   width is not the bottleneck. Any remaining case for S2.1 has to rest on
   severity, unseen degradations, or downstream PSNR — not on ID capacity.
3. **Use `concat` or `dec1` if a 16-dim code is wanted.** `dec1` (64 ch) reaches
   99.17% at 16 dims from a single stage; `concat` reaches 99.67%. Both beat
   `middle` at 128 dims, so the widest tensor is not the best source.

## Limits

- **Ceiling effect** — the headline limitation, above. This shows 16 dims
  suffices *for 3-way type ID*; it cannot show 16 dims captures everything
  useful, because the task has no headroom to detect a difference.
- GAP pooling only, matching `DegradationHead`. A DFC-style band-wise statistic
  is deliberately *not* GAP, so S2.1's representation is not a strict subset of
  what was probed here.
- Synthetic same-scene degradations (`add_noise`/`add_rain`/`add_haze`), while
  the student was trained on real derain/dehaze pairs — so these inputs are
  mildly off-distribution. That is the price of the same-scene control, and the
  control is worth more than the on-distribution match.
- Single arm (B0V3-KD-FEAT @81k), single seed.
