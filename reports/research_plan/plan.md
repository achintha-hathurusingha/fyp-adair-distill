# Scaled student + spatial frequency mining — staged research plan

**Revision 2.** Student size is a **free variable**, not fixed at 7.45M. The
target is reframed accordingly (see "What we are actually aiming at").

**The idea.** A wider/deeper student; a richer degradation representation taken
after the decoder (replacing PCA-16); and a few AdaIR-style frequency-mining
blocks implemented as *learnable spatial kernels* rather than FFT.

---

## What we are actually aiming at

Not "beat the teacher's PSNR." That framing is both harder and less valuable
than it looks:

- The derain gap is **3.48 dB** [corrected, leak-free sets; teacher 38.641 vs
  B0V3-KD-FEAT 35.159]. **It is now the ONLY real gap** — dehaze is 0.33 dB and
  denoise 0.46 dB, i.e. effectively closed. Scaling w16 → w32 typically buys ~0.3-0.5 dB
  in this family, so capacity alone will not close it. The likely causes are
  training length (teacher 285k+ iters, task-specific tuning, vs our 90k) and
  three-task dilution — **not** parameter count.
- The standard 3-degradation benchmark is **saturated**: DFC-IR's SOTA gain over
  the previous best is **+0.12 dB**. It is a low-value, high-cost axis.

**The defensible target instead: match a Restormer-class model with a purely
convolutional architecture that can actually be deployed.**

Two measurements make that a fair fight rather than wishful thinking:

| evidence | value |
|---|---|
| **[measured, TEST18]** AdaIR retrained *without* any frequency module: 28.572 vs 28.674 for the full architecture | its headline mechanism is worth **0.10 dB** — "AdaIR" is functionally Restormer with a dead frequency branch |
| **[measured]** the teacher cannot export to ONNX at all (`torch.fft` has no ONNX op) | it is **undeployable on our target at any size**; we are 100% NPU-mapped, zero fallback |

So parity plus deployability is a real contribution. Beating it by 3.5 dB is not
required, and probably is not available.

**On "less compute" — be precise.** At w32 we would **not** have fewer
parameters (29.5M vs the teacher's 28.78M, 1.02x). What we would have is fewer
*deployable-op obstacles*: no FFT, no MatMul/Softmax, and attention costs
quadratic in spatial size where convolution is linear — so NAFNet-class wins on
FLOPs at high resolution even at matched parameters. **"Fewer parameters" is not
a claim we can make; "deployable and cheaper at resolution" is.**

---

## Why this direction, from our own measurements

| our measurement | what it licenses |
|---|---|
| KD effect tracks the teacher-student gap (r = -0.987 / -0.9999) | a **bigger student** shrinks the gap, so KD should get *more* useful, not less |
| B0V3-KD-FEAT leads both parents on the CORRECTED sets (90k: +0.233 vs KD arm, +0.111 vs GT-only v3) | the architecture x distillation interaction is real, but **KD is worth 0.111 dB, not 0.265** — and it **HURTS derain by 0.388 dB** (GT-only v3 wins that task). Keep KD, but it is now a per-task question, not a global one |
| 7x7-11x11 kernels reproduce the full optimal frequency filter | frequency mining **can** be done spatially |
| separable oriented bands 98.3% vs radial 97.5% vs square 93.3% (dims matched at 24) | the band geometry should be **separable and oriented** |
| AdaIR's own alpha/beta collapse toward equality with depth (AFLB3 0.496/0.497) | it *has* the orientation freedom and does not use it — that is the gap to exploit |

**The constraint that shapes everything.** `torch.fft` does not export to ONNX;
attention is UNKNOWN on all three NPU backends. Any frequency behaviour must be
spatial *at inference*.

---

## Parameter budget (measured, not estimated)

| config | StudentV3 | vs teacher |
|---|---:|---:|
| w16 (current) | 7,447,331 | 0.26x |
| w16 deeper | 14,469,731 | 0.50x |
| w24 | 16,632,387 | 0.58x |
| w24 deeper | 24,471,603 | 0.85x |
| **w32** | **29,458,371** | **1.02x** |

> **Every size must carry an on-device latency number, not just PSNR.** Our
> measured INT8 latencies were already in the seconds. A w32 student is ~4x the
> compute of w16 — without a latency column, Phase 1 optimises us straight out
> of the constraint that started the project.

---

## The literature that makes this buildable

### Structural reparameterization — the key enabler

**RepVGG** (Ding et al., CVPR 2021) and **RepLKNet** (Ding et al., CVPR 2022)
train a rich multi-branch block and then **algebraically merge every branch into
a single convolution** for deployment. RepLKNet specifically trains parallel
*large and small* kernels and merges the small into the large afterwards.

> A frequency-mining block built as parallel oriented/multi-scale spatial
> kernels combined **linearly** collapses at deploy time into **one plain conv**.
> Training-time richness, inference cost of a single convolution, nothing for
> the export gate to reject. Linearity is the binding design constraint —
> merging is exact only if branches combine linearly before any nonlinearity.

### Separable filters — independent support for our geometry result

**Rigamonti, Sironi & Lepetit** (*Learning Separable Filters*, CVPR 2013 /
TPAMI 2015): learned filter banks "can be approximated very precisely as linear
combinations of far fewer separable filters." Anisotropic Gaussians are exactly
separable along their principal axis and its perpendicular.

### Capacity and distillation

Cho & Hariharan (ICCV 2019): student performance degrades with an oversized
teacher; **when the student is large, KD can bring it close to the teacher.**
For WideResNet, increasing *depth* beat increasing *width* per parameter — a
concrete steer for S1.1.

### Degradation representation

AirNet (contrastive), PromptIR (prompts), DA-CLIP (decoupling degradation from
content), DFC (explicit spectral quantification). The field has moved from
implicit to **explicit, measurable** degradation coordinates.

---

## Staged plan

Every stage has a **kill criterion**. The 0.035 dB seed-noise floor is the
yardstick: anything under ~0.07 dB is not a result.

Status: `[ ]` not started · `[~]` running · `[x]` done · `[!]` killed

---

### PHASE 0 — Establish ceilings before building (no GPU)

**S0.1 · Oracle ceiling for separable oriented kernels** `[x]` **PASS (rain only)**
`scripts/oriented_ceiling.py`, `scripts/oriented_angle_control.py` →
`reports/reparam_gate/s0_1_oriented_ceiling.md`.
- **Result:** oriented − isotropic = **+0.385 dB derain** (synthetic), **+0.278 dB**
  (real RainTrainL) — passes. **denoise +0.009, dehaze +0.001 — killed there.**
  Reproduces the DFC geometry finding (gain entirely on rain) in the restoration
  domain. Kernel anisotropy 3.84× for rain vs 1.04× denoise.
- **Random-support control:** from k=11 (where the oriented support is a genuine
  proper subspace) oriented stays within 0.013 dB of the unconstrained oracle
  while a random support of the *same size* loses 0.86–2.05 dB. The geometry is
  doing the work, not the tap count.
- **Confound found and corrected:** `degradations.add_rain` draws
  `angle ~ U(-15,15)`, so synthetic rain is near-vertical **by construction**.
  On that data axis-aligned `rank3` matches the full oracle at 28% of the
  parameters — which would have said "drop the orientation branches". Refitting
  at controlled angles kills that reading: at **45°**, `rank2`/`rank3` collapse
  (−0.458 / −0.288 vs the oracle) while the 4-orientation bank holds at −0.015.
  The bank is the only family within 0.09 dB of the oracle at **every** angle.
- **Real rain measured:** RainTrainL streaks are coherence 0.910, circular mean
  93.4°, sd 13.2° — also near-vertical. So *neither* corpus contains off-axis
  rain (see S3.3 caution, S4.3).
- CPU, ~25 min actual. Depends: nothing.

**S0.2 · NPU export gate for the proposed ops** `[x]` **PASS**
`scripts/probe_reparam_oriented.py` → `reports/reparam_gate/s0_2_export_gate.md`.
- **Result:** merged deployment graph is **one node — a single depthwise `Conv`**.
  **Zero UNKNOWN and zero CAUTION ops on qnn/tflite/tensorrt, in FP32 and INT8**
  (QDQ per-channel adds only Quantize/DequantizeLinear, both supported).
  Holds across dim ∈ {16,32,64} × k ∈ {7,9,11}. Phase 3 is not export-blocked.
- Merge verified, not assumed: max |multi-branch − merged| = **9.5e-07**, and
  ORT(training graph) vs ORT(merged graph) = **1.2e-06** — the PyTorch merge
  being exact does not prove the *exported* one is, so both are checked.
- Per-branch BatchNorm folds exactly at eval, so the RepVGG recipe is available
  at zero deployment cost. Deployment params come out 2.3–2.7× *below* training.
- **Does not establish:** `op_coverage` sees only `Conv` — not depthwise-vs-dense,
  not kernel size. Real AI Hub conversion (S4.4) remains ground truth.
- Minutes. Depends: nothing.

**S0.3 · How much does PCA-16 throw away?** `[x]` **KILL — keep PCA-16**
`scripts/pca_dim_probe.py` → `reports/reparam_gate/s0_3_pca_dims.md`.
Probed the **student's own** decoder features (the teacher's TEST19 `e_D` basis
is not in this repo), same-scene, 5-fold leave-scene-out, PCA fitted on the
train fold only, 400 scenes / 1,200 samples.
- **Result:** largest gain of *any* dim over 16, from *any* source: **+1.17 pp**,
  and mostly inside fold noise (sd 0.3–1.9 pp). The best 16-dim source
  (`concat`, 99.67%) beats the best 128-dim version of every other stage.
  **A wider post-decoder feature set is not justified by degradation-ID capacity.**
- **Controls clean:** clean images labelled as if degraded sit at **exactly
  33.33%** (chance) at both 16 and full dims; shuffled labels 35.25%.
  Sanity gate: the loaded model restores **+11.50 dB** before any probing, so
  the features are not garbage from bad preprocessing.
- **THE IMPORTANT PART — the probe task is saturated, not just the dim axis.**
  `concat` hits **99.67% at TWO dimensions**. Unsurprising in hindsight: the
  student was trained on exactly these three degradations. This means the
  premise "a richer degradation representation replacing PCA-16" has lost its
  stated motivation — **width is not the bottleneck** — and it breaks S2.1's
  kill criterion (see there).
- CPU, ~10 min actual. Depends: nothing.

---

### PHASE 1 — Scaling (now the centre of the plan)

**S1.1 · Width/depth scaling sweep, GT-only** `[ ]`
w16 (have) → w16-deeper → w24 → w32, GT-only, 90k, matched schedule. Literature
says prefer **depth over width** per parameter, so w16-deeper and w24-deeper are
the efficient candidates.
- **Deliver:** PSNR, per-task teacher gap, **and on-device latency** vs params.
- **Success:** monotone improvement with the teacher gap visibly shrinking.
- **Kill:** PSNR saturates before the gap closes → capacity is *not* the binding
  constraint. This is the direct test of whether the 3.54 dB derain gap is
  capacity or training protocol. If killed, go to S1.3.
- 3-4 x ~9-30h GPU (cost scales with width). Depends: S0.3.

**S1.2 · Does KD's value change as the student grows?** `[ ]`
Re-run KD at each size. **The sharpest prediction in the plan**, and the direct
answer to "can we forget distillation?": KD harm should shrink monotonically
with capacity, derain (largest gap) flipping last.
- **Success:** monotone trend either way — both directions are informative.
- **Kill:** flat across sizes → the capacity-gap account is wrong and
  `reports/kd_lit_review/review.md` needs correcting.
- 3-4 GPU runs. Depends: S1.1.

**S1.3 · Is the gap training protocol, not architecture?** `[ ]`
Only if S1.1 is killed. Train the *current* w16 to 285k (teacher-matched) and/or
single-task specialists, to separate capacity from schedule and multi-task
dilution.
- **Kill:** if a 285k single-task w16 still trails by >2 dB on derain, the gap is
  neither capacity nor schedule and needs a different explanation entirely.
- 1-2 x ~30h GPU. Depends: S1.1 outcome.

---

### PHASE 2 — Richer degradation representation

**S2.1 · DFC-style representation from decoder features** `[ ]`
**← CRITERION BROKEN BY S0.3, REDEFINE BEFORE RUNNING**
Band-wise residual-to-degraded ratios on *decoder features*, using the
**separable oriented** bands S0.1 validated. Residual estimated as
degraded-minus-intermediate-restoration (DFC's trick — no GT at inference).
- ~~**Kill:** < +2pp over PCA-16 at matched dimension.~~ **Unsatisfiable.**
  S0.3 measured PCA-16 on decoder features at **99.67%**, so the maximum gain
  physically available is **+0.33 pp**. No representation, however good, can
  clear a +2 pp bar. Running this as written produces a guaranteed kill that
  means nothing.
- **Redefine on a metric with headroom.** Cheapest first:
  1. **Degradation *severity*, not type** — regress σ ∈ {15,25,50}, haze density,
     rain density. Type is trivial; magnitude is what a conditioning signal
     would actually have to carry, and S0.3 shows nothing about whether it is
     present. ~1h CPU extension of `scripts/pca_dim_probe.py`.
  2. **Unseen / composite degradations** (the S4.3 regime) — where a
     representation must generalise rather than recall.
  3. **Downstream PSNR** rather than probe accuracy — the only metric immune to
     ceiling effects, and the one we actually care about.
- **Do (1) before committing to this stage at all** — if severity is also
  saturated, the whole "richer representation" branch is dead cheaply.
- CPU ~3h. Depends: S0.1 `[x]`, S0.3 `[x]`.

**S2.2 · Conditioning that does not fight the KD loss** `[ ]`
v1 FiLM regressed because it modulated the exact tensor feature-KD reads.
Condition on a **different** tensor (decoder-side), with a matched control.
- **Kill:** any regression vs control > 0.07 dB.
- 2 GPU runs. Depends: S2.1, S1.2 (for the size to use).

---

### PHASE 3 — Spatial frequency-mining block

**S3.1 · Build the block, reparameterizable by construction** `[ ]`
Parallel separable oriented kernels (0/45/90/135 + isotropic), learnable
per-band coefficients, combined **linearly** so branches can merge.
- **Deliver:** module + operators-off byte-identity test (the no-confound check
  used for v3).
- **Settled by S0.1/S0.2 — inherit, do not redecide:**
  - **Keep all four orientations.** Only family within 0.09 dB of the oracle at
    every rain angle; the 45°/135° branches are exactly what fails without them.
  - **k = 11.** Gain saturates by then (+0.384 vs +0.385 at k=15, +0.365 at k=7)
    and it is where the oriented support first becomes a real constraint (97 of
    121 taps). Consistent with the 7×7–11×11 convolution-theorem result.
  - **No normalisation and no nonlinearity between the branches and the sum.**
    Any later "improvement" adding either silently destroys the whole deployment
    argument (plan risk 2). S0.2's exactness check is the guard.
  - The working kernel algebra (`full_conv2d_depthwise`, `pad_to`, the
    gradient-masked diagonal support) transfers directly from
    `scripts/probe_reparam_oriented.py`.
  - Orientation is worth +0.009 dB on denoise and +0.001 on haze — if the block
    is ever made task-conditional, it is rain-only.
- ~2h dev. Depends: S0.1 `[x]`, S0.2 `[x]` — **both cleared**.

**S3.2 · Prove the merge is exact** `[~]` **← GATE — pre-satisfied on the stub**
Merge multi-branch → one conv; assert max |diff| < 1e-5 on random input; export
and confirm a single Conv in the graph.
- **Kill:** merge not exact → a nonlinearity has crept in; redesign.
- **Already passing on the S0.2 stub** (9.5e-07 merge, 1.2e-06 through ORT,
  exactly 1 Conv node). Boundary case checked deliberately: 9×9 inputs with k=7
  are almost entirely boundary, and the merge holds there because zero-padding
  commutes through the separable pair. Note the equivalent kernel is the **full
  linear convolution** of the two branch kernels, not their cross-correlation —
  `conv2d` is correlation, so the merge flips one kernel. A merge that gets that
  wrong still passes an interior-only test.
- **Still required against the real S3.1 module** once it lands in `src/models/`.
- ~1h. Depends: S3.1. **Do not train until this passes.**

**S3.3 · Train with the block, ablate placement** `[ ]`
Middle-only vs decoder-only vs both, one variable at a time.
- **Success:** > +0.07 dB over the matched no-block control.
- **Caution from S0.1 — read before believing a positive.** On both corpora we
  have (synthetic ±15°, real RainTrainL 93°±13°) a cheap axis-aligned `rank2`
  kernel scores the same as the full oriented bank; the bank's advantage only
  appears on off-axis rain, which neither corpus contains. **So a win here is
  not coming from the mechanism S0.1 measured**, and needs a different
  explanation before it is claimed as one. The place orientation should pay is
  S4.3.
- **Kill:** three placements all null → the spatial frequency route is closed,
  and that is a publishable negative given the convolution-theorem ceiling.
- 3 GPU runs. Depends: S3.2, S1.1.

---

### PHASE 4 — Integration and honesty checks

**S4.1 · Full model** `[ ]` — best size + representation + block placement, with
KD if S1.2 says it still helps. Depends: S1.2, S2.2, S3.3.

**S4.2 · Multi-seed** `[ ]` — 3 seeds on the final arm and its control. Every
claim so far is single-seed; this is what makes the result reportable.

**S4.3 · Composite / unseen degradations** `[ ]` — DFC's biggest wins were
composite (+1.17 dB) and unseen (+0.59), not standard 3-task (+0.12). **Our
entire evaluation is one-degradation-at-a-time, the regime where this family
helps least.** Build a mixed-degradation test set and evaluate there.
- **Add a rotated-rain condition** (0/22.5/45/67.5/90°). S0.1 showed the oriented
  bank's whole advantage over cheaper families lives at off-axis angles, and
  *neither* of our current corpora contains any. Without this condition the
  orientation machinery cannot be shown to earn its keep anywhere.
  `scripts/oriented_angle_control.py` already generates rain at a fixed angle.

**S4.4 · Final NPU measurement** `[ ]` — real Qualcomm AI Hub latency, FP32 and
INT8, on the merged deployment graph, at the chosen size.

---

## Ordering

```
S0.2 (gate, minutes) ─┐
S0.1 ─────────────────┼─> S3.1 ─> S3.2 (gate) ─> S3.3 ─┐
S0.3 ─> S2.1 ─> S2.2 ─┘                                 ├─> S4.1 ─> S4.2 ─> S4.3 ─> S4.4
S1.1 ─> S1.2 ──────────────────────────────────────────┘
  └──> S1.3 (only if S1.1 killed)
```

Phase 0 is all CPU and can run alongside GPU work. **S0.2 and S3.2 are gates** —
cheap checks that prevent building on an unexportable design.

---

## Risks, stated up front

1. **The linear ceiling is low.** The *full* optimal linear frequency filter is
   worth only +0.68 dB (dehaze) / +0.87 dB (derain). A spatial approximation
   cannot exceed that. Gains beyond it must come from nonlinearity and
   conditioning, not filtering.
2. **Reparameterization requires linearity.** A nonlinearity between branches
   breaks the merge and the NPU advantage evaporates. S3.2 is the guard.
3. **Scaling works against the deployment target.** w32 is ~4x w16's compute and
   our INT8 latencies were already seconds. Latency column mandatory.
4. **Capacity may not be the binding constraint.** 3.54 dB on derain is far more
   than scaling typically buys. S1.1's kill criterion exists to detect this
   early, and S1.3 is the fallback.
5. **Four architecture interventions have produced four nulls.** This may be a
   fifth. Kill criteria exist so that costs days, not a month — and S3.3's
   negative would itself be reportable.

---

## References

1. Ding et al. *RepVGG: Making VGG-style ConvNets Great Again.* CVPR 2021. arXiv:2101.03697
2. Ding et al. *Scaling Up Your Kernels to 31x31* (RepLKNet). CVPR 2022. arXiv:2203.06717
3. Rigamonti, Sironi, Lepetit & Fua. *Learning Separable Filters.* CVPR 2013; TPAMI 2015.
4. Freeman & Adelson. *The Design and Use of Steerable Filters.* TPAMI 13(9), 1991.
5. Cho & Hariharan. *On the Efficacy of Knowledge Distillation.* ICCV 2019. arXiv:1910.01348
6. Mirzadeh et al. *Improved Knowledge Distillation via Teacher Assistant.* AAAI 2020. arXiv:1902.03393
7. Li et al. *All-in-One Image Restoration for Unknown Corruption* (AirNet). CVPR 2022.
8. Potlapalli et al. *PromptIR.* NeurIPS 2023. arXiv:2306.13090
9. Luo et al. *DA-CLIP.* ICLR 2024.
10. Huang et al. *Degradation Frequency Curve.* 2026. arXiv:2605.17506
11. Cui et al. *AdaIR.* ICLR 2025. arXiv:2403.14614
12. Chen et al. *Simple Baselines for Image Restoration* (NAFNet). ECCV 2022. arXiv:2204.04676

**Our measurements this plan is built on:** `reports/kd_lit_review/review.md`,
`reports/dfc/`, `reports/freq_spatial_review/`, `reports/student_v3/`,
`teacher-experiments/test18/`, `scripts/freq_to_spatial.py`, `scripts/dfc_geom.py`.
