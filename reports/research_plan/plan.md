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

- The derain gap is **3.54 dB**. Scaling w16 → w32 typically buys ~0.3-0.5 dB
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
| B0V3-KD-FEAT beats both parents on all 3 tasks (at 75k: +0.159 vs KD arm, +0.265 vs GT-only v3) | the architecture x distillation interaction is real; **KD is currently worth 0.265 dB** and should not be dropped on principle |
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

**S0.1 · Oracle ceiling for separable oriented kernels** `[ ]`
Extend `scripts/freq_to_spatial.py`: fit a *separable oriented* kernel bank per
degradation, measure held-out PSNR.
- **Kill:** oriented ceiling < +0.15 dB over isotropic → drop the orientation
  machinery, use a plain large kernel.
- CPU ~1h. Depends: nothing.

**S0.2 · NPU export gate for the proposed ops** `[ ]` **← DO THIS FIRST**
Export a stub block (parallel oriented convs → linear combine → merged conv)
through `src/export/op_coverage.py` *before* any training.
- **Kill:** any UNKNOWN op surviving reparameterization.
- Minutes. Depends: nothing.

**S0.3 · How much does PCA-16 throw away?** `[ ]`
Probe degradation-ID accuracy from decoder features at 16/32/64/128/full dims,
leave-scene-out.
- **Kill:** 16 dims already saturates → "wider feature set" is unjustified; say
  so and keep PCA-16.
- CPU ~2h. Depends: nothing.

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
Band-wise residual-to-degraded ratios on *decoder features*, using the
**separable oriented** bands S0.1 validated. Residual estimated as
degraded-minus-intermediate-restoration (DFC's trick — no GT at inference).
- **Kill:** < +2pp over PCA-16 at matched dimension.
- CPU ~3h. Depends: S0.1, S0.3.

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
- ~2h dev. Depends: S0.1, S0.2.

**S3.2 · Prove the merge is exact** `[ ]` **← GATE**
Merge multi-branch → one conv; assert max |diff| < 1e-5 on random input; export
and confirm a single Conv in the graph.
- **Kill:** merge not exact → a nonlinearity has crept in; redesign.
- ~1h. Depends: S3.1. **Do not train until this passes.**

**S3.3 · Train with the block, ablate placement** `[ ]`
Middle-only vs decoder-only vs both, one variable at a time.
- **Success:** > +0.07 dB over the matched no-block control.
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
