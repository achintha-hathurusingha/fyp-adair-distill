# Wider student + spatial frequency mining — staged research plan

**The idea.** A wider/deeper student; a richer degradation representation taken
after the decoder (replacing PCA-16); and a few AdaIR-style frequency-mining
blocks implemented as *learnable spatial kernels* rather than FFT.

**Why now.** Four things we measured converge on it:

| our measurement | what it licenses |
|---|---|
| KD effect tracks the teacher-student gap (r = -0.987 / -0.9999) | a **bigger student** shrinks the gap, so KD should flip from harmful to helpful |
| B0V3-KD-FEAT beats both parents on all 3 tasks at 66k | the architecture x distillation interaction is real |
| 7x7-11x11 kernels reproduce the full optimal frequency filter | frequency mining **can** be done spatially |
| separable oriented bands 98.3% vs radial 97.5% vs square 93.3% (matched dims) | the band geometry should be **separable and oriented**, not radial |

**The constraint that shapes everything.** `torch.fft` does not export to ONNX;
attention is UNKNOWN on all three NPU backends. Any frequency behaviour must be
spatial *at inference*.

---

## The literature that makes this buildable

### Structural reparameterization — the key enabler

**RepVGG** (Ding et al., CVPR 2021) and **RepLKNet** (Ding et al., CVPR 2022)
train a rich multi-branch block and then **algebraically merge every branch into
a single convolution** for deployment. RepLKNet specifically trains parallel
*large and small* kernels and merges the small into the large afterwards.

> **This is the answer to our NPU problem.** A frequency-mining block built as
> parallel oriented/multi-scale spatial kernels combined *linearly* collapses at
> deploy time into **one plain conv**. Training-time richness, inference-time
> cost of a single convolution, and nothing for the export gate to reject.
> The linearity requirement is the design constraint: merging is only exact if
> the branches combine linearly before any nonlinearity.

### Separable filters — why the geometry finding is not a fluke

**Rigamonti, Sironi & Lepetit** (*Learning Separable Filters*, CVPR 2013 /
TPAMI 2015) show learned filter banks "can be approximated very precisely as
linear combinations of far fewer separable filters." Anisotropic Gaussians are
exactly separable along their principal axis and its perpendicular. Recent work
uses **directionally-sensitive large-kernel decomposition for anisotropic
degradations with fewer parameters**.

> Independent support for our own measured result that a *separable, oriented*
> band decomposition beats both radial and square.

### Capacity and distillation

Cho & Hariharan (ICCV 2019): increasing teacher capacity does not increase
student accuracy; under insufficient capacity students underfit the teacher's
mapping. The corollary matters more for us: **when the student is large, KD can
bring it close to the teacher.** A useful steer from the same literature —
for WideResNet, *increasing depth beat increasing width* per parameter.

### Degradation representation

AirNet (contrastive), PromptIR (prompts), DA-CLIP (decoupling degradation from
content), DFC (explicit spectral quantification). The field has moved from
implicit to **explicit, measurable** degradation coordinates — which is the
argument for replacing PCA-16 with something richer.

---

## Staged plan

Every stage has a **kill criterion**. The 0.035 dB seed-noise floor is the
yardstick: anything under ~0.07 dB is not a result.

Status: `[ ]` not started · `[~]` running · `[x]` done · `[!]` killed

---

### PHASE 0 — Establish ceilings before building anything (no GPU)

**S0.1 · Oracle ceiling for separable oriented kernels** `[ ]`
Extend `scripts/freq_to_spatial.py`: instead of one isotropic optimal filter,
fit a *separable oriented* kernel bank per degradation and measure held-out PSNR.
- **Deliver:** ceiling in dB per task for the best possible spatial oriented filter.
- **Kill:** if the oriented ceiling is < +0.15 dB over the isotropic one, the
  orientation machinery is not worth building — drop to a plain large kernel.
- **Cost:** CPU, ~1h. **Depends:** nothing.

**S0.2 · NPU export gate for the proposed ops** `[ ]`
Export a stub block (parallel oriented convs → linear combine → merged conv)
through `src/export/op_coverage.py` *before* any training.
- **Deliver:** op table; confirmation the merged form is a single Conv.
- **Kill:** any UNKNOWN op that survives reparameterization.
- **Cost:** minutes. **Depends:** nothing. **Do this first.**

**S0.3 · How much does PCA-16 actually throw away?** `[ ]`
Probe degradation-ID accuracy from decoder features at increasing dimension
(16 / 32 / 64 / 128 / full), same classifier, leave-scene-out.
- **Deliver:** accuracy-vs-dimension curve.
- **Kill:** if 16 dims already saturates, "wider feature set" is unjustified —
  say so and keep PCA-16.
- **Cost:** CPU, ~2h. **Depends:** nothing.

---

### PHASE 1 — Bigger student (tests the capacity-gap prediction)

**S1.1 · Width/depth scaling sweep, GT-only** `[ ]`
Three sizes beyond current 7.4M — e.g. w24, w32, and a deeper w16 — GT-only,
90k, matched schedule. Literature says prefer **depth over width** per parameter.
- **Deliver:** PSNR and per-task teacher gap vs parameter count.
- **Success:** monotone improvement, and the teacher gap visibly shrinking.
- **Kill:** if PSNR saturates before the gap closes, capacity is not the binding
  constraint — go straight to Phase 3.
- **Cost:** 3 x ~9h GPU. **Depends:** S0.3 (informs where to widen).

**S1.2 · Does KD flip sign as the student grows?** `[ ]`
Re-run KD at each size from S1.1. **This is the sharpest prediction the whole
plan makes:** KD harm should shrink monotonically with student capacity, and
derain (largest gap) should be the last to flip.
- **Deliver:** KD-effect-vs-capacity curve, per task.
- **Success:** the sign flip is observed, or the harm shrinks monotonically.
- **Kill:** if KD harm is flat across sizes, the capacity-gap account is wrong
  and the lit review needs correcting.
- **Cost:** 3 x ~9h GPU. **Depends:** S1.1.

---

### PHASE 2 — Richer degradation representation

**S2.1 · DFC-style representation from decoder features** `[ ]`
Compute band-wise residual-to-degraded ratios on *decoder features* using the
**separable oriented** bands S0.1 validated. Residual estimated as
degraded-minus-intermediate-restoration (DFC's trick — no GT needed at inference).
- **Deliver:** representation + its degradation-ID accuracy vs PCA-16's.
- **Kill:** < +2pp over PCA-16 at matched dimension.
- **Cost:** CPU, ~3h. **Depends:** S0.1, S0.3.

**S2.2 · Conditioning that does not fight the KD loss** `[ ]`
Our v1 FiLM regressed because it modulated the exact tensor feature-KD reads.
Condition on a **different** tensor (decoder-side), with a matched control.
- **Deliver:** conditioned arm vs control at 90k.
- **Kill:** any regression vs control > 0.07 dB.
- **Cost:** 2 x ~9h GPU. **Depends:** S2.1, and S1.2 for the size to use.

---

### PHASE 3 — Spatial frequency-mining block

**S3.1 · Build the block, reparameterizable by construction** `[ ]`
Parallel separable oriented kernels (0/45/90/135 + isotropic), learnable
per-band coefficients, combined **linearly** so the branches can merge.
- **Deliver:** module + unit test that operators-off is byte-identical to the
  baseline (the no-confound check we used for v3).
- **Cost:** ~2h dev. **Depends:** S0.1, S0.2.

**S3.2 · Prove the merge is exact** `[ ]`
Merge the multi-branch block into one conv; assert max |diff| < 1e-5 on random
input, then export and confirm the graph contains a single Conv.
- **Deliver:** equivalence test + ONNX op table.
- **Kill:** if the merge is not exact, the block must be redesigned — a
  non-linear branch has crept in.
- **Cost:** ~1h. **Depends:** S3.1. **Gate: do not train until this passes.**

**S3.3 · Train with the block, ablate placement** `[ ]`
Where do the blocks go? Test middle-only vs decoder-only vs both, one variable
at a time.
- **Success:** > +0.07 dB over the matched no-block control.
- **Kill:** three placements all null → the spatial frequency route is closed,
  and that is a publishable negative given the convolution-theorem ceiling.
- **Cost:** 3 x ~9h GPU. **Depends:** S3.2, S1.1.

---

### PHASE 4 — Integration and honesty checks

**S4.1 · Full model** `[ ]` — best size + best representation + best block
placement, with KD. **Depends:** S1.2, S2.2, S3.3.

**S4.2 · Multi-seed** `[ ]` — 3 seeds on the final arm and its control. Every
claim so far is single-seed; this is what makes the result reportable.

**S4.3 · Composite / unseen degradations** `[ ]` — DFC's biggest wins were on
composite (+1.17 dB) and unseen (+0.59), not the standard 3-task benchmark
(+0.12). **Our entire evaluation is one-degradation-at-a-time, which is the
regime where this family of methods helps least.** Build a mixed-degradation
test set and evaluate there.

**S4.4 · Final NPU measurement** `[ ]` — real Qualcomm AI Hub latency, FP32 and
INT8, on the merged deployment graph.

---

## Ordering

```
S0.2 (gate, minutes) ─┐
S0.1 ─────────────────┼─> S3.1 ─> S3.2 (gate) ─> S3.3 ─┐
S0.3 ─> S2.1 ─> S2.2 ─┘                                 ├─> S4.1 ─> S4.2 ─> S4.3 ─> S4.4
S1.1 ─> S1.2 ──────────────────────────────────────────┘
```

Phase 0 is all CPU and can run while GPU work continues. **S0.2 and S3.2 are
gates** — cheap checks that prevent building on an unexportable design.

---

## Risks, stated up front

1. **The linear ceiling is low.** We measured the *full* optimal linear
   frequency filter as worth only +0.68 dB (dehaze) / +0.87 dB (derain). A
   spatial approximation cannot exceed that. Any gain beyond it must come from
   the nonlinearity and conditioning, not the filtering.
2. **Reparameterization requires linearity.** The moment a nonlinearity sits
   between branches, the merge fails and the NPU advantage evaporates. S3.2 is
   the guard.
3. **Bigger student, worse deployment.** Phase 1 works directly against the
   mobile target. Every size must carry its NPU latency number, not just PSNR.
4. **Four architecture interventions have already produced four nulls.** This
   plan may produce a fifth. The kill criteria exist so that costs a few days,
   not a month — and S3.3's negative would itself be reportable.

---

## References

1. Ding et al. *RepVGG: Making VGG-style ConvNets Great Again.* CVPR 2021. arXiv:2101.03697
2. Ding et al. *Scaling Up Your Kernels to 31x31: Revisiting Large Kernel Design in CNNs* (RepLKNet). CVPR 2022. arXiv:2203.06717
3. Rigamonti, Sironi, Lepetit & Fua. *Learning Separable Filters.* CVPR 2013; TPAMI 2015.
4. Freeman & Adelson. *The Design and Use of Steerable Filters.* TPAMI 13(9), 1991.
5. Cho & Hariharan. *On the Efficacy of Knowledge Distillation.* ICCV 2019. arXiv:1910.01348
6. Mirzadeh et al. *Improved Knowledge Distillation via Teacher Assistant.* AAAI 2020. arXiv:1902.03393
7. Li et al. *All-in-One Image Restoration for Unknown Corruption* (AirNet). CVPR 2022.
8. Potlapalli et al. *PromptIR: Prompting for All-in-One Blind Image Restoration.* NeurIPS 2023. arXiv:2306.13090
9. Luo et al. *Controlling Vision-Language Models for Universal Image Restoration* (DA-CLIP). ICLR 2024.
10. Huang et al. *Degradation Frequency Curve.* 2026. arXiv:2605.17506
11. Cui et al. *AdaIR: Adaptive All-in-One Image Restoration via Frequency Mining and Modulation.* ICLR 2025. arXiv:2403.14614
12. Chen et al. *Simple Baselines for Image Restoration* (NAFNet). ECCV 2022. arXiv:2204.04676

**Our own measurements this plan is built on:** `reports/kd_lit_review/review.md`,
`reports/dfc/`, `reports/freq_spatial_review/`, `reports/student_v3/`,
`scripts/freq_to_spatial.py`, `scripts/dfc_geom.py`.
