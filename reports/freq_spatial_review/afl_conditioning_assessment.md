# Assessment: "Degradation-Aware Restoration via Adaptive Frequency Conditioning"

Deep research on the proposal that the **AFLB should learn the degradation**
while the **encoder–decoder learns only conditional restoration**.

**Verdict up front:** the reasoning is sound and the experimental design is
better than what we have been doing — but the core mechanism is well-covered
prior work (2022–2026), and **the proposal's central premise is contradicted by
measurements already in this project**. One component of it is genuinely
valuable and untested. Details below, with sources.

Tags: **[lit]** = published work · **[ours]** = measured in this project.

---

## 1. Novelty: the mechanism is prior work

The proposed architecture — degradation encoder → conditioning signal →
restoration network — is **exactly AirNet**.

| work | venue | what it does |
|---|---|---|
| **AirNet** ([Li et al.](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_All-in-One_Image_Restoration_for_Unknown_Corruption_CVPR_2022_paper.pdf)) | CVPR 2022 | Contrastive-Based Degraded Encoder (CBDE) + Degradation-Guided Restoration Network (DGRN). Learns degradation representation by contrastive learning, uses it to guide restoration. Blind — no corruption prior. |
| **PromptIR** ([arXiv:2306.13090](https://arxiv.org/abs/2306.13090)) | NeurIPS 2023 | Lightweight prompts encode degradation-specific info, dynamically guide the restoration net. |
| **PromptRestorer** | NeurIPS 2023 | Prompting with explicit degradation perception. |
| **DA-CLIP** ([arXiv:2310.01018](https://arxiv.org/pdf/2310.01018)) | ICLR 2024 | Vision-language control for multi-task restoration. |
| **DFC-IR** ([arXiv:2605.17506](https://arxiv.org/pdf/2605.17506)) | 2026 | **"Degradation Frequency Curve" — an explicit frequency-quantified degradation representation.** This is *very close* to "AFLB as the degradation encoder." |

That last row matters most: the specific idea of a **frequency-domain
degradation representation driving all-in-one restoration is already
published**. The proposal would need to differentiate against DFC-IR
explicitly.

**The proposal's own framing is also already the field's critique.**
*"Beyond Degradation Redundancy: Contrastive Prompt Learning"*
([arXiv:2504.09973](https://arxiv.org/pdf/2504.09973)) argues precisely that
AirNet/PromptIR degradation representations are **not sufficiently
discriminative — they overlap** — and proposes contrastive prompts to fix it.

---

## 2. The central premise is contradicted by our own data

The proposal states:

> *"without an adaptive degradation-learning module, an encoder–decoder can
> learn to restore images effectively only for a particular degradation
> distribution, but lacks an explicit mechanism to distinguish or adapt to
> different degradation types."*

**[ours] This is empirically false in this project, twice over.**

**(a) Blind encoder–decoders already learn degradation implicitly — to ~100%.**
`teacher-experiments/` TEST02 source-audited `AdaIR.forward()` and confirmed
**no degradation label is ever passed in**, then linearly probed intermediate
activations: representations were **up to ~100% degradation-discriminative**.
TEST03 rebuilt this with all three degradations synthesised from the *same*
clean scenes and scene-grouped CV, removing the dataset-identity confound —
discriminability **survived** (66.7% → 100% → 37.0% across stages).

So a fully blind network *does* learn "this is noise / rain / haze." It is
simply not labelled as such. The dichotomy the proposal draws — implicit
restoration vs. explicit degradation learning — does not hold: the implicit
path already contains the degradation representation.

**(b) Our own unconditioned multi-task students handle three degradations
fine.** B0V3 (no conditioning of any kind) reaches denoise 30.56 / derain 35.91
/ dehaze 33.67 simultaneously. It is not "a degradation-specific image-to-image
translator."

**(c) We already built the proposed architecture, and it regressed.**
`DegradationHead` (v1): auxiliary degradation classifier + FiLM conditioning —
structurally the proposal. It made **every task worse, with the gap widening**
(dehaze −1.22 dB by iteration 69 k). See
`reports/kd_feature_multitask/cond_regression.md`.

**Important caveat in the proposal's favour:** v1's failure had a *diagnosed,
specific* cause — it FiLM-modulated the `middle_blks` output, the exact tensor
the feature-KD loss also read, so the two objectives fought over one tensor.
**That conflict does not exist in a GT-only model.** v2
(`DecoderDegradationHead` — classify read-only, condition the *decoder*) was
built and smoke-tested but **never trained**. So the proposal is not refuted for
the GT-only case; it is *untested*.

---

## 3. What is genuinely valuable in the proposal

Two things, and they are the parts that are actually novel *for us*:

### 3a. Evaluating the degradation representation directly — **the right instinct**

> *"AFLB should be evaluated primarily on its ability to identify/encode
> degradation characteristics"*

**This is correct and under-done in the literature** — the redundancy paper
above exists precisely because representations were never properly audited.
And it is the one thing this project is unusually well-equipped to do: we have
built and validated exactly this apparatus tonight.

**[ours] We have already measured the ceiling this evaluation would report:**

| feature | 3-way degradation ID |
|---|---|
| FFT radial spectrum (same-scene, controls clean) | **93.6%** |
| NPU-safe Laplacian band energies | 88.0% |
| **small CNN trained on the label** | **97.0%** |

with the clean control at **exactly 33.3% chance**, so the signal is real and
not dataset identity (`scripts/spectral_samescene.py`,
`scripts/spatial_converter.py`).

**This is a problem for the proposal, not a support.** Degradation is ~97%
identifiable by a *small plain CNN*. If identification were the bottleneck,
the task would be nearly solved. It is not the bottleneck — the restoration
mapping is.

### 3b. Generalisation to unseen degradations — **the strongest part, and untested here**

> *"Test not only in-distribution degradation but ideally unseen or mixed
> degradation."*

This is the right experiment and **we have never run it**. It is also where the
field has moved: DAIR, GenDeg
([arXiv:2411.17687](https://arxiv.org/pdf/2411.17687), 550k synthetic samples
across six degradations), and DFC-IR all now evaluate OOD degradations
(underwater, desnowing, unseen levels).

It also reframes the question productively: conditioning may not improve
in-distribution PSNR (our v1 says it does not) while still improving
**adaptation** to unseen degradations. Those are different claims, and only the
first has been tested here.

---

## 4. Where this leaves the proposal

| claim | status |
|---|---|
| Architecture: degradation encoder → conditioning → restoration | **prior work** (AirNet '22, PromptIR '23) |
| Frequency-based degradation representation | **prior work** (DFC-IR '26) |
| Critique that representations are not discriminative | **prior work** (Beyond Degradation Redundancy '25) |
| "Encoder–decoder cannot distinguish degradations without AFLB" | **contradicted by our TEST02/03 (~100% implicit probe) and by B0V3** |
| Conditioning improves in-distribution PSNR | **tested here, regressed** (v1) — but confounded by feature-KD; **untested GT-only** (v2 built, never run) |
| Evaluate the degradation representation on its own | **good instinct; we measured the ceiling at 97% — so ID is not the bottleneck** |
| **Generalisation to unseen / mixed degradation** | **untested here, and the strongest remaining angle** |

### Recommendation

Do **not** run the proposal as stated — the mechanism is prior work and its
premise is contradicted by our own measurements.

**Do** run the one experiment it points at that we have never done, and which
costs almost nothing given v2 is already built:

> **Train B0V3 + v2 decoder conditioning (GT-only, so no feature-KD conflict),
> and evaluate on a HELD-OUT degradation type.** In-distribution PSNR is the
> weak test; the real question is whether an explicit degradation
> representation buys *adaptation* the implicit one does not.

This is cheap (arm registered, smoke-tested), it directly addresses the one
untested claim, and a null result is publishable in the same "reality-check"
frame as the rest of tonight's findings: *degradation conditioning does not
help even when its diagnosed failure mode is removed, because blind networks
already encode degradation implicitly at ~100%.*

**Sequencing note:** this must wait for the seed-variance result now running.
Every arm so far is single-seed, and the effects we have been interpreting are
~0.08 dB, while the first seed-pair checkpoint already differs by 0.21 dB. If
the noise floor exceeds the effect sizes, several conclusions above — including
v1's regression — need re-examination before building on any of them.

---

## References

1. Li, Liu, Yang, Peng & Zhou. *All-In-One Image Restoration for Unknown Corruption* (AirNet). CVPR 2022.
2. Potlapalli, Zamir, Khan & Khan. *PromptIR: Prompting for All-in-One Blind Image Restoration.* NeurIPS 2023. https://arxiv.org/abs/2306.13090
3. Wang et al. *PromptRestorer: A Prompting Image Restoration Method with Degradation Perception.* NeurIPS 2023.
4. Luo et al. *Controlling Vision-Language Models for Multi-Task Image Restoration* (DA-CLIP). ICLR 2024. https://arxiv.org/pdf/2310.01018
5. *Beyond Degradation Redundancy: Contrastive Prompt Learning for All-in-One Image Restoration.* https://arxiv.org/pdf/2504.09973
6. *Degradation Frequency Curve: An Explicit Frequency-Quantified Representation for All-in-One Image Restoration.* https://arxiv.org/pdf/2605.17506
7. *Degradation-Aware All-in-One Image Restoration via Latent Prior Encoding* (DAIR). https://arxiv.org/html/2509.17792
8. *GenDeg: Diffusion-based Degradation Synthesis for Generalizable All-In-One Image Restoration.* CVPR 2025. https://arxiv.org/pdf/2411.17687
9. Cui et al. *AdaIR.* ICLR 2025. https://arxiv.org/abs/2403.14614

**Our own evidence:** `teacher-experiments/test02`, `test03`,
`reports/kd_feature_multitask/cond_regression.md`,
`scripts/spectral_samescene.py`, `scripts/spatial_converter.py`,
`src/models/decoder_degradation_head.py` (v2, built, never trained).
