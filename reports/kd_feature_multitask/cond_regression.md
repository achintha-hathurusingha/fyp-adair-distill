# B0V2-KD-FEAT-COND stopped — DegradationHead/FiLM conditioning made every task worse

## What was tested

The kd_feature_multitask plan's treatment arm (`reports/kd_feature_multitask/plan.md`,
section 3): an auxiliary DegradationHead (GAP -> linear, 256->3) predicting which
degradation the student is looking at from its own `middle_blks` output, trained
with a cross-entropy loss against `_provenance["task"]` (`aux_weight=0.1`,
scale-checked), then FiLM-modulating (`x*(1+scale)+shift`) that same `middle_blks`
tensor with its own prediction. Everything else — architecture, data mix,
schedule, the feature-KD term on `latent_pre` — identical to the control arm
(`B0V2-KD-FEAT`).

The hypothesis (literature-motivated, see plan.md section 2): naive multi-task
distillation risks catastrophic interference; explicit degradation-conditioning
should prevent it.

## Result: not just "no help" — a consistent, growing regression

Stopped at iteration 69,000/90,000 (77%) once the pattern was unambiguous across
five separate checkpoints. At every one of them, on every one of the three
tasks, the treatment arm underperforms the control:

| Iter | Combined Δ | Denoise Δ | Derain Δ | Dehaze Δ |
|---|---:|---:|---:|---:|
| 15,000 | −0.385 | −0.175 | −0.172 | −0.808 |
| 30,000 | −0.474 | −0.160 | −0.398 | −0.863 |
| 45,000 | −0.615 | −0.163 | −0.486 | −1.198 |
| 60,000 | −0.635 | −0.152 | −0.437 | −1.316 |
| 69,000 | −0.598 | −0.146 | −0.429 | −1.220 |

(Δ = treatment − control, dB. Full history in
`runs/b0v2_kd_feat_cond/B0V2-KD-FEAT-COND/B0V2-KD-FEAT-COND_seed0_20260828_200459/history.json`,
kept on devon for the record — the run directory was not deleted.)

Not a divergence: `clampeng`/`deepeng` stayed at 0% throughout, `aux_last`
converged to ~0 (the classifier itself works fine, >99.9% confident by iter
15,000), grad norms stayed small. This is a clean, stable training run that
simply converges to a worse solution — and the gap *widens* over training
rather than closing, concentrated hardest on dehaze (the task feature-KD
normally helps most).

## Diagnosis

FiLM modulates `middle_blks`'s output — the exact same tensor the feature-KD
loss reads (via the adapter) to compare against the teacher's `latent_pre`.
The auxiliary conditioning signal isn't sitting beside the feature-KD
objective; it's directly perturbing the tensor that objective is defined on.
Every step, the optimizer has to reconcile "make this tensor look like
`latent_pre`" against "warp this tensor by a classifier-derived scale/shift" —
two objectives fighting over the same intermediate representation, which is
consistent with why dehaze (feature-KD's strongest task) is hit hardest.

## Decision

Treatment arm stopped (iter 69,000, healthy, no crash — a stop-on-evidence
decision, not a technical failure). Data retained on devon rather than
deleted — 69,000 iterations of real control-vs-treatment comparison is worth
keeping for the record even though this design isn't being pursued further.
Control arm (`B0V2-KD-FEAT`) continues alone.

## Literature review — what actually works, and why it avoids this failure mode

The common thread across four representative approaches: **none of them
condition at the single point where another loss already reads the tensor.**

- **PromptIR** (Potlapalli et al., 2023) injects degradation-conditioned
  prompts only on the **decoder** side, at multiple levels — never at the
  encoder/latent bottleneck alone. Their own ablation quantifies exactly the
  failure mode hit here: latent-space-only injection scores **36.76dB** on
  Rain100L, while spreading the same mechanism across decoder levels 4+3+2
  reaches **37.04dB** — "using only one prompt block in the latent space
  degrades the network's performance" is stated directly in their ablation
  section. [PromptIR: Prompting for All-in-One Blind Image Restoration](https://arxiv.org/html/2306.13090v1)
- **AirNet** (Li et al., CVPR 2022) extracts its degradation representation
  with a **contrastive** loss (positive pairs = same degradation, negative =
  different), not cross-entropy classification — the representation is
  shaped by a self-supervised similarity objective rather than a supervised
  label, which sidesteps building a classifier that has its own gradient
  claim on the shared feature. Costs a second training stage. [Neural Degradation Representation Learning](https://arxiv.org/pdf/2310.12848), [All-In-One Image Restoration for Unknown Corruption](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_All-in-One_Image_Restoration_for_Unknown_Corruption_CVPR_2022_paper.pdf)
- **HAIR** conditions via a **hyper-selection network** that generates
  parameters dynamically from a degradation-awareness classifier, rather than
  a static learned affine (FiLM) applied to one fixed tensor.
- **R2R — Retrieve-to-Restore** (Wang, Zhang, Yang; **CVPR 2026**) is the
  closest match to this project's own mobile-efficiency goal: SOTA-comparable
  PSNR at **~91% fewer MACs**. It stores degradation-specific "clean priors"
  in a compact bank, built during training by a degradation *amalgamator*
  (aggregates GT-guided intra-class features, trained with `L_pixel + L_fft +
  λ_d·L_deg + λ_m·L_match` — yes, it also uses classification-style
  cross-entropy terms, `L_deg`/`L_match`). The injection is still
  single-point (deepest encoder stage) — but the retrieved prior is fused via
  a **gated convolution**, not raw FiLM: "each sample interacts only with its
  retrieved prior," meaning the network can learn to suppress the
  conditioning contribution when it isn't helpful, rather than always
  applying an unconditional affine warp. Critically, the amalgamator — all
  the machinery that builds the bank — is **removed after training**; only a
  lightweight retrieval + gated fusion survives to inference, the same
  train-time-only discipline this project already applies to `FeatureAdapter`.
  [Retrieve-to-Restore, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Retrieve-to-Restore_Efficient_All-in-One_Image_Restoration_with_a_Retrieval-Based_Degradation_Bank_CVPR_2026_paper.html)

## Recommended next step, in order of how much it changes

1. **Cheapest, most diagnostic: move the injection point, change nothing
   else.** Keep the DegradationHead and its cross-entropy loss exactly as
   built; move the FiLM modulation off `middle_blks` and onto the decoder
   stages instead (matching PromptIR's own ablation-validated placement).
   This isolates whether "injection point coincides with feature-KD's tap
   point" was really the cause — if moving it alone fixes the regression,
   that confirms the diagnosis above without touching anything else.
2. **Moderate: keep the injection point, change the fusion mechanism.**
   Replace `x*(1+scale)+shift` with a gated fusion (`x + gate * transform(x,
   cond)`, gate initialized near zero) so the network can learn to ignore the
   conditioning signal early in training, when the classifier is least
   reliable, rather than being forced to apply it from iteration 0.
3. **Larger escalation: R2R-style retrieval.** Only worth it if (1) and (2)
   both fail to help — a genuinely different mechanism (retrieval against a
   trained bank of GT-derived priors) rather than a live per-sample
   classifier, with the added benefit of directly matching this project's own
   efficiency constraint (its 91% MACs reduction is a real, published number
   at a comparable PSNR bar).

(1) is the natural next experiment: smallest diff from what's already built
and smoke-tested, and it directly tests the specific hypothesis this
regression points at.
