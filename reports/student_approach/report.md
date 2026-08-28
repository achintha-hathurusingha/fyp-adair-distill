# Student Model — Current Distillation Approach

*AdaIR-teacher to NAFNet-student distillation, mobile-restoration FYP*  
*2026-08-28*


## 1. Where this sits


This project distills the released AdaIR restoration teacher (28.78M params, all-in-one 3-degradation model) into a small NAFNet-based student (7.37M params, W16 SIDD geometry) sized for on-device mobile deployment. This report is a snapshot of the student's architecture and the distillation recipe as they stand today, at the end of the kd_feature_multitask phase's build-out (reports/kd_feature_multitask/plan.md).


## 2. Locked student architecture


The architecture itself has been locked since the B0 baseline: a 4-stage NAFNet U-Net, width 16, encoder block counts [2, 2, 4, 8], 12 middle blocks, symmetric decoder [2, 2, 2, 2]. 7.37M parameters, 4.13 GMACs, 2.885 ms measured INT8 latency on a Galaxy S24. Normalization is LayerNorm2d throughout, with an affine_clamp(8.0) at full resolution (finding F9, guards against the divergence that plain affine norm alone did not catch) and a deep-stage clamp insurance bound of 32.0 at the enc3 stage (finding F10).


![Student architecture: channel/block counts per stage, normalization, and the opt-in DegradationHead+FiLM insertion point.](figures/student_architecture.png)
*Student architecture: channel/block counts per stage, normalization, and the opt-in DegradationHead+FiLM insertion point.*


Two attention/norm variants were tried against this locked baseline and NOT adopted, both documented in reports/student_arch/: ECA (channel attention swapped in for SCA) measured a genuine, non-noise +0.14dB mean improvement with 13% fewer params across 16/19 matched checkpoints, but training was stopped on explicit instruction rather than pursued further. GroupNorm (in place of LayerNorm2d) was found to be genuinely unstable — pre-clamp activation magnitude exploded from 65 to 5688 over 10,000 iterations with clamp engagement rising from 1.6% to 4.3% — and was stopped as a real divergence risk, not a false alarm.


## 3. Distillation recipe: how it got here


The distillation loss was built up one term at a time, each isolated against the previous stage on a dehaze-only demo protocol before being trusted:


- GT-only (Charbonnier vs ground truth, no teacher): 32.890dB, 3-seed mean — the baseline everything else is measured against.
- + response KD (Charbonnier vs the frozen teacher's own output, weight 1.0): 33.076dB, 3-seed mean — a modest, consistent gain.
- + frequency-domain spectrum term (kd_freq, on top of response KD): 33.190dB single seed, inside the noise band already observed mid-training. A literature review gave a mechanistic reason not to expect more: by Parseval's theorem, a magnitude-mode spectral loss is close to redundant with the spatial Charbonnier term already present. Not pursued past seed 0.
- + feature-level term on the teacher's internal latent_pre bottleneck (kd_feat, replacing the frequency term): 33.695dB, full 60,000 iterations, the strongest and most-validated result so far — +0.805dB over GT-only, +0.619dB over response-KD alone. Motivated by TEST05.5 (teacher-experiments), a causal audit which found latent_pre, not the frequency pathway, is the well-supported distillation signal.


![Established PSNR results, dehaze-only ablation ladder.](figures/results_so_far.png)
*Established PSNR results, dehaze-only ablation ladder.*


![Current recipe: teacher forward (frozen) produces latent_pre + response; student forward produces pred + middle_blks capture (+ degradation logits, opt-in); four loss terms combine additively.](figures/distillation_pipeline.png)
*Current recipe: teacher forward (frozen) produces latent_pre + response; student forward produces pred + middle_blks capture (+ degradation logits, opt-in); four loss terms combine additively.*


The feature term needs a bridge between the teacher's and student's internal representations — latent_pre is 384 channels at 1/8 resolution (AdaIR, dim=48, 3 downsamples), while the student's middle_blks output is 256 channels at 1/16 resolution (width=16, 4 downsamples). A small FeatureAdapter (1x1 conv + bilinear upsample, ~99K params) bridges the two. It is trained jointly with the student but is training-time only — never part of the exported graph, the same export-safety discipline already established for the (abandoned) frequency term.


## 4. Current phase: from single-task to all-in-one (kd_feature_multitask)


kd_feat's 33.695dB result was only ever validated single-task (dehaze-only). The protocol this project actually targets — matching AirNet/PromptIR/AdaIR itself — is one student handling all three degradations (denoising, deraining, dehazing) at once (the B0V2 arm). Naively pointing kd_feat's existing loss at the full 3-degradation data stream is not assumed safe: a literature review found that naive multi-task distillation risks catastrophic interference in a small student, and that the documented fix is degradation-conditioning or routing, not raw capacity. This project's own TEST19 result gives a concrete reason to expect conditioning helps here specifically: the teacher's own latent_pre representation separates degradation type with 99.0% leave-scene-out accuracy — a clean signal a naive loss extension never uses.


![DegradationHead + FiLM: a tiny auxiliary classifier and conditioning mechanism, added as an opt-in extension.](figures/degradation_head_detail.png)
*DegradationHead + FiLM: a tiny auxiliary classifier and conditioning mechanism, added as an opt-in extension.*


The mechanism: a small auxiliary head (global-average-pool -> linear, 256->3) predicts which degradation the student is looking at, purely from its own middle_blks features — no teacher involved. Its own softmax prediction then FiLM-modulates (scale-and-shift) those same features. Ground truth for the auxiliary cross-entropy loss is `_provenance["task"]`, a field that was already flowing through the multi-task data loader completely unused — zero new data-pipeline work. The mechanism is deliberately deployment-safe: at inference the student conditions on its OWN guess, with no teacher present and only plain, exportable ops (a linear layer and an elementwise scale/shift).


Two arms isolate exactly one variable: B0V2-KD-FEAT (control) is the naive extension — kd_feat's existing loss, full 3-task data, no conditioning. B0V2-KD-FEAT-COND (treatment) is identical plus the DegradationHead/FiLM addition. Both use the all_in_one teacher checkpoint (adair3d.ckpt) rather than a single-degradation specialist — a specialist teacher has nothing sensible to say about the two degradations it wasn't trained on, an issue found and fixed while building these arms. A second, independent gap was also found and fixed in the process: B0V2's own periodic validation defaulted to denoise-only, so the existing completed B0V2 baseline (300k iterations) has real denoise numbers but was never evaluated on dehaze or derain at all. Both new arms now validate on all three held-out sets every checkpoint.


aux_weight was scale-checked against real data and the real architecture before being fixed, the same discipline kd_feat's own feat_weight=0.01 used: at weight 1.0 the raw auxiliary cross-entropy sat near ln(3)=1.10 (chance, expected early in training) against the other loss terms' 0.02-0.15 range — a ~10x mismatch. aux_weight=0.1 brings its weighted contribution into that same range.


## 5. Status


- DegradationHead module: built, smoke-tested in isolation (shapes, gradient flow, error handling) — committed.
- Wired into NAFNet + build_nafnet behind an opt-in flag, default off (every existing arm stays byte-identical): smoke-tested end-to-end on CPU — committed.
- Auxiliary loss wired into the trainer, reading the already-present provenance field: smoke-tested with real optimizer steps on real multi-task data — committed.
- B0V2 eval-gap fixed: multi-task validation now covers all three tasks, smoke-tested against the real held-out sets — committed.
- all_in_one teacher checkpoint wired in and verified (0 missing/0 unexpected keys, correct latent_pre shape on all 3 degradations) — committed.
- Both arms registered, aux_weight scale-checked, both smoke-tested end-to-end via the real training CLI — committed.
- B0V2-KD-FEAT (control) launched on devon, 90,000 iterations. B0V2-KD-FEAT-COND (treatment) staged to launch once the control clears an early checkpoint healthy, the same staged-launch protocol B0-denoise itself used.


## 6. What would settle the open question


Once both arms finish (or reach a comparable, matched iteration count), the comparison that matters is per-task PSNR — psnr_denoise / psnr_derain / psnr_dehaze — between control and treatment, not just the combined mean. The literature's prediction is specifically about interference concentrated on whichever tasks compete hardest for capacity, which a single averaged number can hide. If conditioning measurably helps, the natural escalation is replacing the ground-truth-label target with the teacher's own continuous degradation code (TEST19's PCA-16 basis) — this phase deliberately started with the simpler, cheaper version first.
