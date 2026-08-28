# Plan — degradation-conditioned feature-KD on the real 3-degradation protocol

## 1. Where this picks up

kd_feat (dehaze-only demo) finished its full 60,000 iterations cleanly:
**best PSNR 33.695dB**, seed 0. Against the established baselines (also
dehaze-only, 3-seed means): +0.805dB over GT-only (32.8898dB), +0.619dB over
response-KD (33.0759dB). Both deltas are far outside this project's own
measured seed-noise floor (~0.008dB, B0-denoise 3-seed). This is the
strongest result of anything tested here (kd_freq, ECA, GroupNorm all
either null or stopped on instability) — feature-KD from `latent_pre` is
the validated direction to carry forward, not a tentative one.

**But it has only ever been tested single-task.** The real target (per this
project's own protocol, AirNet/PromptIR/AdaIR-style) is one student handling
all three degradations at once. Nothing tested so far speaks to that.

## 2. Why not just add derain+denoise data to the existing kd_feat loss

Literature review (this session): naive multi-task distillation has a
documented failure mode — aggregating tasks into one small student
"introduces a large number of behavioral modes that can exceed the
student's capacity, forcing it to average across behaviors," i.e.
catastrophic interference, not just harder optimization. The documented fix
across multiple 2025 papers is routing/conditioning on degradation identity,
not raw capacity — Multi-Teacher KD routes each sample to a
degradation-specific signal; task-conditioned all-in-one restoration papers
build explicit degradation-conditioning into the bottleneck itself.

This project already has the evidence that makes conditioning the right
call here specifically: **TEST19 found `latent_pre`/`e_D` separates
degradation type with 99.0% leave-scene-out accuracy** — the teacher's own
representation already carries a clean, confident degradation signal.
Right now kd_feat only uses that as a training-time *loss target* (student
learns to produce similar features, no explicit awareness of which
degradation it's looking at). The literature says that version is exactly
what's exposed to interference once denoise/derain get added.

## 3. Design — auxiliary degradation head + FiLM conditioning

**Deployment constraint, stated first because it drives the design**: the
conditioning signal must be available to the student at *inference*, with
no teacher present (matches every other design choice in this project —
F7's export-gate finding, the whole reason feature-KD targets `latent_pre`
as a training-time loss rather than an architectural dependency). So the
conditioning signal cannot be "the teacher's e_D, computed live" — it has
to be something the student produces itself.

**Mechanism** (small additions to the existing NAFNet + trainer, not a
rebuild):

1. **Auxiliary degradation-classification head**: a tiny linear layer off
   `middle_blks`'s pooled output (256 -> 3, softmax over
   {denoise, derain, dehaze}).
2. **Ground truth for it is already flowing through the pipeline**: the
   multi-task loader's third yielded element, `_provenance["task"]`
   (`trainer.py:528`), currently unused. Zero new data-pipeline work.
3. **Auxiliary loss**: cross-entropy between the head's prediction and
   `_provenance["task"]`, added to the existing loss (existing feature-KD
   term on `latent_pre` stays unchanged — this is additive, not a
   replacement).
4. **FiLM conditioning**: the auxiliary head's own pre-softmax features (or
   a small embedding derived from them) modulate `middle_blks` via a
   learned scale+shift (`x = x * (1 + scale) + shift`), the standard cheap
   conditioning mechanism. At inference, the student computes its own
   degradation guess from its own features and conditions on that — no
   teacher, no extra runtime dependency, and it stays a plain-conv graph
   (a linear layer + elementwise scale/shift export cleanly on any NPU
   toolchain already validated in this project).

**This is genuinely testing the interference hypothesis, not assuming
it**: ground-truth-label conditioning is simpler and cheaper than
distilling the teacher's continuous e_D code, and it isolates the actual
variable in question (does *explicit degradation-awareness*, however
provided, prevent interference) from the separate question of whether the
teacher's specific code is worth reproducing (kd_feat's existing loss
already covers that). If this simpler version doesn't help, the
teacher-e_D-distilled version (using TEST19's fixed PCA-16 basis instead of
ground-truth labels) is the natural escalation — not the starting point.

## 4. Fix the eval gap before trusting any B0V2 number

Found earlier: B0V2's own periodic validation defaults to denoise-only
(`trainer.py`'s `self.val_task` defaults to `"denoise"`, and
`b0v2_multitask.yaml`'s `eval:` block never overrides it). The existing
completed B0V2 baseline run (300k iters, `runs/b0v2/B0V2/B0V2_seed0_20260803_210918`)
has real denoise numbers (30.685dB combined) but **no dehaze or derain
numbers at all** — it was never evaluated on them. Both new arms below need
periodic (or at minimum final) evaluation on all three test sets
(`dehaze_demo`, `derain_demo`, `bsd68`), not just denoise, or "does
conditioning help" can't actually be answered.

## 5. Staged experiment — two arms, isolating one variable

Both arms: full 3-degradation training data (`b0v2_multitask.yaml`'s own
locked architecture, data mix, schedule — unchanged), same feature-KD loss
on `latent_pre` (kd_feat's validated term, unchanged). The *only* difference
is conditioning:

- **B0V2-KD-FEAT (control)**: plain extension — kd_feat's existing loss,
  full 3-task data, no auxiliary head, no FiLM. This is the naive version
  the literature predicts may show interference. Worth running even though
  it might underperform — it's the honest baseline the conditioned version
  needs to beat to justify the added complexity.
- **B0V2-KD-FEAT-COND (treatment)**: same as above + the auxiliary
  degradation-classification head + FiLM conditioning from Section 3.

Smoke-test each (`--smoke N`) before committing to a real run, matching
every other experiment in this project. Given B0V2's own full run took
8.2 hours for 300k iterations, and this is genuinely a bigger commitment
than the dehaze demos, a shorter iteration budget (matching the *ablation*
scale AdaIR's own paper uses, not the full 150k+ main-training scale) is
worth considering for the first pass — stated as an open question, not
decided here, since it trades real wall-clock cost against how far into
convergence "does conditioning help" needs to be measured to trust the
answer.

## 6. Build order

1. Auxiliary head + FiLM module (`src/models/degradation_head.py` or
   similar) — small, CPU-smoke-testable in isolation before touching the
   trainer, matching kd_feature's own Steps 1-2 discipline.
2. Wire into `NAFNet`/`build_nafnet` behind a new opt-in config flag
   (default off, so every existing arm stays byte-identical — same
   discipline as `attn_type`/`feat_weight`).
3. Wire the auxiliary loss into `trainer.py`, reading `_provenance["task"]`
   (already available, no data-pipeline change).
4. Fix B0V2's eval config to cover all three tasks.
5. `--smoke` both arms, verify auxiliary loss decreases and FiLM
   parameters actually receive gradient (a real risk worth checking
   directly, not assuming — an auxiliary head with no gradient flowing to
   the FiLM scale/shift would silently do nothing).
6. Launch both arms, staged (control first, or both concurrently if GPU
   headroom allows — decide at launch time based on actual devon/qbits load
   then, not pre-committed here).
