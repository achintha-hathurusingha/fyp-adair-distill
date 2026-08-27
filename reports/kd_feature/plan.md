# Plan — feature-level distillation on latent_pre, replacing output-spectrum KD

## Why this, and why it's a different scope than kd_freq

kd_freq (`reports/kd_freq/plan.md`) tests a fixed FFT-magnitude loss between
student and teacher *output pixels*. Early real data (seed 0, response-KD vs
response+freq-KD, matched iterations) shows the delta oscillating in a
±0.1dB band with no consistent sign through 18,000/60,000 iterations —
consistent with a real, not-yet-conclusive null result.

A literature review of why gave a concrete mechanism, not just "it didn't
work": magnitude-only spectral loss discards phase, which carries most of an
image's structural content; complex-mode spectral loss is, by Parseval's
theorem, close to mathematically redundant with the spatial L1/Charbonnier
term already in the KD loss (a unitary transform preserves L2 norm). Neither
mode was ever likely to add much on top of the existing response-KD term.

The more promising direction, per SFKD (arXiv 2607.01906) and this project's
own prior finding (TEST05.5, teacher-experiments): operate on an *internal
representation* — specifically `latent_pre`, AdaIR's bottleneck tensor
(`self.latent`'s output, `net/model.py:441`, captured *before* `self.fre1`'s
frequency modulation is applied) — rather than final-output pixels.
TEST05.5's causal audit found `latent_pre`/its PCA-16 projection (`e_D`) is
the well-supported distillation signal; the frequency pathway specifically
was what TEST05.5 later contradicted. kd_freq targets final-output frequency
content — precisely the part TEST05.5 said was the weaker signal. This
experiment targets the part TEST05.5 said was strong.

**This is a bigger lift than kd_freq.** kd_freq reused 100% of existing
plumbing — student and teacher outputs are the same shape by construction,
so it only needed a new loss function and a config. This needs three new
pieces: a teacher-side feature hook, a shape adapter, and (only once those
are validated) the actual loss.

## The real shape mismatch (measured, not assumed)

| | teacher `self.latent` | student `middle_blks` (M-DEHAZE-KD-FREQ arch) |
|---|---|---|
| channels | 384 (`dim=48 * 2^3`) | 256 (`width=16 * 2^4`) |
| downsamples from input | 3 | 4 |
| spatial resolution | 1/8 | 1/16 |

Neither channel count nor spatial resolution match. A learned adapter is
required — this is exactly what `src/models/teacher_wrapper.py`'s own
docstring flags as "explicitly out of scope for Phase 01": *"feature-
extraction hooks into AdaIR's internals, adapters/projectors, and any
distillation plumbing."* This plan is a deliberate, scoped move into that
territory, not an oversight of the boundary.

## Step 1 — teacher-side feature hook (CPU-only, no GPU needed to build)

Add `FrozenTeacher.forward_with_latent(x)` — non-invasive, matching TEST18's
established safe pattern (`register_forward_hook` on the real submodule,
never reimplementing forward logic): a forward hook on `self.net.latent`
captures its output into a buffer; `forward()` proper is called unchanged;
the hook's captured tensor is returned alongside the normal output. No
changes to `FrozenTeacher.forward()`'s existing contract — a new method,
additive only, so kd_freq and the existing response-KD path are untouched.

## Step 2 — shape adapter (small, trainable, training-time only)

A student-side adapter: `Conv2d(256, 384, 1x1)` to match channels, then
either strided conv or adaptive avg-pool to reconcile 1/16 -> 1/8 spatial
resolution (upsampling the student's coarser map, since the student's own
representation is what should be pulled toward the teacher's, not the
reverse). Standard FitNets-style adapter — small (~0.1M params), trained
jointly with the student via the KD loss, and — same principle already
established for kd_freq's spectrum loss and validated by the export smoke
test — **discarded at inference/export**. It exists only inside the training
loss; it never enters the deployed graph, so this does not reopen the F7
export-gate problem.

## Step 3 — two-phase loss test, same one-thing-at-a-time discipline as kd_freq

**Phase A — plain L1 feature match (baseline for this direction).**
`L = L_response_KD + w_feat * L1(adapter(student.middle), teacher.latent_pre)`.
Establishes whether feature-level distillation helps *at all*, isolated from
the wavelet mechanism, exactly how kd_freq isolated the frequency term over
plain response-KD. If this alone is null, the wavelet refinement in Phase B
is not worth building on top of it.

**Phase B — SFKD-style multi-level wavelet loss on the same matched
features.** The actual novel component: replace/augment the Phase A L1 with
a multi-level wavelet-subband comparison (DWT on the adapted student feature
and the teacher's `latent_pre`, compared per subband per level) — architecture-
agnostic by construction, which is the property that makes it suited to a
transformer-teacher/CNN-student mismatch in the first place (per SFKD's own
framing), rather than an arbitrary swap of transform.

## Step 4 — verify the mechanism, not just PSNR (mandatory, matches project convention)

Track normalized L2 (or cosine similarity) between `adapter(student.middle)`
and `teacher.latent_pre` directly, per checkpoint, for Phase A and Phase B —
the same "verify the loss is closing the gap it targets" check planned for
kd_freq's Phase 2, applied here. Confirms real convergence of the
representations, not just a PSNR number that could move for unrelated
reasons.

## Scheduling

kd_freq's 3 seeds are using ~4.3GB/24GB VRAM — real headroom — but its 8
dataloader workers are already using most of the 8 healthy P-cores
(0-7,12-15) freed up by today's taskset fix. Steps 1-2 (hook + adapter) are
pure code, buildable and unit-testable on CPU with no GPU contention at all.
Actual training runs for Phase A/B should either queue after kd_freq's 3
seeds finish, or run concurrently with an explicit, smaller taskset carve-out
of the remaining healthy cores — decide once Steps 1-2 are built and ready to
train, not before.
