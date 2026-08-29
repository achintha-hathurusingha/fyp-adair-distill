# Plan — precompute and cache the teacher's outputs (cached_teacher)

## Why

`scripts/profile_step_cost.py` (this session) found the frozen teacher's
forward pass is **~78% of every training step's wall-clock time** (288ms of
370ms/step) — a full fp32 forward through a 28.78M-param model with FFT ops,
paid on every single micro-batch, every iteration, for the whole run. Over a
90,000-iteration run at `accum_steps=2`, that's roughly **14.4 hours** spent
purely on `teacher.forward_with_latent()`.

The teacher is frozen — its output for a given input tensor never changes.
If the same tensor were seen twice, recomputing it the second time is pure
waste. The question is whether that actually happens.

## Why this isn't a free win: there is no cache hit today

Training doesn't feed the teacher whole images. `build_multitask_loader`
extracts a fresh random 128×128 crop every draw, and for denoise, samples a
fresh continuous noise level every draw too (`sigma_range: [0, 55]`, the F10
fix). So "the same source image" produces a *different* actual tensor every
time it's sampled — there is effectively zero repetition across a full
training run's ~2.88M sample-draws (90,000 iters × 32 effective batch). A
naive per-image cache would have a near-0% hit rate.

**The fix has to trade something**: convert "infinite" live random
augmentation into a **large but finite, reusable pool**, precomputed once,
then trained on for multiple epochs. This is standard practice for
expensive-teacher distillation (cache the soft targets once, reuse them),
but it is a genuine methodology change, not just a speed optimization — see
"What this changes" below before deciding to apply it to any in-flight
comparison.

## What to cache, and how big

One pool entry = `(degraded_patch, clean_patch, teacher_response,
teacher_latent_pre, task_id)` — everything a training step currently needs
from the teacher, precomputed once via the same `forward_with_latent()` call
already used live.

Per-task pool size, sized against each task's own real diversity (matching
this project's own established acceptance of per-task diversity limits —
derain's live sampler already "repeats to fill its share"):

| Task | Source cardinality | Target pool size | Rationale |
|---|---:|---:|---|
| Denoise | 5,144 images, continuous σ∈[0,55] | 80,000 | ~15 (crop, σ) draws/image — enough σ coverage for F10 |
| Derain | 200 pairs | 20,000 | 100 crops/pair — matches the live sampler's own repeat-to-fill-share behaviour |
| Dehaze | 72,135 hazy variants | 80,000 | ~1 crop per ~0.9 of the variants already gives huge diversity |
| **Total** | | **180,000** | |

Storage: per-task raw binary blobs (numpy memmap, fixed record size — no
LMDB/HDF5, matching this project's stdlib-first discipline, e.g. the
dashboard), not per-sample files (200k small files would be its own
filesystem-overhead problem):

- `degraded`/`clean`: 128×128×3 **uint8** (49,152 B each) — full precision
  buys nothing here, these are natural images.
- `latent_pre`: 384×16×16 **bf16** (196,608 B) — matches the student's own
  training precision, half the size of fp32 for a target that's already an
  approximation.

Per-sample: ~336 KB. **180,000 samples ≈ 60.5 GB** — devon has 530 GB free
(`df -h`, checked directly), no concern.

## Build cost — the one-time price

Teacher-forward measured at ~18ms/image (288ms / 16-image batch, same
profiling run). **180,000 images ≈ 54 minutes** of GPU time, one-time,
versus the ~14.4 hours a single 90k-iteration run currently pays for the
exact same computation, repeated with zero reuse.

## Diversity mitigation — free D4 re-augmentation at train time

A finite pool anchors each entry to one fixed crop position and, for
denoise, one fixed noise realization — a real diversity reduction versus
live sampling. This project has already measured what insufficient
diversity costs here: finding F10, where discrete-only sigma coverage
({15,25,50}) left the model unable to handle a near-clean input (125.37/255
MAE vs the teacher's 1.88/255) until continuous sampling fixed it. A cached
pool's *aggregate* sigma coverage stays broad (each of the 80,000 denoise
entries draws its own continuous σ independently), but any *one* source
image is anchored to only ~15 of those draws for the whole run, not a fresh
one every epoch.

Flips and 90°-rotations are **equivariant**: flipping/rotating the cached
`degraded` and `clean` together with the cached `response` and
`latent_pre` produces an exactly-valid teacher output for that
transformed input — the teacher is deterministic, so this is not an
approximation, it's the same guarantee a live re-run would give, at zero
extra teacher-forward cost. Applying one of the 8 dihedral transforms
(identity, 3 rotations, and their mirror images) at random each time a
pool entry is drawn multiplies effective diversity by up to 8x for free.
`CachedTeacherDataset.__getitem__` applies this on every draw (Step 2 in
the build order below) — it is not a later optimization, it is part of the
design from the start, precisely because the risk it addresses is already
a documented failure mode in this project.

This does **not** fully substitute for the lost crop-position/σ diversity
(D4 transforms of the same crop are still the same crop, just reoriented),
which is exactly why Step 5's validation against the existing live-trained
control's real PSNR — checked at extreme σ, not just the aggregate number —
stays a mandatory gate, not a formality.

## Training-side integration

- New `CachedTeacherDataset` reads the memmap files + a small JSON index
  (per-task row ranges, per-sample σ for denoise), yields the same 5-tuple a
  live batch currently assembles by hand. Wrapped in a sampler that
  balances tasks per batch, the same discipline `BalancedTaskBatchSampler`
  already applies to the live loader.
- New `Trainer` config flag, `distill.use_cached_teacher: true`. When set,
  the training loop reads `response`/`latent_pre` straight from the batch
  instead of calling `teacher.forward_with_latent()` — the teacher model
  itself never needs to be loaded onto the GPU at all in this mode, which
  also frees the ~350 MB of VRAM it currently occupies alongside the
  student.
- Everything downstream (pixel loss, response-KD, feature-KD via the
  adapter, the aux degradation-classification loss) is unchanged — they
  already only consume `pred`, `clean`, `soft`, `teacher_latent`, whichever
  code path produced them.

## What this changes — and why it must NOT be retrofitted into the current comparison

180,000 cached samples, revisited across 90,000 iterations × 32 samples,
means roughly **16 epochs** over a fixed pool — a completely normal
supervised-training regime, but a real methodology change from "every
sample is fresh, never repeated." `B0V2-KD-FEAT` (control) and
`B0V2-KD-FEAT-COND`/`-COND-DECFILM` (treatment) all either have used, or
are built to use, the live/infinite-augmentation pipeline. Switching only
one of them to cached/finite-pool training would introduce a second
confounded variable (injection point *and* augmentation methodology) into
what's supposed to be a single-variable comparison.

**This plan is scoped as a separate, follow-up efficiency track for future
arms** — not applied to `B0V2-KD-FEAT-COND-DECFILM`, which launches matching
control/v1 exactly (live augmentation), keeping that comparison clean. If
the cached pipeline is later validated (e.g. a fresh cached-control run
reaches statistically the same PSNR as the existing live-trained control,
within noise), it becomes the new default for whichever arms come after —
at which point *both* arms of any future comparison would use it
consistently.

## Build order

1. `scripts/build_teacher_cache.py` — the one-time precompute script. Runs
   the real `build_multitask_loader` sampling logic per task (reusing it,
   not reimplementing), calls `teacher.forward_with_latent()` per batch,
   writes to the memmap files + JSON index. Smoke-test on a tiny pool (e.g.
   500 samples) first — verify the index and memmap round-trip exactly
   (write then read back, byte-for-byte).
2. `CachedTeacherDataset` + sampler, WITH the D4 re-augmentation from the
   section above built in from the start — smoke-test in isolation: yields
   correctly-shaped tuples, task balance matches configured ratios, no
   duplicate/missing indices, and verify the equivariance claim directly
   (flip a cached `degraded`/`clean` pair, flip the cached `latent_pre` the
   same way, confirm it matches what a live teacher forward on the flipped
   input actually produces — not just assumed).
3. `Trainer` wiring — `use_cached_teacher` flag, skip loading the live
   teacher entirely when set. Smoke-test: a few real optimizer steps,
   confirm loss values are the same order of magnitude as the live pipeline
   (sanity, not exact match — the cached pool's crops/σ won't line up
   1:1 with a fresh live draw).
4. Run the full 54-minute cache build on devon.
5. Launch one new arm on the cached pipeline (does NOT replace or rerun the
   existing control/treatment arms) — measure the real iterations/second
   this session's own profiling predicts, confirm the ~4-5x speedup
   materializes end-to-end, not just in the isolated teacher-forward number.
