# Plan — M-DEHAZE-KD-FREQ: does the frequency-domain distillation term help?

## Why this experiment, and why now

`src/losses/frequency.py` (`spectrum_loss`) and `configs/train/m_dehaze_kd_freq.yaml`
were both written on 2026-08-08, in direct response to F7 (`reports/findings.md`):
AdaIR's teacher advantage on dehazing comes substantially from `FreModule`'s FFT
mining, and that exact mechanism is also why the teacher cannot export
(`aten::fft_fft2` has no ONNX path — confirmed independently across 5
exporter/opset combinations, see F7). `m_dehaze_kd.yaml`'s response-only KD term
already asks the student to match the teacher's *pixels*; `m_dehaze_kd_freq.yaml`
adds a second term asking it to match the teacher's *spectrum*
(`|rfft2(student)| vs |rfft2(teacher)|`, L1, magnitude mode, weight 0.2) — trying
to transfer the architectural property through supervision alone, to a student
that has no frequency-domain architecture and therefore still exports and
quantizes cleanly.

The config was authored and diff-tested against `m_dehaze_kd.yaml` (identical
except the two `distill.freq_*` keys) but **no run directory for it exists
anywhere on devon** — it was built and never executed. This plan is to finish
it, not redesign it.

**Independent motivation**: a literature review this week (WKD/CVPR2022,
MTKD/2024, Focal Frequency Loss/ICCV2021, Guided Frequency Loss/2023) found
direct, quantified precedent for exactly this design — fixed (non-learned)
frequency transform, applied only as a training-time loss between student and
teacher outputs, never baked into the deployed graph. MTKD's own ablation
(RCAN×4, Urban100) measured +0.30dB from a DWT-subband loss over plain L1. This
project's own TEST18 (teacher-side AdaIR ablation) separately found that
AdaIR's *internal, learned* frequency mask is degenerate at practical
resolutions and that a crude *fixed* mask outperformed it — which is the same
lesson this design already encodes: prefer a fixed transform, not a learned
gate that can collapse.

## Existing baseline (already complete, 2026-08-05, `reports/report_demo_dehaze.md`)

3 seeds, 60k iters, 4,000-image seeded RESIDE-OTS subset (`reports/dehaze_train_list.txt`),
held out on `dehaze_demo_heldout`:

| model | PSNR | SSIM | gap to teacher |
|---|---:|---:|---:|
| AdaIR (teacher) | 34.5056 | 0.9878 | — |
| M, GT only | 32.8898 | 0.9821 | +1.6159 dB |
| M, GT + response KD | 33.0759 | 0.9828 | +1.4298 dB |

Response KD alone: **+0.1861dB, closes 11.5% of the gap.** Flagged in the
existing report as not yet established above seed noise (single seed measured
for the delta; a separate 3-seed B0-denoise measurement put seed-to-seed
variance at 0.0079dB, given here as the noise-floor reference, not a substitute
for actually measuring it on this task).

## Phase 0 — 1-seed sanity run (do first)

`python src/train/train.py --config configs/train/m_dehaze_kd_freq.yaml --seed 0`

No new code. Purpose: confirm the run doesn't crash or diverge (the frequency
term runs the teacher live in fp32 outside autocast — `frequency.py` notes
`torch.fft` has no bf16 kernel — so this is also a check that the added FFT
call doesn't blow the training step time or memory budget unexpectedly)
before committing to 3 full seeds.

## Phase 1 — 3-seed run, matching the response-KD experiment's rigor

Seeds 0/1/2, same config otherwise. Produces the three-way table the config's
own docstring calls for and that response-KD alone couldn't attribute cleanly:
**GT-only / +response / +response+frequency**, each a single added term over
the last, evaluated on the same held-out set with the same harness
(`src/eval/evaluate.py`).

## Phase 2 — verify the mechanism, not just the metric

`spectrum_loss()` optimizes magnitude-spectrum L1 between student and teacher
output. After Phase 1, compute that exact quantity directly — per radial
frequency band (low/mid/high, same banding convention TEST05 used on the
teacher side) — on held-out images, for all three trained models. This checks
whether the frequency term is actually closing the spectral gap it targets,
not just riding along with whatever moves PSNR. No retraining required —
forward pass + FFT on the already-trained checkpoints.

## Phase 3 — conditional (only if Phase 1 shows a gain that clears seed noise)

- `freq_weight` sweep {0.1, 0.5, 1.0}, same pattern as the existing
  `m_dehaze_kd_w05.yaml` / `m_dehaze_kd_w20.yaml` weight sweep for the response
  term. 0.2 is explicitly flagged untuned in the current config.
- One `freq_mode="complex"` run at the same weight. `frequency.py`'s own
  docstring calls magnitude-vs-complex a genuinely split literature question,
  resolved here by a priori haze-physics argument (haze is an affine map on
  the degradation model, so magnitude should matter more than phase) rather
  than measurement. One run makes it measured.

## Phase 4 — close the loop on the reason this design was chosen at all

Run the existing export smoke test (`reports/export_smoke_test.md`'s script)
on the Phase-1 checkpoint. The premise — a frequency-domain *loss* sidesteps
F7's export-gate finding because it is never present in the deployed graph —
is currently an architectural claim in a docstring, not a measured one for
this specific trained model. Confirms zero frequency ops in the exported
graph and zero NPU fallback, matching the other profiled student candidates
in `reports/student_sweep.md`.

## Output

`reports/report_demo_dehaze_kd_freq.md`, same format as
`reports/report_demo_dehaze.md`, extending its table by one row plus the
Phase 2 spectral-gap figure and, if reached, the Phase 3 sweep table.
