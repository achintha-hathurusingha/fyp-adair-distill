# Overnight autonomous run log

Started 2026-07-30 21:23 UTC


## Task 1 — trainer built, smoke test passed

- 200-iter smoke: loss decreasing, validation via locked harness, checkpoint
  save/resume, **peak VRAM 2.14 GB logged (instrumentation verified non-zero)**.

### Throughput measurement and two deviations from the spec

Measured on the RTX 3050 6GB:

| batch | patch | ms/iter | peak VRAM | 30k iters |
|---|---|---|---|---|
| 32 | 256 | **6214** | **8.38 GB** | 51.8 h |
| 16 | 256 | 473 | 4.21 GB | 3.9 h |
| 32 | **128** | **244** | **2.13 GB** | **2.0 h** |
| 64 | 128 | 473 | 4.21 GB | 3.9 h |

**Deviation 1 — patch 128 instead of 256.** At batch 32 / patch 256 the run
needs 8.38 GB on a 6 GB card, spills to host memory, and collapses to 6.2 s/iter
(25x slower than the same batch at patch 128). Patch **128 is also AdaIR's own
training patch size** (`options.py:15`), so this moves *closer* to the reference
protocol rather than away from it.

**Deviation 2 — 30k iterations instead of 50k.** 50k x 3 arms = 10.2 h of pure
compute before validation, which does not leave room for the escalation ladder
or Task 5. 30k gives ~2.3 h/arm including validation.

Both deviations are applied **identically to every arm**, so the controlled
comparison — the entire point of the ablation — is preserved. Only absolute
quality is affected, not the ranking.

**Caveat to carry into the report:** at 30k iterations the arms may not have
fully separated, and a ranking at 30k is not guaranteed to hold at convergence.
Flagged for review.

=== 22:01 UTC starting Q-A ===

### JUDGEMENT CALL — restarted the arms on BSD400+WED (2026-07-31 ~07:25 local)

Q-A had reached iteration 2000 (PSNR 25.61) on **BSD400 alone** when the WED
download completed. Killed and restarted all arms on the full set.

Reason: 30k iters x batch 32 = 960k samples. Over BSD400's 400 images that is
**2400 epochs** — a severe overfit regime that risks compressing the very
differences between norm variants the ablation exists to measure. With WED
(4744 images, verified count) the set is 5144 images and **187 epochs**, which
is a normal training regime.

Cost: ~15 minutes of Q-A progress. Judged clearly worth it — an ablation run in
a heavy-overfit regime could have produced a null result that reflected the data
budget rather than the architecture.

### Also noted: create_run_dir stalled ~3.5 h

config.yaml written 03:31, env.txt 07:07. The `pip freeze` subprocess in
`create_run_dir` blocked, most likely contending with the concurrent AI Hub
submission and WED download. Training did not begin until 07:07. Not fatal, but
it cost 3.5 h of the night and is worth making non-blocking (or cached) before
the next unattended run.

=== 01:53 UTC starting Q-A (BSD400+WED, 5144 imgs) ===

### Session restart (03:11 UTC) — arms relaunched

The Claude Code session ended and took all background jobs with it. Q-A had only
just started (no validations logged), so nothing measurable was lost.

Fixed the root cause of the earlier 3.5 h stall first: `_pip_freeze` in
`create_run_dir` is now bounded by a 60 s timeout and degrades to a recorded
note. An environment dump is useful provenance but is never worth stalling a
training run for.

Relaunched Q-A -> Q-F -> Q-E with `--resume` support, so a further interruption
costs at most one 2k-iteration checkpoint interval per arm.


## Task 0 COMPLETE — 24/24 norm-variant jobs profiled on S24

| config | N-A ms | N-F ms | N-E ms | N-F x | N-E x |
|---|---|---|---|---|---|
| w16_b8 | 2.513 | 1.580 | 1.072 | 1.59 | 2.34 |
| w16_b14 | 2.742 | 1.818 | 1.170 | 1.51 | 2.34 |
| w16_b28 | 3.299 | 2.355 | 1.458 | 1.40 | 2.26 |
| w32_b8 | 3.160 | 2.263 | 1.489 | 1.40 | 2.12 |
| w24_b8 | 3.195 | 2.248 | 1.536 | 1.42 | 2.08 |
| w32_b14 | 3.590 | 2.679 | 1.732 | 1.34 | 2.07 |
| w24_b14 | 3.519 | 2.576 | 1.735 | 1.37 | 2.03 |
| w16_sidd | 4.742 | 2.873 | 1.840 | 1.65 | 2.58 |
| w24_b28 | 4.256 | 3.320 | 2.124 | 1.28 | 2.00 |
| w32_b28 | 4.469 | 3.543 | 2.217 | 1.26 | 2.02 |
| w32_sidd | 6.158 | 4.226 | 2.673 | 1.46 | 2.30 |
| w24_sidd | 6.091 | 4.225 | 2.792 | 1.44 | 2.18 |

**Mean speedup: N-F 1.43x, N-E 2.19x.** The w16_b8 result generalises — this is
not a single-config artifact.

**Latency span widens as predicted:** N-A 2.45x -> N-F 2.67x, N-E 2.60x. Removing
the large roughly-fixed normalisation cost decompresses the range.

**The speedup is non-uniform, and its ordering supports F1.** N-F ranges
1.26x (w24_b28) to 1.65x (w16_sidd). `sidd` configurations gain most because they
place two blocks at full resolution — the most full-resolution normalisation of
any config — which is exactly what a per-element, resolution-weighted cost model
predicts.


## Task 2 in progress — Q-A training healthily

| iteration | loss | BSD68 PSNR | SSIM | grad norm | peak VRAM |
|---|---|---|---|---|---|
| 2000 | 0.04310 | 25.320 | 0.6655 | 0.126 | 2.14 GB |
| 4000 | 0.02479 | 29.386 | 0.8225 | 0.158 | 2.14 GB |

~10.3 min per 2k iterations including validation, so ~2.6 h per arm and ~7.8 h
for the three primary arms. Gradient norms are small and stable; no sign of
instability.

## Tooling built ahead of the data (Tasks 4 and 6)

Both report generators are written and exercised *before* the results land, so
the analysis path is tested rather than improvised at the end.

- `scripts/norm_quality_report.py` — plots PSNR / loss / gradient norm /
  per-level activation magnitude, and applies the decision rule mechanically,
  including the STOP-AND-REPORT branch.
- `scripts/reselect_family.py` — profile-then-select, keeps both pre- and
  post-fix tables with rank-change markers, exits non-zero on an invariant
  failure.

**Dry run of the re-selection against N-E latency already shows the predicted
effect** (the norm is not locked yet — this is a rehearsal, not a decision):
latency span **2.45x -> 2.60x**, and **M moves from `w24_b8` to `w16_sidd`**.
`w16_sidd` carries the most full-resolution normalisation of any config and so
gained the most from removing it (2.58x). Invariants pass.


## Throughput anomaly — machine sleep, not a training problem (18:20 local)

Q-A reached it 16000/30000. Apparent average throughput looked ~4x worse than
the measured rate, but the validation timestamps show the cause:

| interval | wall clock |
|---|---|
| it 4000 -> 6000 | **3 h 22 min** |
| it 6000 -> 8000 | **5 h 06 min** |
| it 12000 -> 14000 | 10 min |
| it 14000 -> 16000 | 11.5 min |

Two long stalls, with normal ~10.5 min/2k either side. This is the host
suspending while idle, not a slowdown in training. GPU is at **85 C** and
1995 MHz — near laptop throttling limits, worth noting but not the cause.

**Training itself is healthy and plateauing:**

| iteration | loss | BSD68 PSNR | SSIM | grad norm |
|---|---|---|---|---|
| 12000 | 0.02090 | 30.87 | 0.869 | — |
| 14000 | 0.02076 | 30.908 | 0.8703 | 0.055 |
| 16000 | 0.02055 | 30.940 | 0.8711 | 0.157 |

+0.03 dB per 2k iterations and falling, so the arm is close to its plateau. If
time forces an early stop, arms will be compared at a **common iteration count**
(checkpoints and history are written every 2k), which keeps the comparison
controlled — the ablation is a relative measurement, and all arms share data,
seed, schedule and budget.

=== 14:05 UTC Q-A finished exit=0 ===

## Q-A COMPLETE — reference established (14:05 UTC)

| metric | value |
|---|---|
| best BSD68 PSNR | **31.019 dB** |
| SSIM | 0.8734 |
| iterations | 30,000 |
| peak VRAM | 2.14 GB |
| diverged | no |

Converged cleanly — 28k to 30k moved PSNR by +0.001 dB, and gradient norms fell
to ~0.02 without any instability.

### Decision thresholds now fixed against this reference

| Q-E (best rung) | vs Q-A 31.019 | action |
|---|---|---|
| >= 30.919 dB | within 0.10 dB | **lock N-E** (2.34x on w16_b8) |
| 30.719 - 30.919 | 0.10-0.30 dB worse | check Q-F; within 0.10 dB -> lock N-F, else STOP AND REPORT |
| < 30.719 dB | >0.30 dB worse | lock N-A if Q-F also >0.15 dB worse |

Q-F started 19:36 local. Q-E queued behind it.

