# Overnight report — 2026-07-31 → 2026-08-01

**Headline: the architecture is fully locked.** Normalization is N-F, the family
is S=`w16_b8` / M=`w16_sidd` / L=`w24_b28`, and both the norm decision and the
family selection are confirmed on measured data with regression tests pinning
them. B0 is staged and smoke-tested but **not launched**.

---

## 1. Normalization ablation — the ladder, in full

`w16_b8` (S), denoising only, 30k iterations, batch 32, patch 128, single seed,
identical data / augmentation / optimiser / schedule across every arm.

| arm | normalization | mitigation | result | INT8 ms | speedup |
|---|---|---|---|---|---|
| **Q-A** | LayerNorm2d everywhere | — | **31.019 dB** | 2.513 | 1.00x |
| **Q-F** | affine at full-res only | — | **31.014 dB** (−0.005) | 1.580 | **1.59x** |
| Q-E | affine everywhere | — | diverged @ 3040 | 1.072 | 2.34x |
| Q-E′ | affine everywhere | half LR, 2x warmup | diverged @ 4926 | — | — |
| Q-E″ | affine everywhere | + grad clip 1.0 | diverged @ **4926** | — | — |
| Q-E‴ | affine everywhere | + residual init 0.1 | diverged @ 5899 | — | — |

**Decision: lock N-F.** 0.005 dB against the reference — twenty times inside the
0.10 dB threshold — for 37% of the runtime back. N-E's 2.34x is unreachable:
four arms, three distinct mitigations, all diverged.

### The Q-E‴ prediction failed, and the revision is recorded

We predicted Q-E‴ would die *earlier*, reasoning that zero-initialised residual
scales buy a "protection window" of early stability which starting active
removes. It survived roughly **twice as long** (1899 iterations past warmup
versus 926).

The protection-window framing is therefore wrong. Better reading: from zero the
residual scales take a large uncontrolled gradient push and grow fast; starting
at 0.1 gives the optimiser usable signal to *regulate* them from step one. This
still fits activation-scale growth as the mechanism — it changes the rate, not
the endpoint — but the governing quantity is how fast `beta`/`gamma` **grow**,
not how long they sit near zero. Written up as a revision in `findings.md` F6
rather than folded quietly into the original story.

### The mechanism, stated as a falsifiable claim (findings F6)

**Gradient magnitude is bounded and *falling* at the point of failure, which
rules out gradient-spike instability. Divergence is driven by unconstrained
activation-scale growth through the unnormalized residual stack.**

Two independent lines of evidence:

1. **Q-E″ and Q-E′ produced bit-identical trajectories** — loss `0.05443`/`0.03493`,
   PSNR `23.705`/`26.695`, gradient norm `0.153`/`0.111` — and died at the *same*
   iteration, 4926. Q-E″ differs only by an active gradient-clipping safety net.
   A single clip event forks the trajectories permanently in floating point.
   They never forked, so **clipping fired exactly zero times**. That eliminates
   the entire spike-driven hypothesis class without instrumenting the failing
   step.
2. **Gradient norm falls into the failure** (0.153 → 0.111 → dead), the opposite
   of a spike-driven collapse.

Halving the LR only postponed failure in proportion to the longer warmup — both
Q-E and Q-E′ died ~1000 iterations after warmup completed regardless of peak LR
— so the failure tracks a **step budget past full LR**, not an LR magnitude.

**Consequence:** LayerNorm is load-bearing for *trainability* here, not merely a
quality refinement.

---

## 2. M spot-check — the result that could have invalidated the lock

The ablation ran entirely on S. Family re-selection then moved M to `w16_sidd`
**because it carries the most full-resolution normalization of any config** —
exactly the property that would make an N-F quality cost most visible.

`w16_sidd`, N-A vs N-F, 10k iterations, batch 16 (native; see §5), single seed.

| iter | M-A (N-A) | M-F (N-F) | delta |
|---|---|---|---|
| 2000 | 25.525 | 25.446 | −0.079 |
| 4000 | 29.675 | 29.621 | −0.055 |
| 6000 | 30.543 | 30.561 | +0.018 |
| 8000 | 30.729 | 30.728 | −0.000 |
| **10000** | **30.783** | **30.777** | **−0.006** |

**−0.006 dB on M, against −0.005 dB on S.** Two configs 4x apart in parameters
and structurally very different both put N-F under 0.01 dB. The early deficit is
warmup transient that closes by it 6000, not a capacity gap — a stronger result
than the endpoint alone.

**The lock stands.**

---

## 3. Family re-selection — both tables kept

24 AI Hub jobs (12 configs × N-F, N-E) profiled on Samsung Galaxy S24.
Mean speedup **N-F 1.43x, N-E 2.19x** across all 12 configs, so the w16_b8
result generalises rather than being a single-config artifact.

Selection ran through the **corrected profile-then-select path** — measured
latency attached to every candidate *before* `assign_family` runs. The earlier
bug ran selection first, so it never saw the latency it selects on.

| arm | pre-fix (N-A) | **post-fix (N-F)** | params | GMACs |
|---|---|---|---|---|
| S | `w16_b8` | `w16_b8` | 2.44M | 2.13 |
| M | `w24_b8` | **`w16_sidd`** ← moved | 7.37M | 4.13 |
| L | `w24_b28` | `w24_b28` | 9.68M | 9.05 |

**Latency span 2.45x → 2.67x.** M moved because `w16_sidd` carries the most
full-resolution normalization and so gained most from N-F — the F1 mechanism
showing up in the selection itself.

**Invariants pass:** params and MACs strictly increase S < M < L, measured
latency increases across arms, MAC span 4.26x ≥ 2.5x. Eight regression tests
pin the norm, geometry, family membership, monotonicity and the 10M ceiling.

**Peak memory does not bind.** 98–100 MB across a 4.3x MAC range and unchanged
by normalization — dominated by fixed QNN runtime overhead. It was flagged as a
possible binding constraint ahead of latency; measured, it is not one.

---

## 4. F7 — the teacher cannot be exported at all

Opportunistically attempting a measured teacher latency produced a better
result than the number would have been.

| # | model | exporter | opset | result |
|---|---|---|---|---|
| A0 | unpatched | TorchScript | 17 | `SymbolicValueError` at `model.py:349` |
| A17 | slicing patched out | TorchScript | 17 | `UnsupportedOperatorError: aten::fft_fft2` |
| A20 | slicing patched out | TorchScript | 20 | `UnsupportedOperatorError: aten::fft_fft2` |
| B18 | unpatched | **dynamo** | 18 | `TorchExportError` |
| B20 | unpatched | **dynamo** | 20 | `TorchExportError` |

Both blockers are now **confirmed independently**, not inferred: patching the
value-dependent mask to be shape-derived lets tracing pass the first, and export
then fails on the FFT at opset 17 *and* 20. Dynamo fails on the unpatched model
too, so this is not a TorchScript artifact.

**The teacher is not slow on the target hardware — it cannot run on it at all.**
That makes the project's premise categorical rather than quantitative:
deployable versus not deployable, not 29M-at-X-ms versus 7M-at-Y-ms.

**Consequence for framing:** every speedup against AdaIR is necessarily a
**MAC-derived estimate** and must be labelled as such. There is no measured
teacher latency to divide by, and obtaining one would require rewriting
`FreModule` — at which point the artifact is no longer the published AdaIR.

---

## 5. Batch-size policy (findings F8)

`w16_sidd` (36 blocks) OOMs at batch 32 where `w16_b8` (17 blocks) fits at
2.13 GB. **Decision: effective batch 32 everywhere via gradient accumulation.**

Dropping project-wide to 16 would retroactively change the configuration under
which Q-A (31.019) and Q-F (31.014) were measured — the entire basis of the norm
lock — and re-running to check costs ~9 GPU-hours to defend a sound decision.

Caveat stated rather than glossed: accumulation is not bit-identical to a true
large batch, but this architecture uses no batch statistics (LayerNorm and
affine are per-sample), so it is exact in expectation and near-exact
numerically.

The M spot-check is the deliberate exception — native batch 16 on both arms,
since it is a *relative* comparison; its absolute PSNR is not comparable to the
S numbers and is not used as such.

---

## 6. Bugs found and fixed

**`--resume` was broken.** `torch.load(map_location='cuda')` moves the saved RNG
state onto the GPU; `torch.set_rng_state` then raises
`TypeError: RNG state must be a torch.ByteTensor`. M-F tried to resume, died in
10 seconds, and the runner moved silently to the next arm.

This mattered far beyond the spot-check: **B0 is a 3-seed multi-day run whose
entire safety net is `--resume`.** Fixed, with four tests including one that
round-trips a checkpoint through `map_location='cuda'`. M-F then survived three
further session restarts (2000 → 6000 → 10000), which is the fix proving itself
on real data.

**Teacher caching: two bugs.** AdaIR requires dimensions divisible by 16
(`pixel_unshuffle`) and inputs were loaded uncropped; now cropped to base 16 as
AdaIR's own test path does. And the tiling guard **correctly refused to cache**
when tiled output deviated 0.042 from single-pass — inherent to tiling a
global-attention transformer, since a tile cannot see beyond itself. Tiling is
now an OOM-only fallback; the training set (max 800×768) fits in one pass.

**Instrumentation gap closed.** Sampling gradient norm only at validation points
cannot distinguish a spike from gradual drift. Now tracks a running max and a
clip-hit count. It immediately earned its keep: `w16_sidd` shows spikes to
**1.237, 1.616 and 1.311** — above the 1.0 clip threshold — where `w16_b8` never
exceeded ~0.15. The old sampling would have shown a uniformly quiet 0.03–0.07.

### Process lessons

**Verify no stale process is running before trusting a "fixed" result.** Three
stale python processes running old code were masking the caching fix; the
failures looked identical to the bug I had just fixed.

**Background jobs launched as `nohup ... &` from a tool call do not survive the
session.** Every such launch died. Now using a `.bat` under PowerShell
`Start-Process -WindowStyle Hidden`, which genuinely detaches.

**Do not signal completion with a grep-able marker in a shared log.** Waiters
fired twice on markers left by *previous, failed* runs and reported success.
Waiting on the actual output artifact (`metrics.json`) is correct.

**A probe that fails for the wrong reason is indistinguishable from one that
fails for the right reason.** The first F7 export probe was inconclusive in both
arms — my patch dropped a `conv1` that changes channel count, and the dynamo arm
lacked `onnxscript` — yet it read exactly like evidence.

---

## 7. Teacher caching

Paused at **500/15632 (0.12 GB)** to free the GPU for the spot-check;
manifest-resumable and now running again. Rain100L and BSD400+WED × 3 sigmas are
in scope; RESIDE is sequenced last as the largest, and partial completion there
is expected and acceptable. Storage is far inside the ~65 GB budget.

---

## 8. B0 — staged, smoke-tested, NOT launched

`configs/train/b0_baseline.yaml` written against the locked architecture:
N-F normalization, M arm (`w16_sidd`, 7,371,923 params), Charbonnier loss
against ground truth only, **no teacher involvement**, three seeds.

Smoke test on the locked architecture: 200 iterations, loss decreasing,
validation through the locked harness, checkpoint save/resume verified, **peak
VRAM 1.85 GB logged non-zero**.

**Not launched.** A 3-seed multi-day run waits on review.

---

## 9. For review

1. **Gradient spikes above the clip threshold on `w16_sidd`** (1.24, 1.62, 1.31).
   Harmless in these runs — both arms converged cleanly — but B0 trains the same
   config for 300k iterations rather than 10k. Worth deciding whether B0 should
   enable clipping as insurance. It is currently `null`.
2. **The M spot-check ran 10k iterations, not to convergence**, and at native
   batch 16 rather than the accumulated effective 32. The conclusion (−0.006 dB)
   is consistent with S and the curves converge, but it is a trend check by
   design.
3. **Single seed throughout the ablation.** Differences below seed-to-seed
   variance should not be over-read — though at 0.005–0.006 dB the conclusion is
   "no measurable difference", which is robust to that caveat.
4. **RESIDE is not yet in the teacher cache**, so Phase 02 dehaze distillation is
   not yet unblocked. Denoise and derain are in progress.
