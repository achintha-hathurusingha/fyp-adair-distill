# Project context — AdaIR distillation FYP

**"Distillation for Degradation Adaptive Mobile Image Restoration."** Read this
before doing anything. Everything below is measured, not assumed.

---

## 1. Infrastructure

| host | address | GPU | notes |
|---|---|---|---|
| **devon** | `minura@192.248.10.68` | RTX 4090, 24.5 GB | primary. Repo: `/home/minura/fyp-adair-distill` |
| **qbits** | `minura@192.248.10.67` | RTX 4080 SUPER, 16.4 GB | shared. **Often blocked** by another user's idle vLLM |

- SSH key: `/c/Users/User/Documents/FYP/Achintha`
- Conda env: `adair-distill` on devon (`~/miniforge3`), `fyp` on qbits
- **CPU pinning required on devon: `taskset -c 0-7,12-31`** (cores 8-11 unreliable)
- devon→qbits direct transfer key: `~/.ssh/devon2qbits` (much faster than
  relaying through Windows over the VPN — a 15.5 GB transfer took 28 min direct
  after repeated failures via the laptop)

### Hard-won environment gotchas
- **The Bash tool's cwd resets to the Windows path every call.** Always embed
  `cd` in the same command, and always operate on devon — the local Windows
  mirror `C:\Users\User\Documents\FYP\fyp-adair-distill` is **stale and unrelated**.
- **YAML is authoritative over CLI flags** in this repo (`--iters 40` was
  silently ignored; `total_iters: 90000` won).
- **Nested heredocs inside `ssh "..."` break.** The local shell expands `$VAR`
  and backticks before transmission. Write the script locally and `scp` it, or
  use `git commit -F file`. This bit three times.
- **`reports/*.png` is gitignored.** Commit the `.npz`/script, not the figure.
- Another user (`hirusha`) repeatedly leaves an **idle vLLM** holding ~14.8 GB at
  **0% utilisation** on qbits. We cannot kill it (different UID, no sudo). Fix to
  suggest: vLLM `--enable-sleep-mode` + `POST /sleep` frees 90%+ without stopping
  the server.

---

## 2. The one-paragraph story

AdaIR (ICLR 2025) is built on an adaptive frequency mask. We found the mask is
**mathematically zero at every resolution the model trains at** —
`h_ = (h // 128 * rate).int()` floors to 0 below 256px, and AdaIR trains at
128px. With an empty mask the FFT and inverse FFT cancel and the module
degenerates to `torch.abs()`. Repairing it changes restoration by **~0.00 dB**.
The paper's founding *observation* (degradations occupy different frequency
bands) is nonetheless **true** — but a plain CNN beats the spectral feature at
the task the spectrum was designed for.

---

## 3. Measured results that everything else rests on

### The noise floor
**0.035 dB** (mean |seed0 − seed1| over 51k–66k, two full independent seeds).
Early training is ~0.21 dB, collapsing as it converges. **Anything under
~0.07 dB is not a result.**

### Current best arms (our own harness, per-task PSNR)

> **All numbers below are on the CORRECTED, leak-free test sets** (BSD68 /
> Rain100L-100 / SOTS-clean-417), re-scored 2026-08-31. Anything quoted from an
> older report using `test/derain/demo` or `test/dehaze/demo` is invalid: those
> sets were carved out of the TRAINING corpora and multi-task runs never
> excluded them. The leak was worth **~1.9 dB** on the combined metric (dehaze
> ~4.6 dB, derain ~0.6-1.1 dB; denoise unaffected). See
> `reports/clean_eval_rescore.json`, `scripts/rescore_clean.py`.

**Harness validation:** on the corrected sets our teacher reproduces AdaIR's
published table **exactly** — denoise 31.2534 (paper 31.253) and derain 38.6412
(paper 38.64). Two independent exact matches.

| arm | architecture | training | mean3 | denoise | derain | dehaze |
|---|---|---|---|---|---|---|
| AdaIR teacher | Restormer-ish, 28.78M | — | 33.322 | 31.253 | 38.641 | 30.072 |
| **B0V3-KD-FEAT** | **StudentV3 7.45M** | **+KD** | **31.899 @90k** | **30.794** | 35.159 | **29.744** |
| B0V3 | StudentV3 7.45M | GT-only | 31.788 @90k | 30.649 | **35.547** | 29.167 |
| B0V2-KD-FEAT | NAFNet 7.37M | +KD | 31.666 @90k | 30.693 | 35.133 | 29.171 |

(`last.pth` at 90k, not `best.pth` — `best.pth` is selected on the validation
set, and validation runs on the test set, so it would be selection on test. At a
fixed 90k budget the final checkpoint carries no such selection.)

**B0V3-KD-FEAT is still the best arm overall** (+0.111 vs GT-only v3, +0.233 vs
the KD baseline). But the KD story changed: **KD now HURTS derain by 0.388 dB**
(GT-only v3 wins that task) and the overall KD benefit is carried almost
entirely by dehaze (+0.577). "Removing KD costs 0.265 dB" was a leaked-eval
artifact; the true cost is **0.111 dB**, and it is task-dependent.

**The project reframes.** The student is within **0.33 dB on dehaze** and
**0.46 dB on denoise** of a 28.78M Restormer-class teacher, at 7.45M params and
deployable. **Derain is the only real gap left (3.48 dB)** — which is exactly
where S0.1 says the oriented block should act, and exactly where KD is hurting.

### KD is architecture-dependent (the key nuance)
KD *hurt* NAFNet (−0.79 overall, −2.59 rain in TEST07-B) but *helps* StudentV3.
Its effect tracks the per-task teacher-student gap:

**Corrected (leak-free) gaps and KD effects — the hypothesis got STRONGER:**

| task | teacher gap | KD effect (B0V3-KD-FEAT − B0V3) |
|---|---|---|
| dehaze | **0.328 dB** (smallest) | **+0.577** (most positive) |
| denoise | 0.459 dB | +0.144 |
| derain | **3.482 dB** (largest) | **−0.388** (most negative) |

Perfectly monotonic: the smaller the teacher-student gap, the more KD helps.
Cleaner than the pre-correction numbers, and it shows the old ordering was
partly an artifact — **dehaze moved from second-largest gap to smallest**.
Caveat: n=3, so the reproducing *order* is the evidence, not a coefficient.

The older `KD effect (90k arms)` column (denoise +0.007, dehaze −0.545, derain
−0.757) is **withdrawn**: it was leaked-eval, and it also appears to have
compared B0V2-GT-only@300k against B0V2-KD-FEAT@90k — arms differing in
iterations (300k vs 90k) and in noise sampling (`sigma_range`+`clean_prob` vs
discrete `sigmas`), not just in KD. `reports/kd_lit_review/review.md` still
quotes it and needs revisiting.

### Frequency findings
- **AdaIR's mask is inert** — 6 independent tests, plus TEST18 retraining all
  5 Table-7 variants from scratch: **the paper's monotonic a→e ablation does not
  reproduce**. Fixed mask (28.738) beat the full architecture (28.674);
  *learned* mask (27.953) was worst of all five, below the no-AFLB baseline.
- **AdaIR minus frequency = 28.572 vs full 28.674** → its headline mechanism is
  worth **0.10 dB**. Functionally it is Restormer with a dead frequency branch.
- **Spectral premise is TRUE**: 93.6% blind degradation ID same-scene, clean
  control exactly at chance. But the corpus version was inflated by a **65.8%
  dataset-identity floor**.
- **DFC** (arXiv:2605.17506) reimplemented on our data: band-wise
  residual/degraded energy ratio gives **97.5%** vs 87.5% for the raw radial
  spectrum. The ratio cancels image content analytically.
- **Geometry matters** (dims matched at 24): separable oriented **98.3%** >
  radial 97.5% > square 93.3%, and the entire gain is on **rain**, the only
  anisotropic degradation. AdaIR's rectangle *has* that orientation freedom and
  squanders it (AFLB3 α=0.496, β=0.497 — effectively square).
- **Convolution theorem**: 7×7–11×11 kernels reproduce the full optimal
  frequency filter. **But the linear ceiling is low** — the full filter is worth
  only +0.68 dB dehaze / +0.87 dB derain.

### Deployment gate (non-negotiable)
- **`torch.fft` does not export to ONNX.** The teacher is undeployable at any size.
- Attention (MatMul/Softmax/ReduceL2) is **UNKNOWN on all three** NPU backends.
- Verified-supported: Conv, Add, Mul, Sigmoid, ReLU, MaxPool, AveragePool,
  GlobalAveragePool, Concat, Split, Clip, Reshape, Transpose, DepthToSpace.
- StudentV3 exports with **zero new risk categories**.
- Real Snapdragon numbers: `dynamic_conv` is catastrophic (4,059 ms vs 54–139 ms);
  normalization choice, not conditioning, dominates latency (verified **2.80×**).

### Null results worth not repeating
feature-KD on NAFNet · v1 FiLM conditioning (regressed — it modulated the exact
tensor feature-KD reads) · trajectory distillation · higher operator rank ·
adaptive basis · frequency descriptor (CCA 0.867 with the existing embedding) ·
frequency-mask repair (~0.00 dB) · a collaborator's reported +1.07 dB (a
synthetic-haze artifact; on real SOTS it is −1.94 dB).

---

## 4. Where things stand right now

- **devon:** GPU **free**. `B0V3-KD-FEAT` was **deliberately stopped at 81k/90k**
  on 2026-08-31 21:13 (not a crash) — it had plateaued (78k→81k = +0.003 dB) and
  the GPU was being shared. **It still needs finishing to 90k**: B0V2-KD-FEAT and
  B0V3 both ran a full 90k, so an 81k number makes schedule length a second
  variable in the headline comparison. Exact resume command, state and rationale:
  `runs/b0v3_kd_feat/.../B0V3-KD-FEAT_seed0_20260831_083259/RESUME.md`.
  ~75 min on an uncontended card. `last.pth` = iteration 81000, best 33.768,
  full optimizer/EMA/RNG state — `--resume` is an exact continuation.
- **B0V3-KD-FEAT @81k = 33.768**, already above B0V2-KD-FEAT's and B0V3's *90k
  finals* (33.621, 33.520), so the ordering is settled; only the margin is open.
  At the matched 75k: **+0.159** vs B0V2-KD-FEAT, **+0.265** vs B0V3, ahead on
  all three tasks. B0V3 has two seeds (seed0 stopped at 69k, seed1 ran to 90k);
  seed0-vs-seed1 mean |Δ| over ≥51k is **0.031 dB**, independently reconfirming
  the 0.035 noise floor on this architecture.
- **Phase 0 of the plan is COMPLETE: S0.1 `[x]` PASS, S0.2 `[x]` PASS, S0.3 `[x]` KILL.**
  Reports in `reports/reparam_gate/`. S0.2: the reparameterized block's merged
  deployment graph is **one depthwise Conv, zero UNKNOWN ops on all three
  backends in FP32 and INT8** — Phase 3 is not export-blocked, and S3.2 is
  pre-satisfied on the stub. S0.1: orientation is worth **+0.385 dB (derain)**
  over isotropic and **+0.009 / +0.001** on denoise/dehaze — keep the four
  orientations at k=11, rain only. **Caveat that must not be lost:**
  `add_rain` draws `angle ~ U(-15,15)` and real RainTrainL is 93°±13°, so
  *neither corpus contains off-axis rain* — which is the only regime where the
  oriented bank beats a cheap axis-aligned kernel. A win in S3.3 on current test
  sets is therefore **not** the mechanism S0.1 measured.
  **S0.3: KILLED — keep PCA-16.** Degradation-ID from the student's own
  decoder features saturates: `concat` reaches **99.67% at TWO PCA dims**,
  and no dimension anywhere beats 16 by more than +1.17 pp (clean control
  exactly at chance). Width is not the bottleneck, so the plan's premise
  of "a richer representation replacing PCA-16" loses its motivation —
  **and S2.1's kill criterion (< +2pp over PCA-16) is now unsatisfiable,
  since only +0.33 pp exists above 99.67%. S2.1 must be redefined on
  severity / unseen degradations / downstream PSNR before it is run.**
- **qbits:** blocked (idle vLLM, 1,073 MB free). `B0V2-KD-DENOISE-ONLY` is
  **built, registered, verified, and OOM-killed** — relaunch when the card frees.
  Task-selective KD is implemented in `trainer.py` via `distill.kd_tasks`, with
  a passing unit test (correct samples, correct gradients, safe empty-batch guard).
- **Built but never launched:** `B0V3M` (multi-level strip pooling),
  `B0V2-KD-FEAT-COND-DECFILM` (v2 decoder FiLM).

---

## 5. Key documents

| file | what |
|---|---|
| `reports/research_plan/plan.md` | **the staged plan — start here** |
| `reports/kd_lit_review/review.md` | why KD keeps failing; capacity-gap analysis |
| `reports/freq_spatial_review/review.md` | frequency restoration via spatial mathematics |
| `reports/dfc/` | DFC reimplementation + geometry experiment |
| `reports/student_v3/design.md` | StudentV3 rationale |
| `teacher-experiments/AdaIR_Testing_Report.md` | TEST01–17 index |
| `teacher-experiments/test18/` | the from-scratch retrain that breaks Table 7 |
| `C:\Users\User\Documents\FYP\Research_Plan_Staged.md` | local copy of the plan |

---

## 6. Working style that has worked

- **Controls before conclusions.** Every finding checked against self-swap,
  random, zero, mean, cross-scene controls before being believed.
- **Single variable per arm.** Where two differ, say so explicitly.
- **Keep negative results.** Several conclusions here correct an earlier claim
  made in this same project, including ones I made.
- **Verify before spending GPU hours** — unit-test the mechanism first
  (the KD mask test caught nothing, but the byte-identity test for v3 was
  load-bearing).
- **Real measurements only.** No illustrative numbers in any report or figure.
