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

| arm | architecture | training | combined | denoise | derain | dehaze |
|---|---|---|---|---|---|---|
| AdaIR teacher | Restormer-ish, 28.78M | — | — | 31.253 | 39.725 | 36.928 |
| B0V2-KD-FEAT | NAFNet 7.37M | +KD | 33.568 @75k | 30.684 | 36.031 | 33.989 |
| B0V3 | StudentV3 7.45M | GT-only | 33.461 @75k | 30.645 | 36.088 | 33.650 |
| **B0V3-KD-FEAT** | **StudentV3 7.45M** | **+KD** | **33.726 @75k** | **30.789** | **36.190** | **34.200** |

**B0V3-KD-FEAT is the best arm.** At 75k it leads the KD baseline by +0.159 and
the GT-only v3 by +0.265, ahead on all three tasks, ahead at every checkpoint
from 9k on. **Removing KD from this architecture costs 0.265 dB** — so
"drop distillation" is contradicted by current data.

### KD is architecture-dependent (the key nuance)
KD *hurt* NAFNet (−0.79 overall, −2.59 rain in TEST07-B) but *helps* StudentV3.
Its effect tracks the per-task teacher-student gap:

| task | teacher gap | KD effect (TEST07-B) | KD effect (90k arms) |
|---|---|---|---|
| denoise | 0.567 dB | +1.238 | +0.007 |
| dehaze | 2.283 dB | −1.027 | −0.545 |
| derain | 2.897 dB | −2.591 | −0.757 |

r = −0.987 / −0.9999, **identical rank ordering in two independent experiments**.
Caveat: n=3, so the reproducing *order* is the evidence, not the coefficient.

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

- **devon:** `B0V3-KD-FEAT` at ~75k/90k, healthy, ~1h remaining.
  `runs/b0v3_kd_feat/`.
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
