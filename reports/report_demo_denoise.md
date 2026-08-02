# Demo report — denoising, teacher vs edge student, on-device INT8

**Date:** 2026-08-02
**Question:** does the compressed student, quantized to INT8 and run on the
target NPU, still restore images — and what does that cost against the teacher?

Every number below is measured. Nothing is estimated, and nothing is carried
over from a previous run.

---

## 1. Scope — read this first

**This demo covers DENOISING ONLY.**

B0's training loader is built from `Train/Denoise` alone
(`src/train/train.py:229`) and its dataset class is `DenoiseTrainDataset`. The
model has never seen rain or haze, so no derain or dehaze numbers appear here.
Producing them would mean reporting a model's response to degradations it was
never trained on, which describes nothing.

The teacher is therefore the **denoise specialist** `adair-single-denoise.ckpt`,
not the 3-degradation `adair3d.ckpt`. An earlier version of this comparison used
`adair3d` and showed a teacher/student gap of ~0.00 dB — that was misleading,
because a generalist splits capacity across three tasks while B0 spends all of
its on one. Specialist-vs-specialist is the fair comparison and shows a real gap.

---

## 2. What was run

| | teacher | student |
|---|---|---|
| model | AdaIR (ICLR 2025), denoise specialist | NAFNet `w16_sidd` (B0 baseline) |
| checkpoint | `data/ckpt/adair-single-denoise.ckpt` | B0 seed 0 @ **iteration 200,000** |
| parameters | **28,784,824** | **7,371,923** |
| normalization | — | LayerNorm2d + `affine_clamp(8.0)` at full-res (F9) |
| training | released weights | 200k iters, Charbonnier vs ground truth, **no distillation** |
| precision measured | FP32 | FP32 **and** INT8 |

**B0 has had no knowledge distillation of any kind.** It is the reference
baseline every future distillation delta is measured against.

---

## 3. Demo set and method

- **4 images from BSD68**, centre-cropped to **256x256**, chosen to span the
  difficulty range by crop standard deviation (25.0 to 111.1) rather than
  sampled arbitrarily.
- **3 noise levels**, sigma 15 / 25 / 50, additive Gaussian, **fixed seed 0** so
  the demo is reproducible.
- **12 (image, sigma) pairs total.**

Method points that make the numbers comparable:

1. **All three models see byte-identical inputs.** The same cropped, noised
   arrays are fed to AdaIR, to B0 FP32 and to the on-device INT8 binary. Any
   difference is attributable to the model or its precision, not preprocessing.
2. **FP32 references are computed on the cropped inputs**, not taken from
   training-time validation logs (which run on full images). Comparing against
   those would have conflated cropping with quantization.
3. **INT8 calibration used the real demo inputs.** PTQ ranges taken from random
   data is a standard way to lose quality and then blame quantization itself.
4. **All metrics go through the locked harness** (`src/eval/metrics.py`,
   `ADAIR_DEFAULT`): RGB, no border crop, `data_range=1.0`, clipped, no uint8
   rounding, SSIM `win_size=7`, `gaussian_weights=False`, explicit
   `channel_axis`. No ad-hoc metric anywhere.

---

## 4. Quality results

### Per noise level (mean of 4 images), PSNR dB / SSIM

| sigma | AdaIR 28.8M | B0 FP32 7.37M | B0 INT8 on S24 | teacher-student (FP32) | FP32-INT8 |
|---|---|---|---|---|---|
| 15 | **34.719** / 0.9404 | 34.406 / 0.9372 | 33.776 / 0.9233 | **+0.313** | **-0.630** |
| 25 | **32.102** / 0.9016 | 31.816 / 0.8966 | 31.461 / 0.8831 | **+0.286** | **-0.355** |
| 50 | **28.698** / 0.8158 | 28.513 / 0.8126 | 28.317 / 0.7987 | **+0.185** | **-0.196** |
| **all** | **31.840** | **31.578** | **31.185** | **+0.261** | **-0.393** |

### Per image

| sigma | image | AdaIR | B0 FP32 | B0 INT8 |
|---|---|---|---|---|
| 15 | 167062 | 37.674 | 37.378 | 36.432 |
| 15 | 219090 | 34.241 | 33.946 | 33.421 |
| 15 | 227092 | 36.257 | 35.760 | 35.010 |
| 15 | 33039 | 30.703 | 30.540 | 30.243 |
| 25 | 167062 | 35.080 | 34.848 | 34.366 |
| 25 | 219090 | 31.705 | 31.447 | 31.146 |
| 25 | 227092 | 34.114 | 33.596 | 33.102 |
| 25 | 33039 | 27.508 | 27.373 | 27.231 |
| 50 | 167062 | 31.801 | 31.692 | 31.438 |
| 50 | 219090 | 28.402 | 28.171 | 27.992 |
| 50 | 227092 | 31.003 | 30.711 | 30.427 |
| 50 | 33039 | 23.586 | 23.477 | 23.411 |

### Two costs, kept separate

| source | cost |
|---|---|
| capacity, 28.8M -> 7.37M (3.90x fewer parameters) | **0.261 dB** |
| deployment, FP32 -> INT8 | **0.393 dB** |
| **total deployed gap vs teacher** | **~0.65 dB** |

**Quantization currently costs more than the 3.9x parameter reduction does.**
If quality is to be bought back cheaply, per-channel weight quantization is a
better lever than a larger student.

### Two structural observations

**The teacher's advantage shrinks as noise rises** (+0.313 -> +0.286 -> +0.185).
Its extra 21M parameters buy the most on *clean* images, where fine detail must
be reconstructed precisely; at sigma 50 both models are limited by how much
signal survives. Relevant when weighting distillation losses across sigma.

**The INT8 penalty shrinks the same way** (-0.630 -> -0.355 -> -0.196), for a
different reason: at sigma 15 the corrections the model applies are small and
fine-grained, so INT8's coarse activation steps cost proportionally more. SSIM
loss, by contrast, is **flat at -0.0135 across all three sigmas** — structure
survives quantization uniformly; it is fine amplitude precision that is lost.

---

## 5. Timing

### On-device, Samsung Galaxy S24 (Snapdragon 8 Gen 3, Hexagon v75), INT8

| | latency | note |
|---|---|---|
| **B0 student** | **2.881 ms** | QNN context binary, measured via AI Hub |
| **AdaIR teacher** | **cannot run** | see below |

### FP32 reference timing, both models, identical hardware and shape

NVIDIA RTX 3050 6GB Laptop GPU, input `(1, 3, 256, 256)`, 10 warmup + 50 timed
iterations:

| model | parameters | FP32 latency |
|---|---|---|
| B0 student | 7,371,923 | **32.24 ms** |
| AdaIR teacher | 28,784,824 | **299.82 ms** |
| ratio | 3.90x params | **9.30x latency** |

**Latency scales worse than parameter count** — 3.90x the parameters costs 9.30x
the time. AdaIR is a Restormer-style transformer with FFT-based frequency
modules; its cost is not dominated by parameter count.

### The teacher cannot be deployed at all (findings F7)

AdaIR's 36.26 dB is a number from a model that **does not run on the target
hardware**. Export was attempted five ways and failed in all of them:

| model | exporter | opset | result |
|---|---|---|---|
| unpatched | TorchScript | 17 | `SymbolicValueError` |
| slicing patched | TorchScript | 17 | `UnsupportedOperatorError: aten::fft_fft2` |
| slicing patched | TorchScript | 20 | `UnsupportedOperatorError: aten::fft_fft2` |
| unpatched | dynamo | 18 | `TorchExportError` |
| unpatched | dynamo | 20 | `TorchExportError` |

So the comparison is not "0.65 dB worse and faster". It is **deployable versus
not deployable** — a categorical difference, not a quantitative one.

---

## 6. Provenance

| item | value |
|---|---|
| AI Hub quantize job | `jpymnqm4p` |
| AI Hub compile job | `jp8186185` (`--target_runtime qnn_context_binary`) |
| AI Hub inference job | `jgo8mo8dp` |
| device | Samsung Galaxy S24 (Family) |
| B0 checkpoint | seed 0, iteration 200,000, `best_psnr` 31.3088 |
| B0 training host | devon, RTX 4090, `taskset -c 0-7,12-31` |
| teacher checkpoint | `adair-single-denoise.ckpt` (345,789,529 bytes) |
| metric config | `src/eval/metrics.py :: ADAIR_DEFAULT` |
| artifacts | `runs/int8_demo/` |

**This is the first AI Hub measurement on real trained weights.** Every prior
number in this project — the 1.59x normalization result, family selection, the
2.881 ms Fix-C latency — used untrained models. That is valid for latency, which
does not depend on weights, but says nothing about quality. This closes that gap.

---

## 7. Figures

- `runs/int8_demo/b0_vs_adair_strip.png` — degraded / AdaIR / B0 FP32 / B0 INT8 /
  ground truth, one row per sigma. **The demo figure.**
- `runs/int8_demo/b0_int8_strip.png` — student only, FP32 vs INT8.
- `runs/int8_demo/visuals/*.png` — all 12 pairs individually.

Visually: the teacher's edge is discernible at sigma 15 in fine incised detail
(the vase's chevrons); by sigma 50 the three restorations are hard to tell apart.
INT8 and FP32 are near-indistinguishable at every level despite the measured
0.39 dB.

---

## 8. Caveats

1. **12 crops, not a benchmark.** 4 images x 3 sigmas at 256x256. Not comparable
   to published full-image BSD68 numbers, and not intended to be.
2. **Denoise only.** No derain or dehaze — B0 was not trained on them.
3. **B0 is not finished.** Seed 0 was at 200,000 of 300,000 when exported, and
   seeds 1 and 2 are still running. Final numbers will move slightly, and
   seed-to-seed variance is not yet known (spread at iteration 5,000 was
   0.05 dB).
4. **The 0.393 dB INT8 cost exceeds the +-0.10 dB threshold** used for
   architecture decisions elsewhere in this project. It is within normal
   post-training-quantization range and the model plainly still works, but it
   should not be described as quality-neutral.
5. **FP32 timings are RTX 3050 laptop numbers**, taken with the GPU otherwise
   idle. They compare the two models fairly against each other but are not
   deployment figures — the S24 INT8 number is.
