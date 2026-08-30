# Student v3 — design

A student designed from **this project's own measurements**, not from a
teacher's architecture diagram. Every inclusion traces to a measured gap;
every exclusion traces to a measured non-gap or a refuted claim.

Code: [`src/models/student_v3.py`](../../src/models/student_v3.py) ·
Arm: `B0V3` ([`configs/train/b0v3.yaml`](../../configs/train/b0v3.yaml)) ·
Tests: [`scripts/smoke_student_v3.py`](../../scripts/smoke_student_v3.py)

## 1. The design principle

> Match the operator to the degradation's mathematical structure, and add
> **nothing** where the baseline already ties.

Measured per-task gap, current student vs its own GT-only baseline
([cond_regression.md](../kd_feature_multitask/cond_regression.md)):

| task | student | GT-only baseline | gap | action |
|---|---|---|---|---|
| denoise | 30.69 | 30.69 | **tie** | add nothing |
| derain | 36.07 | 36.83 | −0.76 dB | oriented operator |
| dehaze | 34.10 | 34.65 | −0.55 dB | global operator + physical prior |

Independently corroborated by a controlled backbone study
([arXiv:2310.11881](https://arxiv.org/abs/2310.11881)), which ran NAFNet /
SwinIR / Restormer on identical tasks:

| task | NAFNet | SwinIR | Restormer |
|---|---|---|---|
| derain (Test100) | 30.33 | 30.05 | **32.03** |
| dehaze (SOTS-indoor) | 38.97 | 29.14 | **41.87** |
| deblur (GoPro) | **33.08** | 31.66 | 32.92 |

NAFNet loses 1.7 dB / 2.9 dB on derain / dehaze but **wins** on deblur.
Their stated mechanism: depthwise convolution has "relatively weak spatial
mapping capability" vs attention, and derain/dehaze specifically need "the
ability to handle large-range or even global information," while denoising
is architecture-flexible. That is our pattern, independently reproduced,
with a mechanism attached — which is why we treat "global context" as the
real gap rather than "frequency."

## 2. What was added, and why

### Dehaze → physics + global context

**`dark_channel_prior`** (He, Sun & Tang, CVPR'09 / TPAMI'11) concatenated
as a 4th input channel. Koschmieder's atmospheric scattering model
`I = J·t + A·(1−t)` gives haze a known closed form; the dark channel prior
turns it into a zero-learned-parameter, **per-pixel** transmission
estimate. The network does not have to rediscover from data what physics
supplies directly. AdaIR has no equivalent — it learns haze removal purely
from data.

**`StripPoolingGate`** (Hu et al., CVPR'20) at the bottleneck and the first
decoder stage. Pools along one *full* spatial axis at a time — genuinely
global along that axis, not merely a larger local window — supplying the
"large-range information" capability the backbone study identifies as
missing, from `AveragePool`+`Conv` rather than attention.

### Derain → orientation

**`OrientedStreakGate`** at the two highest-resolution decoder stages. Rain
decomposition literature (Kang et al., TIP'12; Li et al., CVPR'16) models a
rainy image as `I = B + R` where `R` is sparse and **directionally
anisotropic**. Every conv in NAFBlock is a square, 4-fold-symmetric kernel
— structurally the wrong shape for a signal whose defining property is its
orientation. Freeman & Adelson (TPAMI'91) give the theory: a small basis of
oriented filters plus learned angle-dependent weights synthesises a
response at any orientation. Implemented as four fixed-orientation
depthwise kernels (0/45/90/135°, diagonals masked at init and held there by
a gradient hook) combined through an SE-style gate.

Placement is deliberate: streaks are 3–10 px structures that do not survive
to the bottleneck, so the operator sits where they are still resolvable.

### Denoise → nothing

We tie the baseline. Adding capacity here would risk exactly the regression
v1 demonstrated and would confound later ablations. **Restraint recorded as
a design decision.**

## 3. What was deleted, and why

**`LaplacianFrequencyGate` is not in the default path.** Its sole
justification was AdaIR's Table-7 claim that "frequency mining" is worth
**+1.58 dB**. We disproved that claim:

- AdaIR's frequency mask is **mathematically zero** at the resolution it
  trains at (`h // 128 * rate` with rate ≈ 0.5 truncates to 0 below 256 px;
  AdaIR trains on 128 px patches). The module degenerates to `torch.abs()`.
  Confirmed independently on 300 images
  ([test01](../../../teacher-experiments/test01)).
- Repaired properly (resolution-independent + differentiable) and retrained
  on **real** data with a control arm, full network unfrozen: the mask fix
  is worth **~0.00 dB**. Every arm lands within 0.02 dB of baseline.
- Two independent SOTA lines have since dropped forward-pass FFT: SFNet
  (ICLR'23) / FSNet (TPAMI) do "frequency selection" with **no `torch.fft`
  at all** — a learned low-pass filter, `high = input − low`, then a
  channel gate; EvoIR states outright that in its high-frequency branch
  "both FFT and IFFT are removed."

Keeping a module whose motivation we ourselves refuted would be incoherent.
It stays importable from `theory_blocks` for ablation only.

*(Note: SFNet's `dynamic_filter` is structurally the same design as our
`LaplacianFrequencyGate` — low-pass, residual high, gate — which is
reassuring about the derivation, but does not restore the motivation.)*

## 4. NPU constraint — empirically enforced

Targets are QNN (Hexagon), TFLite, TensorRT, checked by **real ONNX export**
through [`src/export/op_coverage.py`](../../src/export/op_coverage.py).
This methodology has already caught three bugs that reasoning alone missed:

1. `torch.min(dim=1)` + `−MaxPool(−x)` → lowers to `ReduceMin`/`Neg`,
   **absent from all three backend tables**. Rewritten to
   `min(a,b) = a − relu(a−b)` and `1 − MaxPool(1−x)`.
2. `.expand()` on a runtime shape → `Equal`/`Where`/`ConstantOfShape`, all
   UNKNOWN. Replaced by native `Add` broadcasting.
3. `adaptive_avg_pool2d` with a traced dynamic size → **export failure that
   only appeared once embedded in the full model**, not in the isolated
   probe. Replaced by `mean(dim=…, keepdim=True)`.

Banned with evidence: `torch.fft` (absent everywhere); attention —
[`scripts/probe_mdta.py`](../../scripts/probe_mdta.py) shows MDTA lowers to
`MatMul`/`Softmax`/`ReduceL2`, none of which appear in **any** curated
table (unverified rather than proven-bad, but not worth betting the
deployment target on).

**Known remaining risk, deliberately not changed here:** `LayerNorm2d`
decomposes into `ReduceMean`/`Pow`/`Sqrt`/`Div` — ~394 CAUTION-flagged ops,
by far the dominant NPU risk left, and this project's own findings F1 record
that *normalisation, not convolution*, dominates INT8 Hexagon latency. The
locked config already mitigates the worst of it (affine_clamp at full
resolution, earned through documented divergence debugging, F9). Replacing
the remaining deep-stage LayerNorm2d is the single largest NPU lever left,
but it is a **training-stability** change (F6: affine-everywhere diverged in
all four variants tried) and belongs in its own controlled experiment — not
bundled silently into an architecture change about restoration quality.

## 5. Verified properties

From [`scripts/smoke_student_v3.py`](../../scripts/smoke_student_v3.py),
run on devon:

| check | result |
|---|---|
| **No-confound identity**: all operators off ⇒ plain NAFNet | **max diff 0.0**, 7,371,923 params both, state_dict loads with 0 missing / 0 unexpected |
| Parameter cost | 7,447,331 (**+75,408, +1.02 %**) |
| Gradient flow | all **688** parameter tensors receive finite gradient |
| Operator placement | strip @ bottleneck + dec0; oriented @ dec2,3; DCP ⇒ 4-ch stem |
| Odd-size input (padding path) | (1,3,130,71) → (1,3,130,71) |
| Banned ops in exported graph | **none** (DFT, MatMul, Softmax, ReduceL2, ReduceMin, Neg, Expand, Where, Equal) |
| Whole-model op coverage | identical CAUTION/UNKNOWN categories to the shipped baseline — **zero new risk categories** |

The identity check is load-bearing: because operators-off v3 is *byte-identical*
to NAFNet, any PSNR delta B0V3 shows is attributable to the added operators
and not to an accidentally different backbone.

## 6. Why B0V3 is GT-only

Feature-KD has never helped on this protocol, and we now have two
independent measurements saying so:

- kd_feature_multitask: ties on denoise, **behind** on derain/dehaze.
- [test07_b](../../../teacher-experiments/test07_b) reproduced the KD
  **NO-GO with both teachers** — −0.79 dB (released) and −0.64 dB
  (frequency-fixed) vs the no-KD baseline, 3 seeds each.

Including a KD term would confound an architecture result with a
distillation term that has not yet worked. The intended comparison is
**B0V3 vs the B0V2 GT-only baseline, single-variable: architecture.**

## 7. Status

Architecture- and export-verified only. **Not yet trained** — no claim is
made here about restoration quality. The next step is the B0V3 vs B0V2
GT-only run at matched iterations (90 k), plus the three single-operator
ablations the toggles enable.

## References

1. He, Sun & Tang. *Single Image Haze Removal Using Dark Channel Prior.* CVPR 2009 / TPAMI 33(12) 2011.
2. Koschmieder 1924; McCartney, *Optics of the Atmosphere*, Wiley 1976.
3. Hu, Zhang, Xie & Yang. *Strip Pooling: Rethinking Spatial Pooling for Scene Parsing.* CVPR 2020.
4. Kang, Lin & Lin. *Automatic Single-Image-Based Rain Streak Removal via Image Decomposition.* IEEE TIP 2012.
5. Li, Tan, Guo, Lu & Brown. *Rain Streak Removal Using Layer Priors.* CVPR 2016.
6. Freeman & Adelson. *The Design and Use of Steerable Filters.* IEEE TPAMI 13(9) 1991.
7. Chen, Chu, Zhang & Sun. *Simple Baselines for Image Restoration* (NAFNet). ECCV 2022.
8. Cui, Zamir, Khan, Knoll, Shah & Khan. *AdaIR.* ICLR 2025. https://arxiv.org/abs/2403.14614
9. Cui et al. *Selective Frequency Network for Image Restoration* (SFNet). ICLR 2023 · *Image Restoration via Frequency Selection* (FSNet). TPAMI.
10. *A Comparative Study of Image Restoration Networks for General Backbone Network Design.* https://arxiv.org/abs/2310.11881
