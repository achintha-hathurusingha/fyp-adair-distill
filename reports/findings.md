# Findings

Standing results with their evidence. Each is stated as a claim, with the
measurement that supports it and the limits of that measurement.

**Measurement context for all on-device results below**
Qualcomm AI Hub · Samsung Galaxy S24 (Snapdragon 8 Gen 3, Hexagon v75) ·
INT8 QNN context binary (`--target_runtime qnn_context_binary`) ·
weights and activations `QuantizeDtype.INT8`, per-channel weights ·
input `(1, 3, 256, 256)` fixed · untrained weights (latency is
weight-independent) · placeholder uniform-noise calibration ·
measured 2026-07-30 · `qai-hub` 0.53.0, `torch` 2.5.1, ONNX opset 17.

---

## F1. Normalization dominates INT8 restoration latency on Hexagon; convolution does not

On `w16_b8` (2.44M params, 2.13 GMACs), the fused LayerNorm accounts for
**~62% of NPU cycles** against **~3.4% for all convolutions**. Total 2.51 ms,
637 layers, **zero NPU→CPU fallback**.

The cost is per-element, so it scales with the spatial resolution at which each
normalization runs — the four full-resolution norms alone are ~40% of runtime.

**Consequence:** neither parameter count nor MAC count predicts on-device
latency for this model class. Across the 12-config sweep (n=12), correlation
with measured latency was **GMACs r=0.66**, **block count r=0.75**,
**normalization-area proxy r=0.87**.

**Strongest single piece of evidence** (controlled, not correlational):
`w32_b28` and `w32_sidd` have *identical* block counts (36) and MACs within
0.6% (15.96 vs 16.05 GMACs), yet differ **38% in latency** (4.47 vs 6.16 ms).
The only difference is *where* blocks sit in the pyramid, and therefore how many
normalizations run at full resolution.

**Direct confirmation by intervention (not just observation).** Removing
normalization from only the two full-resolution stages of `w16_b8` — **4 of 34
normalizations, 11.8% of them** — cuts latency 2.510 → 1.572 ms. Removing *all
34* reaches 1.069 ms. So those 4 carry **65% of the entire removable
normalization cost**, a 5.5× over-representation, exactly as per-element
scaling predicts. This is a controlled intervention on one architecture, which
makes it considerably stronger than the correlational evidence above.

| variant | normalization | ms | vs N-A | norm % of cycles |
|---|---|---|---|---|
| N-A | `LayerNorm2d` everywhere | 2.510 | 1.00× | 62.3% |
| N-F | affine at full resolution only | 1.572 | **1.60×** | 36.9% |
| N-E | affine everywhere (floor) | 1.069 | **2.35×** | 1.5% |

**Limits:** one architecture family, one device, one runtime. n=12 is small and
the configs are heterogeneous; the correlation gap between MACs and the
normalization proxy should not be over-read. The matched pair and the N-F/N-E
intervention are the reliable parts. The area-weighted proxy predicted 72% for
the full-resolution share against 65% measured — good enough to rank
architectures, not good enough to be a cost model.

**These are latency results only.** N-F and N-E change the computed function and
their quality cost is **unmeasured** until Task 1.5b.

---

## F2. NAFBlock MACs are invariant to pyramid depth

Each downsample quarters `H·W` while doubling `C`. A NAFBlock's cost is
dominated by 1×1 convolutions scaling as `c²·H·W`, so the 4× increase in `c²`
cancels the 4× reduction in spatial area **exactly**. A NAFBlock therefore costs
the same MACs at every depth.

Measured: `b28` and `sidd` have the same total block count (36) and MACs within
1% at every width, despite `sidd` carrying ~1.7× the parameters. GMACs/block is
constant within a width (e.g. 0.44–0.48 at width 32).

**Consequence:** MACs track *total block count × width²*; block placement is
MAC-free but parameter-expensive. Depth buys capacity at zero MAC cost. The real
cost of full-resolution blocks is normalization and memory bandwidth, not
arithmetic.

**Design principle following from F1 + F2:** *push work deep, push normalization
out of full resolution.*

**Untested caveat — this is a speed argument, not a quality one.** Restoration
may require full-resolution processing to recover high-frequency detail, so
deep-heavy configurations like `b28` may trade PSNR for latency. Task 1.5b
measures this; until then the principle is unvalidated on quality.

---

## F3. Per-op cycle attribution is misleading in fused graphs — verify before optimizing

**This finding invalidated our own first conclusion, and cost a profiling
cycle.**

The `w16_b8` profile showed `Div` at 61.7% of cycles, which reads naturally as
"fixed-point division is expensive on the Hexagon integer pipeline." That
reading is **wrong**. Inspecting the nine nodes of a single LayerNorm:

```
ReduceMean=0  Sub=0  Pow=0  ReduceMean=0  Add=0  Sqrt=0  Div=1,837,518  Mul=0  Add=16,239
```

Eight of nine nodes report **zero cycles**. QNN fuses the entire normalization
subgraph into one kernel and attributes the whole cost to the terminal node.
The 61.7% is the cost of *the fused LayerNorm*, not of division.

**Rule:** before optimizing an op that dominates a profile, check whether its
neighbours report ~0 cycles. If they do, you are looking at a fused subgraph and
the expensive "op" is a label, not a target.

**Corollary — the canonical form is the fast path.** Any rewrite that breaks the
fusion pattern match loses, even when it is arithmetically cheaper. See F4.

---

## F4. `torch.rsqrt` does not lower to `Rsqrt`, and rewriting normalization to avoid division is a net loss

Two separate practical results.

**(a) Exporter gotcha.** ONNX has no native `Rsqrt` in standard opsets, and the
lowering is not what you would guess:

| PyTorch spelling | ONNX lowering (opset 17 **and** 20) |
|---|---|
| `torch.rsqrt(v)` | `Sqrt` → `Div(const, ·)` → `Mul` — **keeps the `Div`, adds a `Mul`** |
| `torch.reciprocal(torch.sqrt(v))` | `Sqrt` → `Reciprocal` — **`Div` count drops to zero** |

Mathematically identical, materially different graphs. The obvious spelling is
the worse one. Pinned by a regression test so it cannot be "simplified" back.

**(b) The rewrite still loses on device.** Replacing division-by-broadcast with
reciprocal-then-multiply (N-A′) is mathematically identical — `allclose`
`atol=1e-5`, checkpoints interchangeable, so it needs no retraining. It removed
every elementwise norm `Div`, and `Reciprocal` is fully supported (zero
fallback).

On its own terms the substitution was dramatic, but two different percentages
are in play and they must not be confused — **both are shares of that variant's
own total cycles**, and the totals themselves differ (17.8M vs 20.3M):

| quantity | N-A | N-A′ |
|---|---|---|
| the single divide-family op (`Div` → `Reciprocal`) | **61.7%** of 17.8M | **3.4%** of 20.3M |
| *all* normalization layers combined | **62.3%** of 17.8M | **67.6%** of 20.3M |

So the op itself became ~18× cheaper as a share, while normalization *as a
whole* got **more** expensive — because the work did not disappear, it moved out
of the fused kernel and into its neighbours.

**But total latency got 14% worse: 2.51 ms → 2.87 ms** (17.8M → 20.3M cycles,
637 → 671 layers). Because the rewrite broke QNN's LayerNorm fusion match, the
primitives that previously cost zero began executing separately:

| op | N-A (fused) | N-A′ (unfused) | Δ cycles |
|---|---|---|---|
| `Div` | 10,995,309 | 0 | −10,995,309 |
| `Sub` | 0 | 4,349,858 | +4,349,858 |
| `ReduceMean` | 0 | 4,179,885 | +4,179,885 |
| `Mul` | 2,436,125 | 4,870,584 | +2,434,459 |
| `Pow` | 0 | 1,866,775 | +1,866,775 |
| `Reciprocal` | 0 | 697,669 | +697,669 |

**Conclusion: there is no free graph-rewrite win available for normalization
here.** The only lever is *removing* normalization work (affine-only, identity,
or resolution-selective), not re-expressing it. **N-A′ is rejected** and retained
in the codebase only as a documented negative result.

**A hypothesis worth stating, and its status.** The appealing general rule was:
*division by a broadcast denominator should become reciprocal-of-the-small-tensor
followed by broadcast multiply, saving in proportion to the broadcast factor.*
The arithmetic is sound — `Reciprocal` ran on `C×` fewer elements and its own
cost fell 15.7×. **But on a fused backend the rule is false in practice**,
because the saving is smaller than the fusion it forfeits. It may still hold on
backends that do not fuse normalization, or for divisions outside a
fusion-matched pattern. **Untested.**

---

## F5. A 2025 paper's evaluation code cannot run on a 2026 environment, and one failure mode is silent

AdaIR (ICLR 2025) pins `scikit-image==0.19.3` and `scikit-video==1.1.11`
(`env.yaml:221-222`). On a current environment (scikit-image 0.24, NumPy 2.x)
its evaluation path fails two ways:

1. **Loud.** `utils/val_utils.py:5` does `from skvideo.measure import niqe` at
   *module scope*. `scikit-video` is unmaintained and does not install cleanly on
   NumPy 2.x, so the import kills the module before any metric runs — even though
   `niqe` is never used by the three tasks in the protocol.

2. **Silent, and the dangerous one.** `utils/val_utils.py:62` calls
   `structural_similarity(..., multichannel=True)`. That argument was deprecated
   in scikit-image 0.19 and **removed in 0.23**. It is not rejected: it is
   swallowed by `**kwargs` and ignored, leaving `channel_axis=None`, so a
   `(H, W, 3)` image is treated as a **3-D volume** and a volumetric SSIM is
   returned instead of the mean of per-channel 2-D SSIMs.

   In our case it happened to raise (`win_size exceeds image extent`, because the
   channel axis is 3 < 7). **That was luck, not safety** — for a tensor whose
   channel count is ≥ `win_size`, or with an explicit `win_size`, it returns a
   plausible wrong number with no warning.

**Why this matters beyond AdaIR.** The AirNet / PromptIR / AdaIR lineage shares
this evaluation code. Anyone reproducing any of it on a modern environment risks
SSIM values that are silently non-comparable with the published tables — the
exact failure that convention-locking is supposed to prevent.

**Practical rules adopted here:** never rely on a library default that has
already changed once — pass `channel_axis=-1` explicitly and pin it with a
known-answer test; and run the original artifact in its *own* pinned environment
rather than a patched modern one, so "reproduces" means the released code, not
our edit of it.

### Rebuilding the pinned environment is itself non-trivial (2026-07-31)

Reconstructing AdaIR's `env.yaml` to run its code unmodified surfaced three
further obstacles, none of them in AdaIR's own source:

1. **`lightning==2.0.1` will not import.** It eagerly loads `lightning.app`,
   which imports `lightning_cloud`, which at *import time* reads the Windows
   certificate store and dies with
   `ssl.SSLError: [ASN1: NOT_ENOUGH_DATA]`. The pinned `lightning-cloud==0.5.32`
   reproduces it. Nothing in the restoration code path needs any of this;
   `test.py` uses Lightning only for `LightningModule.load_from_checkpoint`.
   Resolved by moving to `lightning==2.2.5`, which no longer forces the
   `lightning.app` import. **This is a deviation from the pin** and is recorded
   as such — the *research code* is unmodified, one *dependency* is not.
2. **Pinned transitive versions conflict with each other.** Installing
   `lightning-utilities==0.8.0` (as pinned) pulls a `pytorch-lightning` that
   requires `>=0.10.0`. The published environment is not self-consistent under a
   current resolver.
3. **`test.py` hardcodes `.cuda()`** (lines 61, 85, 136) with no CPU fallback, so
   reproduction additionally requires a CUDA build of the pinned torch 1.13.1,
   not merely the pinned version.

The *metric* half of the environment — NumPy 1.23.5, scikit-image 0.19.3,
scikit-video 1.1.11, torch 1.13.1 — rebuilt cleanly and was sufficient to
generate the cross-implementation oracle in `tests/golden/adair_metrics.json`.

**Reading:** the reproducibility cost of a two-year-old paper sits mostly in its
dependency graph, not its research code. A frozen container image at publication
time would have prevented all three.

---

## AI Hub job IDs (verifiable provenance)

Every on-device number in this repository traces to a job below. Job pages are
at `https://workbench.aihub.qualcomm.com/jobs/<id>/`. All on
**Samsung Galaxy S24 (Family)**, INT8, `(1,3,256,256)`, measured 2026-07-30.

### Architecture sweep (Task 1.5a)

| model | quantize | compile | profile | ms |
|---|---|---|---|---|
| `w16_b8` | `jp06l9enp` | `jgd2o4kl5` | `j5w4qr66g` | 2.513 |
| `w16_b14` | `jgk830dng` | `jp43v4zl5` | `jpv7vnkjp` | 2.742 |
| `w32_b8` | `jgd2o46k5` | `jpxxyrv9p` | `jpv7vnwkp` | 3.160 |
| `w24_b8` | `jp06l960p` | `jp16e1rl5` | `jgjqe8nx5` | 3.195 |
| `w16_b28` | `j5w4qz8mg` | `jp2eld2mp` | `j56wn020g` | 3.299 |
| `w24_b14` | `j56wnmwng` | `jgd2o4jl5` | `jgd2omee5` | 3.519 |
| `w32_b14` | `j5qv3117g` | `j5m83krqp` | `j5w4qrmmg` | 3.590 |
| `w24_b28` | `jgo83wr1p` | `j579xnzrg` | `jpeykn4v5` | 4.256 |
| `w32_b28` | `j5qv31x7g` | `jgnk3q2mg` | `jp06lxj0p` | 4.469 |
| `w16_sidd` | `jgnk3q7jg` | `jgjqeyj85` | `jp43v2d85` | 4.742 |
| `w24_sidd` | `jgd2o4dk5` | `jp43v4ql5` | `jp16emd75` | 6.091 |
| `w32_sidd` | `jgnk3q4mg` | `j579xqdvg` | `jp36eyq3p` | 6.158 |

### Normalization sweep (Task 1.5c), all on the `w16_b8` skeleton

| variant | quantize | compile | profile | ms |
|---|---|---|---|---|
| N-A (reference) | `jp06l9enp` | `jgd2o4kl5` | `j5w4qr66g` | 2.513 |
| N-A′ (rejected) | `jgo83ylkp` | `jg9dwvvq5` | `jp43vonl5` | 2.869 |
| N-F | `jp2el6l4p` | `jgll3z38g` | `jgnk3yerg` | 1.572 |
| N-E (floor) | `j5m8323wp` | `jp06lql6p` | `jpeyk1k15` | 1.069 |

> The N-A row is the same job as `w16_b8` above — the sweep's baseline *is* the
> normalization reference, not a re-run.

**Outstanding caveat:** all of the above used **placeholder uniform-noise
calibration**. Latency is weight-independent, but quantization *ranges* derive
from calibration and could in principle influence kernel selection. To be
re-confirmed with real calibration images once datasets land (Task 2.3); no
change expected.

---

## Future work (parked — out of scope for this project)

Per the scope guard: this is a distillation project, not an NPU optimization
project. Recorded and deliberately not pursued.

- **Second backend.** Division is cheap on GPU, and TensorRT/Jetson fuses
  differently. If the optimal architecture proves backend-dependent, that is a
  stronger claim than "normalization is slow." Not run.
- **Measured teacher latency.** All compression ratios currently use a MAC count
  that under-estimates AdaIR (the counter does not model its FFT frequency
  modules). A measured AdaIR latency on the same S24 would be the honest
  denominator — and a compile failure would itself be a publishable result
  ("the teacher cannot deploy"). Not run.
- **Fusion-aware architecture search.** F3 implies the search space should be
  expressed over *fused kernels*, not ONNX ops.
- **`DepthToSpace` / `Slice`.** Flagged CAUTION statically at G1 but never
  observed to fall back; alternatives (transposed conv, resize+conv) untested.
