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

**Limits:** one architecture family, one device, one runtime. n=12 is small and
the configs are heterogeneous; the correlation gap between MACs and the
normalization proxy should not be over-read. The matched pair is the reliable
part.

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
fallback). On its own terms the substitution was dramatic: **61.7% → 3.4%**.

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
