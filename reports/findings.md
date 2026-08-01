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

## F8. Batch-size policy: gradient accumulation to a fixed effective batch of 32

**Decision: effective batch 32 everywhere, reached by gradient accumulation when
the micro-batch does not fit.** Recorded here rather than only in the config
because it governs the comparability of every number in the project.

### The problem

The S-arm norm ablation (Q-A / Q-F / the Q-E ladder) ran at **batch 32, patch
128** on `w16_b8` — 2.13 GB peak, comfortable on a 6 GB card. The M arm
`w16_sidd` has **36 blocks against `w16_b8`'s 17**, roughly double the activation
memory, and **OOMs at batch 32**. Measured on the same GPU:

| config | blocks | batch 16 peak | batch 32 |
|---|---|---|---|
| `w16_b8` (S) | 17 | 1.08 GB | 2.13 GB — fits |
| `w16_sidd` (M) | 36 | 2.11 GB | **OOM** |

So the family cannot share a single native batch size.

### Why accumulation rather than dropping to batch 16 project-wide

Dropping to 16 would be simpler, but it would **retroactively change the
configuration under which Q-A (31.019 dB) and Q-F (31.014 dB) were measured**.
Those numbers are the entire basis of the normalization lock. Batch size affects
gradient noise and therefore training dynamics, so a batch-16 rerun is not
guaranteed to reproduce a 0.005 dB gap — and re-running the whole ablation to
find out costs ~9 GPU-hours to defend a decision already made on sound evidence.

Gradient accumulation keeps the effective batch at 32 for every arm, so:

* the existing S-arm numbers stay valid as measured, not merely re-labelled;
* S, M and L are trained under the same effective batch, so family comparisons
  are like-for-like;
* the only cost is wall-clock on arms that need accumulation (2 micro-steps of
  16 instead of 1 step of 32 — same samples, marginally more overhead).

### Exactness caveat, stated rather than glossed

Accumulation is **not bit-identical** to a true large batch. Batch-norm-style
statistics would differ, and gradient accumulation sums micro-batch gradients
whereas a single large batch averages over one forward pass. This architecture
uses no batch statistics (LayerNorm and affine only, both per-sample), so the
gradient is mathematically equivalent up to floating-point summation order. The
equivalence is therefore **exact in expectation and near-exact numerically**,
which is sufficient here — but it is an approximation, not an identity.

### The M spot-check is the exception

The `w16_sidd` N-A vs N-F spot-check ran at **native batch 16 on both arms**,
without accumulation. That is deliberate: it is a *relative* comparison between
two norms on one config, so what matters is that both arms share settings. Its
absolute PSNR is **not comparable to the S-arm numbers** and is not used as
such.

---

## F7. The AdaIR teacher cannot be exported at all — the deployment gap is categorical, not quantitative

Attempting to export the released AdaIR to ONNX (opset 17, fixed 256x256 input,
no quantization) **fails outright**:

```
SymbolicValueError: Rank must be 0 or 1, not 2
  in net.model.FreModule::fre1, model.py:349
```

The failure is in `FreModule` — AdaIR's frequency mining and modulation block,
i.e. its central contribution. Two independent reasons it cannot be traced:

**1. Data-dependent dynamic slicing** (`model.py:345-349`). The frequency mask is
built by slicing at bounds computed from a *learned* threshold:

```python
threshold = self.rate_conv(F.adaptive_avg_pool2d(x, 1)).sigmoid()
h_ = (h//n * threshold[i,0,:,:]).int()
mask[i, :, h//2-h_:h//2+h_, w//2-w_:w//2+w_] = 1
```

Slice bounds depend on tensor **values**, not shapes, so the graph is not static
and cannot be traced to a fixed ONNX graph. This is what raises.

**2. Complex-valued FFT** (`model.py:351-361`). `torch.fft.fft2` / `ifft2` on
complex tensors, which the standard opsets do not cover.

### Why this matters more than a missing benchmark

The intent was opportunistic — measure the teacher's on-device latency so
compression ratios have an honest denominator rather than a MAC count that
under-estimates the FFT work. That number is unobtainable, and its absence is
the more interesting result:

**The teacher is not slow on the target hardware. It cannot run on it at all.**

That converts the project's premise from quantitative to categorical. The
comparison is not "29M params at some latency versus 7M at a lower one" — it is
**deployable versus not deployable**. Gate G1 established that the student
exports, quantizes to INT8, compiles to a QNN context binary and runs with
**zero CPU fallback**; the teacher does not clear the first of those steps.

**Consequence for how results are framed.** Speedup factors against AdaIR are
necessarily *estimates from MACs*, and should be labelled as such — there is no
measured teacher latency to divide by, and there cannot be without rewriting
`FreModule` to remove the data-dependent slicing. That rewrite would change the
model, so the resulting number would not describe the published AdaIR.

### Due-diligence: both causes confirmed independently, exporter ruled out

The two causes above were initially *inferred from source* — only H1 was proven,
since the export died there before ever reaching the FFT. A follow-up probe
(`scripts/probe_adair_export.py`) separated them and also tested a second
exporter. All five attempts, verbatim:

| # | model | exporter | opset | result |
|---|---|---|---|---|
| A0 | unpatched | TorchScript | 17 | `SymbolicValueError: Rank must be 0 or 1, not 2` at `model.py:349` |
| A17 | slicing patched to a fixed mask | TorchScript | 17 | `UnsupportedOperatorError: aten::fft_fft2 ... not supported` |
| A20 | slicing patched to a fixed mask | TorchScript | 20 | `UnsupportedOperatorError: aten::fft_fft2 ... not supported` |
| B18 | unpatched | **dynamo** | 18 | `TorchExportError: Failed to export the model with torch.export` |
| B20 | unpatched | **dynamo** | 20 | `TorchExportError: Failed to export the model with torch.export` |

Environment: `torch 2.5.1+cu121`, `onnxscript 0.7.1`, CPU tracing, fixed
`(1,3,256,256)` input.

**Both hypotheses are now confirmed, not inferred.** Replacing the
value-dependent mask with a shape-derived one lets tracing proceed past H1, and
export then fails on `aten::fft_fft2` — at opset 17 *and* opset 20. So the FFT is
an independent blocker, not merely a suspected second one. And the dynamo
exporter fails on the unpatched model too, so this is not an artifact of the
TorchScript tracer.

> A methodological note worth keeping: the first run of this probe was
> **inconclusive in both arms** and looked like evidence. The patched-model arm
> failed on an assertion inside AdaIR (`model.py:190`) because the replacement
> dropped a `conv1` call that changes the channel count — my bug, not the
> model's — and the dynamo arm failed on a missing `onnxscript` dependency.
> Neither said anything about exportability. A probe that fails for the wrong
> reason reads exactly like a probe that fails for the right one.

**Remaining limit.** Two exporters and three opsets were tried. A custom symbolic
for the FFT, or an ONNX Runtime contrib/custom operator, could in principle
carry `fft_fft2` — but both require modifying or extending the model, so the
resulting artifact would no longer be the published AdaIR. The claim is that
**the released model as written does not export by any standard path available in
current PyTorch**, which is what matters for a deployment argument.

---

## F6. Removing normalization from NAFNet fails by activation-scale growth, not by gradient-spike instability

Training `w16_b8` with **affine-only normalization everywhere** (no statistics
anywhere in the network) diverges reproducibly. Three arms, three mitigations,
all dead:

| arm | mitigation | LR peak | warmup ends | diverged at | gap after warmup |
|---|---|---|---|---|---|
| Q-E | none | 1e-3 | 2000 | 3040 | 1040 |
| Q-E′ | half LR, 2x warmup | 5e-4 | 4000 | 4926 | 926 |
| Q-E″ | + grad clip (norm 1.0) | 5e-4 | 4000 | **4926** | 926 |
| Q-E‴ | + residual init 0.1 | 5e-4 | 4000 | 5899 | **1899** |

**The claim: gradient magnitude is bounded and *falling* at the point of
failure, which rules out gradient-spike instability. Divergence is driven by
unconstrained activation-scale growth through the unnormalized residual stack,
not by an update-magnitude event.**

Two independent lines of evidence converge on this.

**(a) Q-E″ and Q-E′ produced bit-identical trajectories.** Every logged value
matched exactly — loss `0.05443`/`0.03493`, PSNR `23.705`/`26.695`, gradient norm
`0.153`/`0.111` — and both died at the *same* iteration, 4926. Q-E″ differs from
Q-E′ only by an active gradient-clipping safety net. Floating-point arithmetic
guarantees that a single clip event forks the trajectories permanently. They
never forked, so **clipping fired exactly zero times**: no gradient ever reached
the 1.0 threshold. This eliminates the entire spike-driven hypothesis class
without needing to instrument the failing step.

**(b) The gradient norm falls, rather than rises, into the failure.** 0.153 at
it 2000 → 0.111 at it 4000 → dead at 4926. A spike-driven collapse shows
gradients climbing beforehand; this shows the opposite.

**Why halving the learning rate only postponed it.** Q-E and Q-E′ both died
roughly 1000 iterations *after warmup completed*, whatever the peak LR. The
failure tracks a **step budget past full LR**, not an LR magnitude — consistent
with residual branches growing to a critical scale after a roughly fixed number
of full-rate updates, and inconsistent with an instability threshold in LR.

**Mechanism.** NAFNet's `beta`/`gamma` residual scales initialise to zero, so
every block starts as an exact identity and the network is trivially stable
early. Nothing in the affine-only variant bounds activation magnitude once those
scales become non-trivial: across 17 blocks the residual stream compounds freely
until the forward pass produces a non-finite value. LayerNorm's role here is
therefore **load-bearing for trainability**, not merely a quality refinement —
which is a stronger statement than "normalization helps."

**A prediction that failed, and what it revises.** We expected Q-E‴ (residual
scales initialised to 0.1 instead of 0) to die *earlier*, on the reasoning that
zero-init buys a "protection window" of early stability which starting active
removes. It died **later** — 1899 iterations past warmup versus 926, roughly
double the survival.

So the protection-window framing is wrong. A better reading: from zero, the
residual scales receive a large uncontrolled gradient push and grow rapidly;
starting them at 0.1 gives the optimiser usable signal to *regulate* them from
the first step, slowing the growth. This still fits activation-scale growth as
the mechanism — it changes the *rate*, not the endpoint — but it means the
relevant quantity is how fast `beta`/`gamma` grow, not how long they stay near
zero. **Recorded as a revision rather than folded quietly into the original
story.**

**Practical consequence.** The 2.34x speedup available from removing
normalization entirely is not reachable by tuning optimisation: four arms, three
distinct mitigations (LR, clipping, initialisation), all diverged. It would
require an architectural mechanism that bounds activation scale without
per-element statistics — weight normalisation, a hard cap on residual scale, or
a fixed (non-learned) scale. Out of scope here; recorded under future work.

That said, the delay Q-E‴ bought is the one positive signal: **initialisation of
the residual scales measurably controls the failure rate**, so a scheme that
constrains `beta`/`gamma` growth directly is the most promising direction if
norm-free training is ever revisited.

**Limits.** One architecture (`w16_b8`), one task (denoising), one seed, 30k
iterations. The bit-identical-trajectory argument is exact and does not depend
on sample size; the step-budget observation rests on three runs and should be
treated as strongly indicative rather than established.

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

## F9. N-F's unnormalized full-resolution stage turns a rare input into a run-killing gradient

**B0 died at optimizer step 24356 with a gradient norm of 6.5e7.** Deterministic
— two resumes from the same checkpoint reproduced it bit-identically, including
`maxgn 65240022.433` to every printed digit — so not hardware.

### The mechanism, end to end

One crop out of 32 does it. At the spike, through the same weights:

| | loss | output range |
|---|---|---|
| samples 0-11, 13-15 | 0.013-0.030 | ~[0, 1] |
| **sample 12** | **1864.64** | **[-471206, +705137]** |

Sample 12 is a dark, low-variance crop (std 0.084 against a typical 0.25, 10.65%
exact zeros) — unusual, but not corrupt, not saturated, no non-finite values.

Stage activations for that sample show exactly where it goes wrong:

| enc0 | enc1 | enc2 | enc3 | middle | dec0 | dec1 | dec2 | **dec3** |
|---|---|---|---|---|---|---|---|---|
| 0.06 | 0.07 | 0.15 | 0.36 | 23.5 (max 970) | 16.4 | 8.2 | 4.8 | **2821 (max 5.6e6)** |

The encoder is fine. The middle stage runs hot — and **Q-A does that too**, so
that part is depth-general, not an N-F artifact. The decoder walks it back down.
Then `dec3` explodes by five orders of magnitude.

**`dec3` is the full-resolution decoder stage — precisely where N-F replaces
LayerNorm2d with affine.** It is the one stage with no normalization to bound a
large-but-finite input. Q-A, which keeps LayerNorm there, saw spikes of 61, 293
and 135 over 45k iterations on the identical schedule and **survived every one**,
improving monotonically to PSNR 31.214.

Three things had to coincide: a depth-general hot middle stage, a rare
low-variance sample, and an unnormalized output stage. That is why 30k
iterations of ablation never surfaced it.

### Ruled out, with evidence

- **Not precision.** fp32 reproduces it: grad norm 1.2e6, loss 116.6 on the same
  batch and weights. bf16 amplifies ~54x but does not cause it.
- **Not the residual gates.** `beta` max 0.551, `gamma` max 0.432 (means 0.18 /
  0.16) — modest. Extra weight decay on them would not have helped.
- **Not gradient clipping's absence.** `grad_clip` 8.0 -> 1.0 moved death from
  step 25582 to 28654. Clipping bounds the step, it does not stop the model
  reaching a state where a normal input produces a catastrophic output. It is
  also not a guard against Inf: `clip_grad_norm_` scales by
  `max_norm / (total_norm + eps)`, which for an Inf norm is ~0, and `inf * 0 =
  nan` — clipping actively CONVERTS an Inf gradient into NaN weights.

### The fix, measured rather than assumed

Same spike-state weights loaded into each candidate (LayerNorm2d, AffineNorm2d
and AffineClampNorm2d share parameter shapes, so this is a direct comparison):

| variant | sample 12 max\|out\| | healthy max\|out\| |
|---|---|---|
| N-F (affine) | 705,100 | 1.048 |
| Fix-A (restore LayerNorm at full-res) | 15.71 | 11.69 |
| Fix-C clamp 64 | 667.5 | 1.048 |
| **Fix-C clamp 8** | **17.49** | **1.048** |
| Fix-C clamp 2 | 2.572 | 1.048 |

**Fix-C (`affine_clamp`) at bound 8 matches Fix-A's containment while leaving
healthy outputs unchanged**, because a clamp is inert until it engages, whereas
restoring LayerNorm changes the function the weights were trained under. A clamp
is also *stronger* in principle: it bounds unconditionally, where normalization
rescales by measured statistics that a sufficiently pathological input can still
evade. Latency on-device is still to be measured — `Clip` is a first-class
quantized Hexagon op and usually fuses, but that is a claim to verify, not
assume.

### Process lessons

**A per-step trace that logs only the last micro-batch is blind to the anomaly
it exists to catch.** With `accum_steps: 2` the trace recorded micro-batch 1's
loss (0.019) and never micro-batch 0's (7931). This produced a confidently wrong
intermediate conclusion — "loss is normal, the gradient explodes anyway" — which
pointed the whole investigation at precision. Log every micro-batch, or the
worst across them; never the last.

**A function can be unreachable dead code behind a green test suite.** The fp32
recheck was added, tested, and shipped — but its call site never landed, because
a string replacement silently failed to match. 179 tests passed and the feature
was entirely absent; a 30-minute diagnostic run produced no measurement. Every
test asserted the artifact produced, none asserted the invocation fired. Write
the call-site test.

**Anomaly thresholds chosen by intuition miss the anomaly.** The per-sample
screen flagged `std < 0.01` and saturation `> 50%`. Sample 12 has std 0.084 and
10.65% zeros, so it passed as clean and was reported as such. The batch was
declared normal twice before per-sample forward losses found it.

**The default-argument trap.** `AffineClampNorm2d` took
`bound: float = AFFINE_CLAMP_BOUND`. Python evaluates default arguments **once,
at `def` time**, so reassigning the module constant afterwards had no effect —
a sweep over bounds 64/16/8/4/2 returned an identical 667.5 at every setting.
Read as "the clamp cannot contain this", it would have rejected the fix that
actually works. Any value that is meant to be overridable must be read at call
or construction time, never bound as a default. This generalises: mutable and
module-level defaults are captured at definition, not at use.

### The trigger cannot be characterised from input statistics — which is why the fix must be structural

Against the **exact spike-state weights**, none of the deliberately worst-case
inputs reproduced the failure:

| input | max\|out\| under N-F |
|---|---|
| all-black | 0.031 |
| all-white | 1.047 |
| near-zero-variance (std 0.002) | 0.049 |
| dark, low-variance (mean 0.055, std 0.084) | 0.049 |
| extreme / clipped noise | 1.374 |
| half-saturated | 1.058 |
| real crop, std **0.001** | 0.026 |
| real crop, std 0.006 | 0.025 |
| **the actual sample 12** | **705,100** |

Sample 12 is *not* the most extreme input by any statistic measured — several
synthetic and real crops are far more degenerate and pass cleanly. Whatever
distinguishes it is a property of the interaction between that specific image
content and that specific weight state, not a summary statistic of the input.

**Three consequences, and they drive the fix decision:**

1. **Input-property filtering cannot be relied on as a defence.** There is no
   threshold on mean, variance, saturation or dynamic range that would have
   caught this sample without also rejecting large numbers of healthy ones.
   Data filtering is therefore admissible only as supplementary insurance,
   never as the primary fix.
2. **Other trigger classes cannot be ruled out.** One trigger was found because
   it happened to fire during a 300k run. The absence of a characterisation
   means there is no basis for claiming it is the only one — a bright or
   high-contrast sample could hit the same unprotected stage.
3. **Therefore the fix must close the mechanism, not the instance.** Bounding
   `dec3`'s magnitude structurally is robust to triggers nobody has enumerated;
   filtering the one known sample is not.

A stress suite built only from imagined worst cases would have certified N-F as
safe. `scripts/stress_test_norm.py` therefore always includes the captured spike
batch, and its synthetic cases are treated as a smoke test rather than as
evidence of safety.

### Exposure window — a general caution about the 1.5b protocol

**A short ablation can establish that a quality difference is negligible while
saying nothing at all about rare catastrophic failure modes.** The norm lock
rested on 30k iterations on the S arm (-0.005 dB) and 10k on M (-0.006 dB). Both
conclusions remain valid *as quality statements*. Neither had any power to detect
a failure that needs ~24k iterations and a 1-in-thousands sample to surface —
and B0 is a 300k run, i.e. more than an order of magnitude more exposure than
the evidence that licensed it.

This applies to every decision taken on a short ablation in this project, not
only the norm lock. The mitigation is not longer ablations, which are too
expensive: it is an adversarial stress pass, which costs seconds and would have
caught this before B0 launched.

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

**Calibration caveat — CLOSED (2026-07-31).** The measurements above used
placeholder uniform-noise calibration. Latency is weight-independent, but
quantization *ranges* derive from calibration and could in principle influence
kernel selection, so this needed confirming rather than assuming.

Re-profiled `w16_b8` N-A with **real calibration data** — 8 genuine degraded
inputs spanning all three tasks (noisy BSD68, rainy Rain100L, hazy SOTS),
centre-cropped to the export resolution:

| calibration | INT8 latency | layers | fallback |
|---|---|---|---|
| placeholder (uniform noise) | 2.510 ms | 637 | 0 |
| **real degraded images** | **2.507 ms** | 637 | 0 |

**Difference 0.003 ms (0.1%)** — within run-to-run variation, with an identical
layer count and compute-unit split. Calibration content does not affect latency
or kernel selection on this backend. Jobs: quantize `jgk8yvr2g`, compile
`jp81728x5`, profile `jg9d40dl5`.

Every latency number in this repository therefore stands as measured. Note this
says nothing about INT8 *accuracy*, which does depend on calibration and is not
measured anywhere yet.

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
