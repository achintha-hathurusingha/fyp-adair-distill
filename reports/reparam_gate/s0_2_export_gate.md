# S0.2 — NPU export gate for the reparameterizable oriented-kernel block

**Status: PASS.** Ran 2026-08-31 on devon. Script: `scripts/probe_reparam_oriented.py`.
Artefacts: `runs/reparam_oriented_{train,merged,merged_block,merged_int8}.onnx`.

**Kill criterion (plan S0.2): any UNKNOWN op surviving reparameterization.**
None survives, in FP32 or INT8, on any of the three backends. Phase 3 is not
blocked by op coverage.

---

## What was gated

A block that gives frequency-selective behaviour *spatially*, because
`torch.fft` has no ONNX op and attention is UNKNOWN on all three NPU backends.
Structural reparameterization (RepVGG, RepLKNet): train a multi-branch bank,
merge every branch algebraically into one convolution for deployment. The merge
is exact only if the branches combine **linearly** — that is the binding design
constraint, and it is what distinguishes this block from the existing
`OrientedStreakGate` (`scripts/probe_oriented_filter.py`), which puts a Sigmoid
channel gate between the bands and the fuse and therefore **cannot** be merged.

Branch bank, all depthwise, all linear, all zero-padded, learnable per-channel
band coefficients:

| branch | training-time form | support |
|---|---|---|
| 0° | separable `(1,k)` long-axis then `(kp,1)` short-axis | kp × k |
| 90° | separable `(k,1)` then `(1,kp)` | k × kp |
| 45° / 135° | k × k depthwise, gradient-masked to a diagonal band | diagonal band |
| isotropic | kp × kp depthwise | kp × kp |
| identity | delta kernel (repo's zero-init-residual idiom) | 1 × 1 |

---

## Results

### Op coverage (qnn / tflite / tensorrt)

| graph | op histogram | UNKNOWN | CAUTION |
|---|---|---|---|
| training (multi-branch) | `Add`×6, `Conv`×8, `Identity`×5, `Mul`×6 | `Identity` | none |
| **deployment (merged core)** | **`Conv`×1** | **none** | **none** |
| deployment (merged block, + 1×1 fuse) | `Add`×1, `Conv`×2 | none | none |
| **deployment INT8 (QDQ, per-channel)** | `Conv`×1, `QuantizeLinear`×2, `DequantizeLinear`×4 | **none** | **none** |

The merged core graph is **one node**: a single depthwise `Conv`. That is what
S3.2 asks for, reached here already.

`Identity` in the training graph is an export artefact of the per-channel
coefficient multiply; it is a no-op, any simplifier removes it, and the training
graph is never shipped. It is recorded rather than hidden.

### Merge exactness (this is S3.2's gate, done early)

Without it, exporting the merged graph would prove nothing about the trained
model. Parameters randomised well away from init so the test has teeth.

| condition | max &#124;multi-branch − merged&#124; |
|---|---|
| no BatchNorm, 9×9 and 32×32 inputs | 9.54e-07 |
| per-branch BatchNorm folded (RepVGG recipe) | 9.54e-07 |
| ORT(training graph) vs ORT(merged graph) | 1.19e-06 |
| eager PyTorch vs ORT(merged graph) | 1.19e-06 |

All ≪ the 1e-5 threshold; residual is float32 rounding.

Two things this deliberately checks that a naive test would miss:

- **9×9 inputs with k=7.** Almost every pixel is a boundary pixel. A merge that
  assumed interior-only equivalence would break here. It does not: with zero
  padding, padding commutes through the separable pair, so the composition of
  `(1,k)` then `(kp,1)` is exactly the single k×k kernel *including* at
  boundaries. The equivalent kernel is the **full linear convolution** of the
  two branch kernels, not their cross-correlation — PyTorch's `conv2d` is
  cross-correlation, so the merge flips one kernel.
- **ONNX Runtime parity, not just PyTorch parity.** The PyTorch merge being
  exact does not prove the *exported* graph is. Checked separately.

**Per-branch BatchNorm folds exactly at eval.** So the RepVGG training recipe is
available to S3.1 at zero deployment cost, should it be wanted. (It is not in
the repo's NAFNet-style idiom, so this is an option, not a recommendation.)

### Design-box sweep

The convolution-theorem result says 7×7–11×11 kernels reproduce the full optimal
frequency filter, and the plan contemplates w16 → w32. The merge and the op set
hold across that whole box — merged graph is `Conv`×1 with no UNKNOWN at every
point:

| dim | k | max &#124;diff&#124; | training params | merged params | collapse |
|---:|---:|---:|---:|---:|---:|
| 16 | 7 | 9.54e-07 | 2,128 | 800 | 2.66× |
| 16 | 9 | 9.54e-07 | 3,216 | 1,312 | 2.45× |
| 16 | 11 | 2.38e-06 | 4,560 | 1,952 | 2.34× |
| 32 | 7 | 1.43e-06 | 4,256 | 1,600 | 2.66× |
| 32 | 9 | 1.67e-06 | 6,432 | 2,624 | 2.45× |
| 32 | 11 | 2.38e-06 | 9,120 | 3,904 | 2.34× |
| 64 | 7 | 1.19e-06 | 8,512 | 3,200 | 2.66× |
| 64 | 9 | 1.91e-06 | 12,864 | 5,248 | 2.45× |
| 64 | 11 | 2.86e-06 | 18,240 | 7,808 | 2.34× |

Deployment cost is ~2.3–2.7× *below* training cost — the intended asymmetry.

### Other checks

- Identity at init (zero-init residual fuse), gradients finite.
- Diagonal masks hold after an SGD step: max off-band tap exactly 0.0 at both
  45° and 135°. The gradient hook is load-bearing; without it the "oriented"
  branches would drift into generic square kernels, which is precisely AdaIR's
  failure (AFLB3 α=0.496, β=0.497).

---

## What this gate does **not** establish

Stated so it is not over-read later.

1. **`op_coverage` sees only `Conv`.** It does not distinguish depthwise
   (`groups=dim`) from dense, and does not see kernel size. A merged 7×7 or 11×11
   depthwise conv is a different INT8 proposition from a 3×3 dense one, and
   backend kernel-size limits are real. **Real Qualcomm AI Hub conversion (S4.4)
   is the ground truth**; this is a static signal, as the module's own docstring
   says.
2. **No latency number.** The merged block is one depthwise conv plus one 1×1,
   but the FLOP delta against whatever it replaces in StudentV3 depends on
   placement, which is S3.3. That number should be measured, not estimated —
   `dynamic_conv` measured 4,059 ms against 54–139 ms for its neighbours, so
   static intuition about this hardware has been wrong before by ~30×.
3. **Nothing about accuracy.** Whether the orientation machinery earns its place
   at all is S0.1's oracle ceiling, and the linear ceiling is known to be low
   (+0.68 dB dehaze / +0.87 dB derain for the *full* optimal linear filter).
   This gate says the design is *deployable*, not that it is *worth deploying*.
4. **Calibration is random noise**, per `quantize.py`'s own caveat. Fine for an
   op-set gate; not a statement about INT8 accuracy.

---

## Consequence for the plan

- **S0.2 → done, PASS.** Phase 3 is unblocked on the deployment axis.
- **S3.2 is effectively pre-satisfied** — merge exactness and the single-`Conv`
  export assertion both pass here. It still needs re-running against the real
  S3.1 module once that lands in `src/models/`, since this was a stub.
- **S3.1 inherits a working merge**: the `full_conv2d_depthwise` /`pad_to`
  kernel algebra and the gradient-masked diagonal support transfer directly.
- The block must carry **no normalisation and no nonlinearity between branches
  and the sum**. Any later "improvement" that adds either one silently destroys
  the entire deployment argument. That is risk 2 in the plan, and it now has a
  concrete test to guard it.
