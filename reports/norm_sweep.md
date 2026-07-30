# Normalization variant latency sweep — Task 1.5c

**Methodology (pinned for citability).** Qualcomm AI Hub · Samsung Galaxy S24
(Snapdragon 8 Gen 3, Hexagon v75) · INT8 QNN context binary
(`--target_runtime qnn_context_binary`) · weights and activations
`QuantizeDtype.INT8`, per-channel weights · input `(1, 3, 256, 256)` fixed ·
skeleton `w16_b8` (width 16, enc `[1,1,1,8]`, middle 2, dec `[1,1,1,1]`;
2.44M params, 2.13 GMACs) · **untrained weights** (latency is
weight-independent) · placeholder uniform-noise calibration (valid for latency
and op support, *not* for INT8 accuracy) · measured **2026-07-30** ·
`qai-hub` 0.53.0, `torch` 2.5.1, ONNX opset 17.

---

## Results

| variant | norm | INT8 ms (S24) | vs N-A | layers | cycles | norm % of cycles | peak mem MB | fallback | needs retraining? | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| **N-A** | `LayerNorm2d` | 2.510 | 1.00× | 637 | 17,812,918 | 62.3% | 98.0 | 0 | reference | **reference** |
| **N-A′** | `reciprocal(sqrt)` | 2.869 | **0.87×** | 671 | 20,336,850 | 67.6% | 98.5 | 0 | **no** | **REJECTED — slower** |
| **N-F** | LN + affine @ full-res | **1.572** | **1.60×** | 609 | 10,662,229 | 36.9% | 98.9 | 0 | yes | **carry to 1.5b** |
| **N-E** | affine only (floor) | **1.069** | **2.35×** | 399 | 6,704,194 | 1.5% | 98.6 | 0 | yes | **carry to 1.5b** |

**Zero NPU→CPU fallback in every variant.** All ops — including `Reciprocal`,
which never appeared in the Gate G1 coverage tables — are QNN-supported.

Dropped before profiling, per review: **N-B** (BatchNorm — its only attraction
was folding to zero ops, which N-E already bounds, against EDSR-lineage quality
evidence and a patch-vs-full-resolution statistics mismatch) and **N-C** (no
norm — instability risk for marginal remaining gain over N-E).

---

## Exported op deltas (FP32 graph, vs N-A)

| variant | Div | Sqrt | ReduceMean | Pow | Reciprocal | Mul |
|---|---|---|---|---|---|---|
| N-A | 68 | 34 | 68 | 34 | 0 | 187 |
| N-A′ | 34 | 34 | 68 | 34 | **34** | 221 |
| N-F | 64 | 30 | 60 | 30 | 0 | 187 |
| N-E | 34 | **0** | **0** | **0** | 0 | 187 |

The 34 `Div` present in *every* variant are SimpleGate `chunk(2, dim=1)` split
arithmetic — scalar shape computation, constant-folded during INT8
preprocessing — not elementwise normalization division.

---

## N-A′: numerically free, but rejected

`allclose(N-A, N-A′) atol=1e-5` passes at module level, with randomised affine
parameters, and across the full network via a shared `state_dict`. N-A′ is a
pure graph rewrite: **no retraining, no quality ablation, checkpoints load
directly.** It also did exactly what it was designed to do — `Div` fell from
61.7% to 3.4% of cycles, a 15.7× reduction on that operation.

**It is still 14% slower**, because QNN *fuses* the LayerNorm subgraph and the
rewrite breaks the fusion match. See `findings.md` F3/F4 for the full cycle
table. Retained in the codebase only as a documented negative result.

---

## The headline: normalization cost is concentrated at full resolution

**N-F removes 4 of 34 normalizations — 11.8% of them — and captures 65% of the
total available normalization saving.**

| | ms | saving vs N-A |
|---|---|---|
| N-A → N-F (drop 4 full-res norms) | 1.572 | 0.938 ms |
| N-A → N-E (drop all 34 norms) | 1.069 | 1.441 ms (the maximum available) |
| **N-F share of maximum** | | **65.1%** |

In cycles: normalization costs 11.10M (N-A) → 3.93M (N-F) → 0.10M (N-E). N-F
eliminates 7.17M of the 11.00M that is removable at all.

This is a **5.5× over-representation** — 11.8% of the normalizations carrying
65% of the cost — and it directly confirms F1: normalization cost is
per-element, so it scales with the spatial resolution at which it runs.

**Quantitative check of the model.** The area-weighted proxy in `findings.md`
predicts the two full-resolution blocks should carry 72% of normalization cost
for this skeleton; measured is 65%. Good first-order agreement. The gap is real
and unexplained — likely because fusion overhead and memory traffic do not scale
purely with element count — so the proxy should be treated as a ranking
heuristic, not a cost model.

---

## Recommendation: which variants proceed to the 1.5b quality ablation

**Carry two, against N-A as reference:**

1. **N-F** (1.60×, 1.572 ms) — the conservative option. Only the two
   full-resolution stages lose their statistics; every deeper stage keeps
   `LayerNorm2d` untouched. Lowest quality risk of the two, and it already
   captures two-thirds of the achievable gain.
2. **N-E** (2.35×, 1.069 ms) — the aggressive option and the **floor
   reference**: 1.5% norm cycles means essentially nothing remains to chase
   beyond it. If N-E's quality holds, it is the answer and nothing further is
   worth testing.

**Do not carry N-A′** (slower) or N-B/N-C (dropped above).

**The gap between them is the whole question, and it is a quality question, not
a latency one.** N-E is 1.47× faster than N-F; whether that is worth removing
normalization statistics from the deep stages — where they are nearly free —
is exactly what 1.5b must measure. Prior expectation: N-F wins, because it buys
most of the speed for a fraction of the representational disturbance.

---

## Consequences for the student family (open)

The 12-config sweep and the resulting family were both selected on **N-A
latency**, i.e. on numbers that are ~62% normalization overhead — a large,
roughly-fixed cost sitting on every configuration and compressing the range
(MAC span 4.26× vs latency span only 1.70×).

Once 1.5b locks the normalization, **`assign_family` must be re-run on
re-profiled latencies**, through the corrected path where profiling precedes
selection. Ordering will probably survive; the span and possibly the M choice
may not, since configurations differ in how much full-resolution normalization
they carry — precisely the quantity N-F/N-E change.

That re-profile is 12 jobs and roughly 40 minutes of AI Hub time. It is not
worth doing before the normalization is locked on quality.

---

## Not run (parked — scope guard)

- **Second backend (Jetson Orin / TensorRT).** Division is cheap on GPU and
  TensorRT fuses differently; the fusion finding (F3/F4) may be
  Hexagon-specific. Backend-dependent optima would be a stronger claim than
  "normalization is slow."
- **Measured AdaIR teacher latency on the same S24.** Would replace an
  acknowledged MAC under-estimate with a real denominator; a compile failure
  would itself be a result.
