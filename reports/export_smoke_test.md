# Export smoke test — Gate G1

Fixed input shape `(1, 3, 256, 256)` · ONNX Runtime static INT8 PTQ (QDQ) · CPU latency, single-thread.

**Verdict: PASS** — both configurations export to ONNX + INT8 and run.

## Summary

| variant | params | FP32 MB | INT8 MB | FP32 ms (med) | INT8 ms (med) | quant peak RSS MB |
|---|---|---|---|---|---|---|
| nafnet_w32 | 29,159,715 | 111.45 | 30.39 | 2845.5 | 3092.2 | 3878 |
| nafnet_w32_gate | 29,160,267 | 111.46 | 30.39 | 928.6 | 1354.6 | 3819 |

## Findings

### 1. The channel gate is export-safe (the primary G1 question)

Adding the gate changes the INT8 graph by only: `Conv`(+2), `DequantizeLinear`(+9), `GlobalAveragePool`(+1), `Mul`(+1), `QuantizeLinear`(+5), `Sigmoid`(+1).

**All gate ops are SUPPORTED on QNN, TFLite and TensorRT.**
Parameter cost is negligible (552 params).

### 2. `LayerNorm2d` is the real INT8 risk, not the gate

NAFNet's channel LayerNorm has no native op and decomposes into `ReduceMean`x144, `Pow`x72, `Sqrt`x72, `Div`x72, `Sub`x72.
These are flagged CAUTION on QNN and TFLite: `Pow`/`Sqrt`/`Div` on near-zero variance quantize poorly in INT8, and QNN may fall back to CPU for them, destroying the latency benefit.
**Mitigation to evaluate in Task 5/Phase 02:** replace LayerNorm2d with a fixed-scale or BatchNorm-style normalisation that folds into the preceding conv, or fuse it into a single custom op per backend.

### 3. `SimpleGate` emits `Slice` (QNN caution)

The channel-chunk gating yields `Slice`x146, flagged CAUTION on QNN. Usually convertible, but confirm on-device in Task 5.

### 4. CPU INT8 latency is NOT a proxy for on-device NPU latency

INT8 measures *slower* than FP32 here. That is expected: ONNX Runtime's x86 CPU provider gains little from QDQ INT8 and pays de/quantize overhead (note the large `QuantizeLinear`/`DequantizeLinear` counts). These numbers prove the graph **quantizes and executes**; they say nothing about NPU speed. Real latency comes from the on-device benchmark in Task 5.

> **Treat the latency column as order-of-magnitude only.** It was measured on a shared laptop CPU and varies by >2x between runs under load. The gate adds 552 parameters and one GAP/conv/sigmoid/multiply, so any large gate-vs-no-gate latency gap in the table is measurement noise, not a real cost. Do not quote these figures as results.

### 5. Shape ops vanish after quantization preprocessing

The FP32 graphs contain `Shape`/`Gather`/`Mod`/`ConstantOfShape` from the dynamic padding helper. Because the export shape is fixed, `quant_pre_process` constant-folds them away — they are absent from the INT8 graphs and are therefore not a deployment risk at fixed resolution.

## Full op-coverage tables

### nafnet_w32

### Op coverage — `nafnet_w32.onnx`

| op_type           |   count | qnn       | tflite    | tensorrt   |
|-------------------|---------|-----------|-----------|------------|
| Add               |     293 | SUPPORTED | SUPPORTED | SUPPORTED  |
| Cast              |       2 | UNKNOWN   | UNKNOWN   | UNKNOWN    |
| Concat            |       3 | SUPPORTED | SUPPORTED | SUPPORTED  |
| Constant          |     608 | UNKNOWN   | UNKNOWN   | UNKNOWN    |
| ConstantOfShape   |       1 | UNKNOWN   | UNKNOWN   | UNKNOWN    |
| Conv              |     226 | SUPPORTED | SUPPORTED | SUPPORTED  |
| DepthToSpace      |       4 | CAUTION   | SUPPORTED | SUPPORTED  |
| Div               |     144 | CAUTION   | CAUTION   | SUPPORTED  |
| Gather            |      73 | UNKNOWN   | UNKNOWN   | UNKNOWN    |
| GlobalAveragePool |      36 | SUPPORTED | SUPPORTED | SUPPORTED  |
| Identity          |     206 | UNKNOWN   | UNKNOWN   | UNKNOWN    |
| Mod               |       4 | UNKNOWN   | UNKNOWN   | UNKNOWN    |
| Mul               |     396 | SUPPORTED | SUPPORTED | SUPPORTED  |
| Pad               |       1 | SUPPORTED | SUPPORTED | SUPPORTED  |
| Pow               |      72 | CAUTION   | CAUTION   | SUPPORTED  |
| ReduceMean        |     144 | CAUTION   | CAUTION   | SUPPORTED  |
| Reshape           |       2 | SUPPORTED | SUPPORTED | SUPPORTED  |
| Shape             |      73 | UNKNOWN   | UNKNOWN   | UNKNOWN    |
| Slice             |     147 | CAUTION   | SUPPORTED | SUPPORTED  |
| Sqrt              |      72 | CAUTION   | CAUTION   | SUPPORTED  |
| Sub               |      75 | SUPPORTED | SUPPORTED | SUPPORTED  |
| Transpose         |       1 | SUPPORTED | SUPPORTED | SUPPORTED  |
| Unsqueeze         |       4 | UNKNOWN   | UNKNOWN   | UNKNOWN    |

- **qnn**: **UNKNOWN**: Cast, Constant, ConstantOfShape, Gather, Identity, Mod, Shape, Unsqueeze; _caution_: DepthToSpace, Div, Pow, ReduceMean, Slice, Sqrt
- **tflite**: **UNKNOWN**: Cast, Constant, ConstantOfShape, Gather, Identity, Mod, Shape, Unsqueeze; _caution_: Div, Pow, ReduceMean, Sqrt
- **tensorrt**: **UNKNOWN**: Cast, Constant, ConstantOfShape, Gather, Identity, Mod, Shape, Unsqueeze


INT8 (post-QDQ) graph:

### Op coverage — `nafnet_w32_int8.onnx`

| op_type           |   count | qnn       | tflite    | tensorrt   |
|-------------------|---------|-----------|-----------|------------|
| Add               |     221 | SUPPORTED | SUPPORTED | SUPPORTED  |
| Conv              |     226 | SUPPORTED | SUPPORTED | SUPPORTED  |
| DepthToSpace      |       4 | CAUTION   | SUPPORTED | SUPPORTED  |
| DequantizeLinear  |    1418 | SUPPORTED | SUPPORTED | SUPPORTED  |
| Div               |      72 | CAUTION   | CAUTION   | SUPPORTED  |
| GlobalAveragePool |      36 | SUPPORTED | SUPPORTED | SUPPORTED  |
| Mul               |     180 | SUPPORTED | SUPPORTED | SUPPORTED  |
| Pad               |       1 | SUPPORTED | SUPPORTED | SUPPORTED  |
| Pow               |      72 | CAUTION   | CAUTION   | SUPPORTED  |
| QuantizeLinear    |     959 | SUPPORTED | SUPPORTED | SUPPORTED  |
| ReduceMean        |     144 | CAUTION   | CAUTION   | SUPPORTED  |
| Slice             |     146 | CAUTION   | SUPPORTED | SUPPORTED  |
| Sqrt              |      72 | CAUTION   | CAUTION   | SUPPORTED  |
| Sub               |      72 | SUPPORTED | SUPPORTED | SUPPORTED  |

- **qnn**: _caution_: DepthToSpace, Div, Pow, ReduceMean, Slice, Sqrt
- **tflite**: _caution_: Div, Pow, ReduceMean, Sqrt
- **tensorrt**: all ops supported


### nafnet_w32_gate

### Op coverage — `nafnet_w32_gate.onnx`

| op_type           |   count | qnn       | tflite    | tensorrt   |
|-------------------|---------|-----------|-----------|------------|
| Add               |     293 | SUPPORTED | SUPPORTED | SUPPORTED  |
| Cast              |       2 | UNKNOWN   | UNKNOWN   | UNKNOWN    |
| Concat            |       3 | SUPPORTED | SUPPORTED | SUPPORTED  |
| Constant          |     608 | UNKNOWN   | UNKNOWN   | UNKNOWN    |
| ConstantOfShape   |       1 | UNKNOWN   | UNKNOWN   | UNKNOWN    |
| Conv              |     228 | SUPPORTED | SUPPORTED | SUPPORTED  |
| DepthToSpace      |       4 | CAUTION   | SUPPORTED | SUPPORTED  |
| Div               |     144 | CAUTION   | CAUTION   | SUPPORTED  |
| Gather            |      73 | UNKNOWN   | UNKNOWN   | UNKNOWN    |
| GlobalAveragePool |      37 | SUPPORTED | SUPPORTED | SUPPORTED  |
| Identity          |     206 | UNKNOWN   | UNKNOWN   | UNKNOWN    |
| Mod               |       4 | UNKNOWN   | UNKNOWN   | UNKNOWN    |
| Mul               |     397 | SUPPORTED | SUPPORTED | SUPPORTED  |
| Pad               |       1 | SUPPORTED | SUPPORTED | SUPPORTED  |
| Pow               |      72 | CAUTION   | CAUTION   | SUPPORTED  |
| ReduceMean        |     144 | CAUTION   | CAUTION   | SUPPORTED  |
| Relu              |       1 | SUPPORTED | SUPPORTED | SUPPORTED  |
| Reshape           |       2 | SUPPORTED | SUPPORTED | SUPPORTED  |
| Shape             |      73 | UNKNOWN   | UNKNOWN   | UNKNOWN    |
| Sigmoid           |       1 | SUPPORTED | SUPPORTED | SUPPORTED  |
| Slice             |     147 | CAUTION   | SUPPORTED | SUPPORTED  |
| Sqrt              |      72 | CAUTION   | CAUTION   | SUPPORTED  |
| Sub               |      75 | SUPPORTED | SUPPORTED | SUPPORTED  |
| Transpose         |       1 | SUPPORTED | SUPPORTED | SUPPORTED  |
| Unsqueeze         |       4 | UNKNOWN   | UNKNOWN   | UNKNOWN    |

- **qnn**: **UNKNOWN**: Cast, Constant, ConstantOfShape, Gather, Identity, Mod, Shape, Unsqueeze; _caution_: DepthToSpace, Div, Pow, ReduceMean, Slice, Sqrt
- **tflite**: **UNKNOWN**: Cast, Constant, ConstantOfShape, Gather, Identity, Mod, Shape, Unsqueeze; _caution_: Div, Pow, ReduceMean, Sqrt
- **tensorrt**: **UNKNOWN**: Cast, Constant, ConstantOfShape, Gather, Identity, Mod, Shape, Unsqueeze


INT8 (post-QDQ) graph:

### Op coverage — `nafnet_w32_gate_int8.onnx`

| op_type           |   count | qnn       | tflite    | tensorrt   |
|-------------------|---------|-----------|-----------|------------|
| Add               |     221 | SUPPORTED | SUPPORTED | SUPPORTED  |
| Conv              |     228 | SUPPORTED | SUPPORTED | SUPPORTED  |
| DepthToSpace      |       4 | CAUTION   | SUPPORTED | SUPPORTED  |
| DequantizeLinear  |    1427 | SUPPORTED | SUPPORTED | SUPPORTED  |
| Div               |      72 | CAUTION   | CAUTION   | SUPPORTED  |
| GlobalAveragePool |      37 | SUPPORTED | SUPPORTED | SUPPORTED  |
| Mul               |     181 | SUPPORTED | SUPPORTED | SUPPORTED  |
| Pad               |       1 | SUPPORTED | SUPPORTED | SUPPORTED  |
| Pow               |      72 | CAUTION   | CAUTION   | SUPPORTED  |
| QuantizeLinear    |     964 | SUPPORTED | SUPPORTED | SUPPORTED  |
| ReduceMean        |     144 | CAUTION   | CAUTION   | SUPPORTED  |
| Sigmoid           |       1 | SUPPORTED | SUPPORTED | SUPPORTED  |
| Slice             |     146 | CAUTION   | SUPPORTED | SUPPORTED  |
| Sqrt              |      72 | CAUTION   | CAUTION   | SUPPORTED  |
| Sub               |      72 | SUPPORTED | SUPPORTED | SUPPORTED  |

- **qnn**: _caution_: DepthToSpace, Div, Pow, ReduceMean, Slice, Sqrt
- **tflite**: _caution_: Div, Pow, ReduceMean, Sqrt
- **tensorrt**: all ops supported

