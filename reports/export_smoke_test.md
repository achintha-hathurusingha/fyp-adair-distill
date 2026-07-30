# Export smoke test — Gate G1

Fixed input shape `(1, 3, 256, 256)` · ONNX Runtime static INT8 PTQ (QDQ) · CPU latency, single-thread.

**Verdict: PASS** — both configurations export to ONNX + INT8 and run.

## Summary

| variant | params | FP32 MB | INT8 MB | FP32 ms (med) | INT8 ms (med) | quant peak RSS MB |
|---|---|---|---|---|---|---|
| nafnet_w32 | 29,159,715 | 111.45 | 30.39 | 898.7 | 1187.3 | 4059 |
| nafnet_w32_gate | 29,160,267 | 111.46 | 30.39 | 969.0 | 1209.3 | 2843 |

## nafnet_w32

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


## nafnet_w32_gate

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

