"""Static ONNX op-coverage report against edge INT8 backends.

For each unique op type in a graph, report whether the target backend
(QNN / TFLite / TensorRT) is expected to support it in an INT8 pipeline.
Verdicts are curated from the backends' public op-support docs and are a
*static* signal — a real on-device convert (Task 5) is the ground truth. This
report exists to surface architecture-level export risks in week one.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

# Curated op-support knowledge. "supported" = generally available; "caution" =
# available but with INT8 quirks / decomposition / accuracy risk on that
# backend. Anything not listed is reported as UNKNOWN (treated as a risk).
_SUPPORTED = {
    "qnn": {
        "Conv", "Relu", "Sigmoid", "Mul", "Add", "Sub", "Concat", "Split",
        "GlobalAveragePool", "AveragePool", "MaxPool", "Clip", "Pad",
        "QuantizeLinear", "DequantizeLinear", "Reshape", "Transpose",
    },
    "tflite": {
        "Conv", "Relu", "Sigmoid", "Mul", "Add", "Sub", "Concat", "Split",
        "GlobalAveragePool", "AveragePool", "MaxPool", "Clip", "Pad",
        "DepthToSpace", "QuantizeLinear", "DequantizeLinear", "Reshape",
        "Transpose", "Slice",
    },
    "tensorrt": {
        "Conv", "Relu", "Sigmoid", "Mul", "Add", "Sub", "Concat", "Split",
        "GlobalAveragePool", "AveragePool", "MaxPool", "Clip", "Pad",
        "DepthToSpace", "ReduceMean", "Sqrt", "Pow", "Div", "Slice",
        "QuantizeLinear", "DequantizeLinear", "Reshape", "Transpose",
    },
}
# Ops that are technically supported but flagged as INT8-risky per backend.
_CAUTION = {
    "qnn": {"ReduceMean", "Sqrt", "Pow", "Div", "DepthToSpace", "Slice"},
    "tflite": {"ReduceMean", "Sqrt", "Pow", "Div"},
    "tensorrt": set(),
}


def op_histogram(onnx_path: str | Path) -> Counter:
    """Return a Counter of op_type -> count for the graph (incl. subgraphs)."""
    import onnx

    model = onnx.load(str(onnx_path))
    hist: Counter = Counter()

    def walk(graph):
        for node in graph.node:
            hist[node.op_type] += 1
            for attr in node.attribute:
                if attr.g.ByteSize():
                    walk(attr.g)
                for g in attr.graphs:
                    walk(g)

    walk(model.graph)
    return hist


def verdict(op: str, backend: str) -> str:
    """Return SUPPORTED / CAUTION / UNKNOWN for one op on one backend."""
    if op in _CAUTION.get(backend, set()):
        return "CAUTION"
    if op in _SUPPORTED.get(backend, set()):
        return "SUPPORTED"
    return "UNKNOWN"


def coverage_table(onnx_path: str | Path,
                   backends: list[str] | None = None) -> tuple[list[str], list[list[str]], dict]:
    """Build a coverage table; return (headers, rows, per-backend risk summary)."""
    backends = backends or ["qnn", "tflite", "tensorrt"]
    hist = op_histogram(onnx_path)
    headers = ["op_type", "count", *backends]
    rows: list[list[str]] = []
    risks = {b: {"CAUTION": [], "UNKNOWN": []} for b in backends}

    for op in sorted(hist):
        row = [op, str(hist[op])]
        for b in backends:
            v = verdict(op, b)
            row.append(v)
            if v in ("CAUTION", "UNKNOWN"):
                risks[b][v].append(op)
        rows.append(row)
    return headers, rows, risks


def render_markdown(onnx_path: str | Path, backends: list[str] | None = None) -> str:
    """Render the coverage table + risk summary as Markdown."""
    from tabulate import tabulate

    headers, rows, risks = coverage_table(onnx_path, backends)
    md = [f"### Op coverage — `{Path(onnx_path).name}`", "",
          tabulate(rows, headers=headers, tablefmt="github"), ""]
    for b, r in risks.items():
        parts = []
        if r["UNKNOWN"]:
            parts.append(f"**UNKNOWN**: {', '.join(r['UNKNOWN'])}")
        if r["CAUTION"]:
            parts.append(f"_caution_: {', '.join(r['CAUTION'])}")
        md.append(f"- **{b}**: " + ("; ".join(parts) if parts else "all ops supported"))
    return "\n".join(md) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="ONNX op-coverage report.")
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--backends", nargs="*", default=["qnn", "tflite", "tensorrt"])
    args = ap.parse_args()
    print(render_markdown(args.onnx, args.backends))


if __name__ == "__main__":
    main()
