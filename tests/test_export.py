"""Tests for the export path: ONNX validity, gate op set, coverage reporting."""
from __future__ import annotations

import pytest

from src.export.op_coverage import op_histogram, verdict
from src.export.to_onnx import export_onnx
from src.models.gate import ChannelGate
from src.models.nafnet import NAFNet


@pytest.fixture(scope="module")
def tiny_onnx(tmp_path_factory) -> str:
    """Export a tiny NAFNet so tests stay fast."""
    model = NAFNet(width=8, enc_blk_nums=[1, 1], middle_blk_num=1,
                   dec_blk_nums=[1, 1])
    out = tmp_path_factory.mktemp("onnx") / "tiny.onnx"
    return str(export_onnx(model, out, (1, 3, 64, 64)))


def test_export_produces_valid_onnx(tiny_onnx: str) -> None:
    hist = op_histogram(tiny_onnx)
    assert hist["Conv"] > 0


def test_gate_export_uses_only_allowed_ops(tmp_path) -> None:
    """The gate must expand to exactly the documented minimal op set."""
    gate = ChannelGate(16, reduction=4)
    path = export_onnx(gate, tmp_path / "gate.onnx", (1, 16, 32, 32))
    ops = set(op_histogram(path))
    allowed = {"GlobalAveragePool", "Conv", "Relu", "Sigmoid", "Mul",
               "AveragePool", "ReduceMean"}
    assert ops <= allowed, f"gate emitted unexpected ops: {ops - allowed}"


def test_verdict_classification() -> None:
    assert verdict("Conv", "qnn") == "SUPPORTED"
    assert verdict("ReduceMean", "qnn") == "CAUTION"
    assert verdict("SomeExoticOp", "qnn") == "UNKNOWN"


#: Every op the INT8 student graph is currently allowed to contain. Ops here are
#: either SUPPORTED on all three backends, or a CAUTION we have consciously
#: accepted and tracked (LayerNorm arithmetic, SimpleGate's Slice, PixelShuffle's
#: DepthToSpace). Adding an op to the architecture that is NOT in this set should
#: fail CI, not be discovered during on-device conversion months later.
INT8_OP_ALLOWLIST = {
    # core compute
    "Conv", "Add", "Sub", "Mul", "Div", "Relu", "Sigmoid",
    # quantization boundaries
    "QuantizeLinear", "DequantizeLinear",
    # pooling / attention-free channel gating
    "GlobalAveragePool", "AveragePool",
    # accepted-risk ops, tracked in reports/export_smoke_test.md
    "ReduceMean", "Pow", "Sqrt",   # LayerNorm2d decomposition
    "Slice", "Concat",             # SimpleGate channel chunk
    "DepthToSpace",                # PixelShuffle upsampling
    # shape/no-op plumbing that survives folding
    "Pad", "Reshape", "Transpose", "Identity", "Constant",
}


def test_int8_graph_contains_no_op_outside_allowlist(tmp_path) -> None:
    """Regression guard: the whole INT8 graph must stay within the allowlist.

    This is the standing version of the Gate G1 op-coverage table. It converts a
    one-time report into a guard, so an architecture change that introduces an
    export-hostile op (GELU, Softmax, LayerNormalization, Resize, Einsum, ...)
    breaks the build immediately.
    """
    from src.export.quantize import quantize_static_int8

    model = NAFNet(width=8, enc_blk_nums=[1, 1], middle_blk_num=1,
                   dec_blk_nums=[1, 1], use_gate=True)
    shape = (1, 3, 64, 64)
    fp32 = export_onnx(model, tmp_path / "guard.onnx", shape)
    int8 = quantize_static_int8(fp32, tmp_path / "guard_int8.onnx", shape,
                               calib_samples=2)

    ops = set(op_histogram(int8))
    unexpected = ops - INT8_OP_ALLOWLIST
    assert not unexpected, (
        f"INT8 graph gained op(s) outside the allowlist: {sorted(unexpected)}. "
        "Either the op is deployable (add it to INT8_OP_ALLOWLIST with a note) "
        "or the architecture change must be reverted."
    )
