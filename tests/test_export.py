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
