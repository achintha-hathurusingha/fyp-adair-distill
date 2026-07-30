"""Export a NAFNet (optionally + channel gate) to ONNX at a fixed resolution.

Gate G1 uses a fixed input shape (protocol is fixed-resolution), so no dynamic
axes are emitted — this keeps the graph simple and NPU-friendly.

CLI:
    python -m src.export.to_onnx --gate --out runs/g1/nafnet_gate.onnx
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.models.nafnet import build_nafnet
from src.utils.config import load_yaml, require
from src.utils.seeding import seed_everything


def export_onnx(model: torch.nn.Module, out_path: str | Path,
                input_shape: tuple[int, int, int, int], opset: int = 17) -> Path:
    """Export ``model`` to ONNX at ``input_shape``; return the output path.

    Raises if the ONNX graph fails to load back (no silent success).
    """
    model.eval()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(*input_shape)

    with torch.no_grad():
        torch.onnx.export(
            model, dummy, str(out_path),
            input_names=["input"], output_names=["output"],
            opset_version=opset, do_constant_folding=True, dynamic_axes=None,
        )

    import onnx
    onnx_model = onnx.load(str(out_path))
    onnx.checker.check_model(onnx_model)
    return out_path


def _build_from_configs(model_cfg_path: str, use_gate: bool) -> torch.nn.Module:
    cfg = load_yaml(model_cfg_path)
    return build_nafnet(cfg, use_gate=use_gate)


def main() -> None:
    ap = argparse.ArgumentParser(description="Export NAFNet to ONNX.")
    ap.add_argument("--model-config", default="configs/model/nafnet_w32.yaml")
    ap.add_argument("--export-config", default="configs/export/qnn_int8.yaml")
    ap.add_argument("--gate", action="store_true", help="insert the channel gate")
    ap.add_argument("--out", required=True, help="output .onnx path")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    seed_everything(args.seed)
    exp = require(load_yaml(args.export_config), "export", context="export config")
    shape = tuple(require(exp, "input_shape", context="export.export"))
    opset = exp.get("opset", 17)

    model = _build_from_configs(args.model_config, args.gate)
    path = export_onnx(model, args.out, shape, opset)
    print(f"[to_onnx] exported {'NAFNet+gate' if args.gate else 'NAFNet'} "
          f"at shape {shape} (opset {opset}) -> {path}")


if __name__ == "__main__":
    main()
