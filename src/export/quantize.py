"""INT8 quantization via ONNX Runtime static PTQ (Gate G1 smoke-test path).

QDQ format, per-channel int8 weights and int8 activations. Calibration data for
G1 is random noise of the right shape (``calib_source: random``); real
calibration images are wired in once the dataloader exists (Task 2).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


class RandomCalibrationReader:
    """Yields random calibration batches for static PTQ.

    Placeholder calibration for the G1 smoke test only. Real deployment
    calibration must use in-distribution images (Task 2+).
    """

    def __init__(self, input_name: str, shape: tuple[int, ...], n: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self._data = [
            {input_name: rng.standard_normal(shape).astype(np.float32)}
            for _ in range(n)
        ]
        self._it = iter(self._data)

    def get_next(self):  # onnxruntime CalibrationDataReader protocol
        return next(self._it, None)


def quantize_static_int8(fp32_path: str | Path, int8_path: str | Path,
                         input_shape: tuple[int, ...], *, per_channel: bool = True,
                         calib_samples: int = 32, seed: int = 0) -> Path:
    """Statically quantize an FP32 ONNX model to INT8 (QDQ). Returns int8 path.

    Raises if the quantized model fails to load back.
    """
    from onnxruntime.quantization import (
        CalibrationMethod, QuantFormat, QuantType, quantize_static,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process

    fp32_path = Path(fp32_path)
    int8_path = Path(int8_path)
    int8_path.parent.mkdir(parents=True, exist_ok=True)

    # Preprocess (shape inference + graph opt) improves quantization coverage.
    prep_path = int8_path.with_suffix(".prep.onnx")
    quant_pre_process(str(fp32_path), str(prep_path), skip_symbolic_shape=True)

    import onnx
    input_name = onnx.load(str(prep_path)).graph.input[0].name
    reader = RandomCalibrationReader(input_name, input_shape, calib_samples, seed)

    quantize_static(
        str(prep_path), str(int8_path), reader,
        quant_format=QuantFormat.QDQ,
        per_channel=per_channel,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
    )
    prep_path.unlink(missing_ok=True)

    quant = onnx.load(str(int8_path))
    onnx.checker.check_model(quant)
    return int8_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Static INT8 PTQ of an ONNX model.")
    ap.add_argument("--fp32", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shape", nargs=4, type=int, default=[1, 3, 256, 256])
    ap.add_argument("--calib-samples", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    path = quantize_static_int8(
        args.fp32, args.out, tuple(args.shape),
        calib_samples=args.calib_samples, seed=args.seed,
    )
    print(f"[quantize] INT8 model -> {path}")


if __name__ == "__main__":
    main()
