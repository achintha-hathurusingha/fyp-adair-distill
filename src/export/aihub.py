"""Qualcomm AI Hub submission: real INT8 quantization, compilation, profiling.

Why this exists: ``op_coverage.py`` is a *static* lookup table. It cannot prove a
graph converts. AI Hub performs a real quantize + convert (returning an actual
error when a backend rejects something) and profiles on real hardware, replacing
x86 QDQ latency noise with genuine NPU numbers and a per-layer compute-unit
breakdown (which reveals NPU->CPU fallback).

Three-stage flow, matching the AI Hub API:
  1. ``submit_quantize_job``  — FP32 ONNX + calibration -> INT8 ONNX
  2. ``submit_compile_job``   — INT8 ONNX -> device binary (QNN / TFLite)
  3. ``submit_profile_job``   — on-device latency + compute-unit breakdown

Requires a token (free account at https://aihub.qualcomm.com)::

    pip install qai-hub
    qai-hub configure --api_token <TOKEN>

Every function raises a clear, actionable error when credentials are missing — it
never silently degrades to fabricated numbers.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

_TOKEN_HELP = (
    "Qualcomm AI Hub credentials not found. Get a free token at "
    "https://aihub.qualcomm.com (sign in -> Settings -> API token), then run:\n"
    "    qai-hub configure --api_token <TOKEN>\n"
    "Without it, on-device latency cannot be measured and op support cannot be "
    "verified for real (op_coverage.py is only a static estimate)."
)


class AIHubUnavailable(RuntimeError):
    """Raised when AI Hub cannot be reached or is not configured."""


@dataclass
class DeviceJobResult:
    """Outcome of one quantize+compile+profile submission for a single model."""

    name: str
    quantized: bool = False
    compiled: bool = False
    profiled: bool = False
    error: str | None = None
    stage_failed: str | None = None
    inference_latency_ms: float | None = None
    peak_memory_mb: float | None = None
    compute_unit_breakdown: dict[str, int] = field(default_factory=dict)
    job_urls: dict[str, str] = field(default_factory=dict)

    @property
    def npu_fallback_layers(self) -> int:
        """Layers NOT executing on the NPU — the LayerNorm fallback signal."""
        return sum(n for unit, n in self.compute_unit_breakdown.items()
                   if unit.upper() not in ("NPU", "HTP", "DSP"))


def _hub():
    """Import qai_hub and verify credentials work."""
    try:
        import qai_hub as hub
    except ImportError as exc:  # pragma: no cover
        raise AIHubUnavailable(
            "qai-hub is not installed. Run: pip install qai-hub\n" + _TOKEN_HELP
        ) from exc
    try:
        hub.get_devices()
    except Exception as exc:  # noqa: BLE001 - surface any auth/network failure
        raise AIHubUnavailable(f"{_TOKEN_HELP}\n\nUnderlying error: {exc}") from exc
    return hub


def list_devices() -> list[str]:
    """Return the AI Hub device names available to this account."""
    hub = _hub()
    return [f"{d.name} | {d.os} | {','.join(d.attributes)}" for d in hub.get_devices()]


def make_calibration_data(input_name: str, shape: tuple[int, ...], n: int,
                          seed: int = 0) -> dict[str, list[np.ndarray]]:
    """Build placeholder calibration data (uniform in [0, 1], image-like).

    PLACEHOLDER for the architecture sweep: it exercises the real quantize path
    but is not distribution-matched, so INT8 *accuracy* from it is meaningless.
    Latency and op-support conclusions are unaffected. Replace with real
    degraded/clean image batches once the Task 2 dataloader exists.
    """
    rng = np.random.default_rng(seed)
    return {input_name: [rng.random(shape, dtype=np.float32) for _ in range(n)]}


def submit_and_profile(onnx_path: str | Path, name: str, device: str, *,
                       input_shape: tuple[int, ...],
                       calib_samples: int = 8,
                       compile_options: str = "",
                       profile_options: str = "",
                       seed: int = 0) -> DeviceJobResult:
    """Quantize to INT8, compile for ``device``, and profile. Never raises on
    a backend rejection — the failure is captured so a sweep can report which
    architectures real hardware refuses, and why.
    """
    import onnx
    import qai_hub as hub

    _hub()  # validate credentials up front
    result = DeviceJobResult(name=name)
    target = hub.Device(device)
    input_name = onnx.load(str(onnx_path)).graph.input[0].name

    # --- stage 1: real INT8 quantization -------------------------------------
    try:
        qjob = hub.submit_quantize_job(
            model=str(onnx_path),
            calibration_data=make_calibration_data(
                input_name, tuple(input_shape), calib_samples, seed),
            weights_dtype=hub.QuantizeDtype.INT8,
            activations_dtype=hub.QuantizeDtype.INT8,
            name=f"{name}-quantize",
        )
        result.job_urls["quantize"] = getattr(qjob, "url", "") or ""
        int8_model = qjob.get_target_model()
        if int8_model is None:
            raise AIHubUnavailable("quantize job produced no model")
        result.quantized = True
    except Exception as exc:  # noqa: BLE001
        result.error, result.stage_failed = str(exc), "quantize"
        return result

    # --- stage 2: real device compilation ------------------------------------
    try:
        cjob = hub.submit_compile_job(
            model=int8_model, device=target, options=compile_options,
            name=f"{name}-compile",
        )
        result.job_urls["compile"] = getattr(cjob, "url", "") or ""
        compiled = cjob.get_target_model()
        if compiled is None:
            raise AIHubUnavailable("compile job produced no target model")
        result.compiled = True
    except Exception as exc:  # noqa: BLE001
        result.error, result.stage_failed = str(exc), "compile"
        return result

    # --- stage 3: on-device profiling ----------------------------------------
    try:
        pjob = hub.submit_profile_job(
            model=compiled, device=target, options=profile_options,
            name=f"{name}-profile",
        )
        result.job_urls["profile"] = getattr(pjob, "url", "") or ""
        profile = pjob.download_profile()
        result.inference_latency_ms = _extract_latency_ms(profile)
        result.peak_memory_mb = _extract_peak_memory_mb(profile)
        result.compute_unit_breakdown = _extract_compute_units(profile)
        result.profiled = True
    except Exception as exc:  # noqa: BLE001
        result.error, result.stage_failed = str(exc), "profile"
    return result


def _extract_latency_ms(profile: dict[str, Any]) -> float | None:
    """Median inference time in ms (AI Hub reports microseconds)."""
    summary = profile.get("execution_summary", {})
    for key in ("estimated_inference_time", "inference_time",
                "estimated_inference_time_median"):
        val = summary.get(key)
        if isinstance(val, (int, float)):
            return val / 1000.0
    return None


def _extract_peak_memory_mb(profile: dict[str, Any]) -> float | None:
    """Peak inference memory in MB."""
    summary = profile.get("execution_summary", {})
    for key in ("inference_memory_peak_range", "estimated_inference_peak_memory"):
        val = summary.get(key)
        if isinstance(val, (list, tuple)) and val:
            return max(val) / (1024 ** 2)
        if isinstance(val, (int, float)):
            return val / (1024 ** 2)
    return None


def _extract_compute_units(profile: dict[str, Any]) -> dict[str, int]:
    """Count layers per compute unit (NPU/GPU/CPU).

    A non-zero CPU count is the concrete signal that ops fell off the NPU —
    exactly the LayerNorm decomposition risk flagged at Gate G1.
    """
    counts: dict[str, int] = {}
    detail = profile.get("execution_detail") or []
    for layer in detail:
        unit = str(layer.get("compute_unit", "UNKNOWN"))
        counts[unit] = counts.get(unit, 0) + 1
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description="Qualcomm AI Hub helper.")
    ap.add_argument("--list-devices", action="store_true")
    args = ap.parse_args()
    if args.list_devices:
        for d in list_devices():
            print(d)
        return
    ap.error("nothing to do; pass --list-devices")


if __name__ == "__main__":
    main()
