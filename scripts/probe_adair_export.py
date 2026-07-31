"""F7 due-diligence: isolate WHY the AdaIR teacher fails to export.

Two hypotheses were read off the source (`net/model.py:337-361`):
  H1  data-dependent dynamic slicing — mask bounds come from a learned
      threshold, so slice extents depend on tensor VALUES, not shapes
  H2  complex-valued FFT — torch.fft.fft2 / ifft2

Only H1 was *confirmed* by the original failure; H2 was inferred from reading
the code. This script separates them, and also rules out "wrong exporter" before
the finding is stated as categorical.

  A  patch out the dynamic slicing, re-export  -> does it now fail on the FFT?
  B  unpatched model via the dynamo exporter    -> is the TorchScript tracer the
                                                   only thing that cannot do it?

Output correctness is irrelevant here: the patch replaces a value-dependent
slice with a fixed one purely so tracing can proceed past it.

    python -m scripts.probe_adair_export
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import torch

from src.utils.config import REPO_ROOT

warnings.filterwarnings("ignore")


def _load_adair():
    repo = REPO_ROOT / "third_party" / "AdaIR"
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from net.model import AdaIR
    return AdaIR


def _static_fft(self, x, n=128):
    """`FreModule.fft` with the value-dependent slice replaced by a fixed one.

    Mirrors the original except that the mask bounds are constants rather than
    derived from ``self.rate_conv(...).sigmoid()``. Everything else — including
    both FFT calls — is untouched, so if export still fails it fails on the FFT.
    """
    # MUST match the original: conv1 changes the channel count, and omitting it
    # trips AdaIR's own shape assertion at model.py:190 — which on the first
    # attempt masked the export result entirely.
    x = self.conv1(x)
    mask = torch.zeros(x.shape, device=x.device, dtype=x.dtype)
    h, w = x.shape[-2], x.shape[-1]
    hq, wq = h // 4, w // 4                      # fixed, shape-derived
    mask[:, :, h // 2 - hq:h // 2 + hq, w // 2 - wq:w // 2 + wq] = 1

    fft = torch.fft.fft2(x, norm="forward", dim=(-2, -1))
    fft = self.shift(fft)
    fft_high = fft * (1 - mask)
    high = self.unshift(fft_high)
    high = torch.fft.ifft2(high, norm="forward", dim=(-2, -1))
    high = torch.abs(high)

    fft_low = fft * mask
    low = self.unshift(fft_low)
    low = torch.fft.ifft2(low, norm="forward", dim=(-2, -1))
    low = torch.abs(low)
    return high, low


def attempt(label: str, build, out: Path, **kwargs) -> tuple[bool, str]:
    """Run one export attempt and report the outcome verbatim."""
    print(f"\n=== {label} ===")
    try:
        model = build().eval()
        x = torch.randn(1, 3, 256, 256)
        torch.onnx.export(model, x, str(out), input_names=["input"],
                          output_names=["output"], **kwargs)
        print("  RESULT: export SUCCEEDED")
        return True, "succeeded"
    except Exception as exc:  # noqa: BLE001
        msg = f"{type(exc).__name__}: {str(exc)[:400]}"
        print(f"  RESULT: FAILED — {msg}")
        return False, msg


def main() -> None:
    adair_cls = _load_adair()
    out_dir = REPO_ROOT / "runs" / "export_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"torch {torch.__version__}")

    results: dict[str, str] = {}

    # --- Baseline: reproduce the original failure, for the record -----------
    _, results["baseline_opset17_torchscript"] = attempt(
        "A0 baseline — unpatched, TorchScript tracer, opset 17",
        lambda: adair_cls(decoder=True), out_dir / "a0.onnx", opset_version=17)

    # --- A: patch out the dynamic slicing; does the FFT then fail? ----------
    from net.model import FreModule
    original = FreModule.fft
    FreModule.fft = _static_fft
    try:
        for opset in (17, 20):
            _, results[f"static_slice_opset{opset}"] = attempt(
                f"A{opset} — dynamic slicing PATCHED OUT, TorchScript, opset {opset}",
                lambda: adair_cls(decoder=True), out_dir / f"a_{opset}.onnx",
                opset_version=opset)
    finally:
        FreModule.fft = original

    # --- B: unpatched model, dynamo exporter -------------------------------
    for opset in (18, 20):
        _, results[f"dynamo_opset{opset}"] = attempt(
            f"B{opset} — unpatched, DYNAMO exporter, opset {opset}",
            lambda: adair_cls(decoder=True), out_dir / f"b_{opset}.onnx",
            opset_version=opset, dynamo=True)

    print("\n=== SUMMARY ===")
    for k, v in results.items():
        print(f"  {k:34s} {v[:110]}")


if __name__ == "__main__":
    main()
