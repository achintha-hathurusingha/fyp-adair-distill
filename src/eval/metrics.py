"""PSNR / SSIM with EXPLICITLY documented conventions (Task 2).

Metric conventions are the single most common source of irreproducible numbers
in this field. When implemented in Task 2, the following must be stated in code
AND mirrored in the eval config, then held fixed for every reported number:

  * colour space: RGB vs Y-channel (YCbCr) PSNR
  * border crop: pixels removed from each edge before comparison
  * data range: 1.0 (float) vs 255 (uint8)
  * clipping: whether predictions are clipped to the valid range first
  * rounding: whether predictions are quantised to uint8 before comparison

These are pending confirmation from the user (they determine comparability with
the AdaIR/PromptIR published numbers) — see Task 2.
"""
from __future__ import annotations

_TASK = "Task 2 — evaluation harness"


def psnr(*_args, **_kwargs):
    """Placeholder — implemented in Task 2 with documented conventions."""
    raise NotImplementedError(f"metrics.psnr is scaffolded for {_TASK}.")


def ssim(*_args, **_kwargs):
    """Placeholder — implemented in Task 2 with documented conventions."""
    raise NotImplementedError(f"metrics.ssim is scaffolded for {_TASK}.")
