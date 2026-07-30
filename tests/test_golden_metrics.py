"""Cross-implementation oracle: our metrics vs AdaIR's own function.

``tests/golden/adair_metrics.json`` is produced by
``scripts/make_golden_metrics.py`` running inside the legacy environment
(Python 3.8 / scikit-image 0.19.3), where AdaIR's ``compute_psnr_ssim``
executes as released. Asserting against it validates our implementation against
the reference independently of datasets, teacher and dataloaders — so if Gate G3
fails, metric correctness is already excluded as a cause.

Skips (rather than fails) when the golden file is absent, so the suite stays
green before the legacy environment has been built.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.metrics import ADAIR_DEFAULT, psnr_ssim

GOLDEN = Path(__file__).parent / "golden" / "adair_metrics.json"
#: Tolerance. Differences should be at float round-off; anything larger means a
#: genuine convention divergence, not numerical noise.
ATOL_PSNR = 1e-4
ATOL_SSIM = 1e-6


def _load() -> dict:
    if not GOLDEN.exists():
        pytest.skip(
            f"{GOLDEN.name} not generated yet — run scripts/make_golden_metrics.py "
            "inside the legacy (skimage 0.19.3) environment")
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


def _rebuild(case: dict) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct a case's arrays from its recorded seed and shape.

    Must mirror ``build_cases`` in the generator exactly, including drawing both
    ``rand`` and ``randn`` from the same RandomState in that order.
    """
    shape = tuple(case["shape"])
    rng = np.random.RandomState(case["seed"])
    clean = rng.rand(*shape)
    degraded = np.clip(clean + rng.randn(*shape) * case["noise"], 0, 1)
    return clean, degraded


def test_golden_file_is_from_the_legacy_environment() -> None:
    """The oracle is only meaningful if produced where multichannel= still works."""
    data = _load()
    version = data.get("skimage_version", "")
    assert version < "0.23", (
        f"golden file was generated with scikit-image {version}; "
        "multichannel= was removed in 0.23, so the values would not reflect "
        "AdaIR's released behaviour")


def test_matches_adair_reference_implementation() -> None:
    """Every recorded case must match our implementation."""
    data = _load()
    mismatches = []
    for case in data["cases"]:
        clean, degraded = _rebuild(case)
        p, s = psnr_ssim(degraded, clean, ADAIR_DEFAULT)
        if abs(p - case["psnr"]) > ATOL_PSNR or abs(s - case["ssim"]) > ATOL_SSIM:
            mismatches.append(
                f"case {case['id']} shape={case['shape']}: "
                f"PSNR ours {p:.6f} vs AdaIR {case['psnr']:.6f} "
                f"(Δ{p - case['psnr']:+.2e}); "
                f"SSIM ours {s:.6f} vs AdaIR {case['ssim']:.6f} "
                f"(Δ{s - case['ssim']:+.2e})")
    assert not mismatches, "\n".join(mismatches)


def test_golden_covers_bsd68_post_crop_geometry() -> None:
    """The oracle must include the shape our real evaluation actually uses."""
    data = _load()
    shapes = {tuple(c["shape"]) for c in data["cases"]}
    assert (320, 480, 3) in shapes, (
        "golden cases should include BSD68's post-crop geometry (320x480x3)")


def test_golden_covers_identical_and_noisy_pairs() -> None:
    """Both the degenerate case and realistic error levels must be represented."""
    data = _load()
    noises = {c["noise"] for c in data["cases"]}
    assert 0.0 in noises, "no identical-image case (PSNR should be inf)"
    assert any(n > 0 for n in noises), "no noisy case"
