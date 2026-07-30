"""Degradation synthesis, replicating AdaIR exactly.

AdaIR adds Gaussian noise in **[0, 255] space and casts the result to uint8**
(`utils/dataset_utils.py:243-246`)::

    noise = np.random.randn(*clean_patch.shape)
    noisy_patch = np.clip(clean_patch + noise * self.sigma, 0, 255).astype(np.uint8)

Two details worth stating precisely, because the obvious framing of the first
one is wrong:

* **The cast to uint8 quantises the input; the choice of scale does not matter.**
  Adding ``noise * sigma`` in 255-space and adding ``noise * sigma/255`` in
  [0,1]-space are *mathematically equivalent* and produce byte-identical uint8
  output (verified in ``tests/test_degradations.py``). What actually differs is
  whether the result is quantised at all: keeping the noisy image as float and
  never casting gives a marginally cleaner input. Measured effect: **+0.0033 dB**
  on a 64x64 patch at sigma=25 — real, reproducible, and far below the +/-0.10 dB
  gate tolerance. We replicate the cast for fidelity, not because it is
  numerically important.
* **sigma is on the 0-255 scale** and the noise is added *after* the
  multiple-of-16 crop, so the crop changes which pixels get which realisation.

Two seeding modes are provided, and the difference between them is a reportable
quantity rather than a hidden choice:

``global``
    AdaIR's own behaviour — one global ``np.random.seed(0)`` consumed in dataset
    iteration order. Since ``os.listdir`` is unsorted (``dataset_utils.py:238``),
    the realisation is filesystem-dependent. Used for G2/G3 comparability.

``filename`` (default)
    Per-image seed derived from a hash of the filename. Order-independent and
    stable across machines, filesystems and re-downloads — sorting fixes only
    the first of those, and only until a file is added.
"""
from __future__ import annotations

import hashlib

import numpy as np

#: Noise levels of the 3-degradation protocol, on the 0-255 scale.
SIGMAS = (15, 25, 50)


def filename_seed(filename: str) -> int:
    """Derive a stable 32-bit seed from a filename.

    Order-independent and reproducible across machines and re-downloads.
    """
    digest = hashlib.sha256(filename.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def add_gaussian_noise(clean: np.ndarray, sigma: float, *,
                       rng: np.random.RandomState | np.random.Generator | None = None,
                       filename: str | None = None) -> np.ndarray:
    """Add Gaussian noise exactly as AdaIR does, returning uint8.

    Args:
        clean: uint8 image, HWC, values in [0, 255].
        sigma: noise standard deviation on the 0-255 scale.
        rng: explicit generator. Pass a seeded ``np.random.RandomState`` to
            reproduce AdaIR's legacy stream; mutually exclusive with ``filename``.
        filename: derive a per-image seed from this name (our default mode).

    Returns:
        uint8 noisy image, same shape as ``clean``.

    Raises:
        ValueError: if both or neither of ``rng``/``filename`` are given, or the
            input is not uint8 — a float input here silently changes the result.
    """
    if (rng is None) == (filename is None):
        raise ValueError("pass exactly one of rng= or filename=")
    if clean.dtype != np.uint8:
        raise ValueError(
            f"clean must be uint8 in [0,255], got {clean.dtype}. AdaIR adds "
            "noise in 255-space; passing float would change the result.")

    if filename is not None:
        # RandomState, NOT default_rng: AdaIR calls np.random.randn, which is
        # the legacy MT19937 stream. Generator produces a different sequence, so
        # switching would silently break byte-identity with the reference.
        rng = np.random.RandomState(filename_seed(filename))  # noqa: NPY002

    # np.random.randn semantics: standard normal, same shape.
    noise = rng.standard_normal(clean.shape) if isinstance(rng, np.random.Generator) \
        else rng.randn(*clean.shape)
    noisy = np.clip(clean.astype(np.float64) + noise * sigma, 0, 255)
    return noisy.astype(np.uint8)


def legacy_rng(seed: int = 0) -> np.random.RandomState:
    """A ``RandomState`` matching AdaIR's global seeding (``test.py:114``).

    Draws must then be made in the same order AdaIR makes them for the
    realisations to match.
    """
    return np.random.RandomState(seed)
