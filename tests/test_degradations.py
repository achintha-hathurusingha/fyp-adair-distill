"""Tests for degradation synthesis.

The decisive test is :func:`test_byte_identical_to_adair_reference`, which runs
AdaIR's own noise code inline and requires an exact match. If synthesis drifts,
our inputs stop being the inputs the published numbers were produced from.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.data.degradations import (SIGMAS, add_gaussian_noise, filename_seed,
                                   legacy_rng)


def _clean(seed: int = 0, shape=(32, 32, 3)) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, shape, dtype=np.uint8)


# --------------------------------------------------------------------------
# byte-identity with the reference implementation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("sigma", SIGMAS)
def test_byte_identical_to_adair_reference(sigma: int) -> None:
    """Reproduce AdaIR's _add_gaussian_noise (dataset_utils.py:243-246) exactly."""
    clean = _clean()

    # Reference: AdaIR's code, verbatim.
    ref_state = np.random.RandomState(1234)
    reference = np.clip(
        clean + ref_state.randn(*clean.shape) * sigma, 0, 255).astype(np.uint8)

    ours = add_gaussian_noise(clean, sigma, rng=np.random.RandomState(1234))
    assert np.array_equal(ours, reference)


def test_scale_choice_is_equivalent_but_quantisation_is_not() -> None:
    """Pins what actually differs, correcting an intuitive but wrong framing.

    Adding noise in 255-space vs [0,1]-space is *mathematically equivalent* and
    yields byte-identical uint8. The real distinction is whether the result is
    cast to uint8 at all. We replicate AdaIR's cast for fidelity; the numerical
    effect is ~0.003 dB, far below the +/-0.10 dB gate.
    """
    clean = _clean()
    ours = add_gaussian_noise(clean, 25, rng=np.random.RandomState(7))

    # Same maths expressed in [0,1] space, still cast to uint8 -> identical.
    st = np.random.RandomState(7)
    equivalent = (np.clip(clean / 255.0 + st.randn(*clean.shape) * (25 / 255.0),
                          0, 1) * 255).astype(np.uint8)
    assert np.array_equal(ours, equivalent), "scale choice must not matter"

    # Never quantising DOES differ -- that is the real convention.
    st = np.random.RandomState(7)
    unquantised = np.clip(clean / 255.0 + st.randn(*clean.shape) * (25 / 255.0), 0, 1)
    assert not np.array_equal(ours / 255.0, unquantised)


def test_output_is_quantised_uint8() -> None:
    out = add_gaussian_noise(_clean(), 15, filename="a.png")
    assert out.dtype == np.uint8
    assert out.min() >= 0 and out.max() <= 255


def test_float_input_rejected() -> None:
    """A float input would silently change the result -- must raise."""
    with pytest.raises(ValueError, match="must be uint8"):
        add_gaussian_noise(_clean().astype(np.float32), 15, filename="a.png")


# --------------------------------------------------------------------------
# seeding behaviour
# --------------------------------------------------------------------------

def test_filename_seeding_is_deterministic() -> None:
    a = add_gaussian_noise(_clean(), 25, filename="img_042.png")
    b = add_gaussian_noise(_clean(), 25, filename="img_042.png")
    assert np.array_equal(a, b)


def test_filename_seeding_is_order_independent() -> None:
    """The property sorting cannot give you: identical result regardless of
    what else was generated before."""
    clean = _clean()
    first = add_gaussian_noise(clean, 25, filename="target.png")

    for other in ("a.png", "b.png", "c.png"):      # consume other realisations
        add_gaussian_noise(clean, 25, filename=other)
    later = add_gaussian_noise(clean, 25, filename="target.png")

    assert np.array_equal(first, later)


def test_different_filenames_give_different_noise() -> None:
    clean = _clean()
    a = add_gaussian_noise(clean, 25, filename="one.png")
    b = add_gaussian_noise(clean, 25, filename="two.png")
    assert not np.array_equal(a, b)


def test_filename_seed_is_stable_across_runs() -> None:
    """Hardcoded so a change to the hashing scheme is caught, not absorbed."""
    assert filename_seed("0001.png") == int(
        __import__("hashlib").sha256(b"0001.png").hexdigest()[:8], 16)
    assert 0 <= filename_seed("anything.png") < 2 ** 32


def test_global_mode_matches_sequential_draws() -> None:
    """Legacy mode must consume one shared stream, as AdaIR does."""
    clean = _clean()
    rng = legacy_rng(0)
    ours = [add_gaussian_noise(clean, 15, rng=rng) for _ in range(3)]

    ref = np.random.RandomState(0)
    expected = [np.clip(clean + ref.randn(*clean.shape) * 15, 0, 255).astype(np.uint8)
                for _ in range(3)]
    for a, b in zip(ours, expected):
        assert np.array_equal(a, b)


def test_seeding_modes_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="exactly one of"):
        add_gaussian_noise(_clean(), 15)
    with pytest.raises(ValueError, match="exactly one of"):
        add_gaussian_noise(_clean(), 15, rng=legacy_rng(0), filename="a.png")


@pytest.mark.parametrize("sigma", SIGMAS)
def test_higher_sigma_is_noisier(sigma: int) -> None:
    clean = np.full((64, 64, 3), 128, dtype=np.uint8)
    out = add_gaussian_noise(clean, sigma, filename="flat.png")
    measured = float(np.std(out.astype(np.float64) - 128.0))
    # Clipping biases the estimate low at high sigma; allow generous tolerance.
    assert 0.7 * sigma <= measured <= 1.3 * sigma
