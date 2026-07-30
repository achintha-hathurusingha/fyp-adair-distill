"""Tests for paired augmentation.

The point of every test here is that the degraded image and its ground truth
receive the *same* geometric operation. A desynchronised pair does not crash —
it silently costs several dB — so pairing is asserted, not assumed.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.data.transforms import (paired_augment, paired_random_crop,
                                 paired_transform)


def _pair(h: int = 64, w: int = 64):
    """A pair where clean is a known function of degraded, so desynchronisation
    is detectable: clean == degraded + 100 at every pixel."""
    rng = np.random.default_rng(0)
    degraded = rng.integers(0, 156, (h, w, 3), dtype=np.uint8)
    clean = (degraded.astype(np.int16) + 100).astype(np.uint8)
    return degraded, clean


def _still_paired(degraded: np.ndarray, clean: np.ndarray) -> bool:
    return np.array_equal(clean.astype(np.int16) - degraded.astype(np.int16),
                          np.full(degraded.shape, 100, dtype=np.int16))


def test_crop_keeps_pair_aligned() -> None:
    d, c = _pair()
    for seed in range(20):
        dd, cc = paired_random_crop(d, c, 32, np.random.default_rng(seed))
        assert dd.shape == cc.shape == (32, 32, 3)
        assert _still_paired(dd, cc), f"crop desynchronised the pair (seed {seed})"


def test_augment_keeps_pair_aligned() -> None:
    d, c = _pair()
    for seed in range(20):
        dd, cc = paired_augment(d, c, np.random.default_rng(seed))
        assert _still_paired(dd, cc), f"augment desynchronised the pair (seed {seed})"


def test_full_transform_keeps_pair_aligned() -> None:
    d, c = _pair()
    for seed in range(20):
        dd, cc = paired_transform(d, c, patch_size=32,
                                  rng=np.random.default_rng(seed))
        assert dd.shape == (32, 32, 3)
        assert _still_paired(dd, cc)


def test_crop_is_reproducible_under_a_fixed_seed() -> None:
    d, c = _pair()
    a = paired_random_crop(d, c, 32, np.random.default_rng(7))
    b = paired_random_crop(d, c, 32, np.random.default_rng(7))
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_crop_actually_varies_with_seed() -> None:
    """Guards against a degenerate crop that always returns the same window."""
    d, c = _pair()
    windows = {paired_random_crop(d, c, 16, np.random.default_rng(s))[0].tobytes()
               for s in range(30)}
    assert len(windows) > 1


def test_augment_actually_varies_with_seed() -> None:
    d, c = _pair()
    outs = {paired_augment(d, c, np.random.default_rng(s))[0].tobytes()
            for s in range(30)}
    assert len(outs) > 1


def test_rotation_can_transpose_dimensions() -> None:
    """90-degree rotation of a non-square patch must swap H and W."""
    d, c = _pair(32, 64)
    shapes = {paired_augment(d, c, np.random.default_rng(s))[0].shape
              for s in range(30)}
    assert (64, 32, 3) in shapes and (32, 64, 3) in shapes


def test_shape_mismatch_raises() -> None:
    d, _ = _pair(64, 64)
    _, c = _pair(32, 32)
    with pytest.raises(ValueError, match="paired inputs must match"):
        paired_random_crop(d, c, 16, np.random.default_rng(0))


def test_patch_larger_than_image_raises() -> None:
    d, c = _pair(16, 16)
    with pytest.raises(ValueError, match="cannot crop"):
        paired_random_crop(d, c, 32, np.random.default_rng(0))


def test_output_is_contiguous() -> None:
    """Rotations produce views; downstream torch.from_numpy needs contiguity."""
    d, c = _pair()
    for seed in range(10):
        dd, cc = paired_augment(d, c, np.random.default_rng(seed))
        assert dd.flags["C_CONTIGUOUS"] and cc.flags["C_CONTIGUOUS"]
