"""The training image cache must stay inside a byte budget.

Workers are persistent and each holds its own cache, so an unbounded dict grows
to num_workers x the full decoded training set. On the development machine that
is 18.4 GB against 15.7 GB of RAM. It does not fail fast — the system pages, the
GPU starves, and a multi-day run dies partway. These tests pin the bound.
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.data.build import DenoiseTrainDataset

MB = 2 ** 20


@pytest.fixture
def image_root(tmp_path):
    """40 images of 256x256x3 = 192 KB decoded each, ~7.5 MB total."""
    root = tmp_path / "imgs"
    root.mkdir()
    rng = np.random.default_rng(0)
    for i in range(40):
        arr = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
        Image.fromarray(arr).save(root / f"{i:03d}.png")
    return root


def _ds(root, budget_gb):
    return DenoiseTrainDataset([root], patch_size=64, sigmas=(15,),
                               cache_budget_gb=budget_gb)


def test_cache_never_exceeds_budget(image_root) -> None:
    budget_mb = 1.0
    ds = _ds(image_root, budget_mb / 1024)
    for i in range(len(ds.files) * 3):          # several passes over the set
        ds._image(i)
        assert ds._cache_bytes <= ds._cache_budget, (
            f"cache {ds._cache_bytes / MB:.2f} MB exceeded budget "
            f"{ds._cache_budget / MB:.2f} MB at access {i}")


def test_accounting_matches_actual_contents(image_root) -> None:
    """The byte counter must not drift from what is really held."""
    ds = _ds(image_root, 1.0 / 1024)
    for i in range(len(ds.files) * 2):
        ds._image(i)
    actual = sum(v.nbytes for v in ds._cache.values())
    assert ds._cache_bytes == actual, (
        f"counter says {ds._cache_bytes}, contents are {actual}")


def test_unbounded_growth_is_prevented(image_root) -> None:
    """A tiny budget must hold far fewer images than the full set."""
    ds = _ds(image_root, 1.0 / 1024)            # 1 MB, ~5 images of 192 KB
    for i in range(len(ds.files)):
        ds._image(i)
    assert len(ds._cache) < len(ds.files), "cache held the entire set"
    assert len(ds._cache) >= 1, "cache held nothing at all"


def test_evicts_least_recently_used(image_root) -> None:
    ds = _ds(image_root, 1.0 / 1024)
    ds._image(0)
    for i in range(1, 4):
        ds._image(i)
    ds._image(0)                                 # refresh 0's recency
    before = list(ds._cache)
    for i in range(4, 12):                       # force evictions
        ds._image(i)
    assert before[1] not in ds._cache, "an older entry survived eviction"


def test_cache_still_returns_correct_pixels(image_root) -> None:
    """Eviction must never change what the dataset returns."""
    ds = _ds(image_root, 1.0 / 1024)
    first = ds._image(0).copy()
    for i in range(1, len(ds.files)):            # evict index 0
        ds._image(i)
    assert 0 not in ds._cache, "test did not actually evict index 0"
    np.testing.assert_array_equal(ds._image(0), first)


def test_caching_can_be_disabled(image_root) -> None:
    ds = DenoiseTrainDataset([image_root], patch_size=64, sigmas=(15,),
                             cache_images=False)
    ds._image(0)
    assert ds._cache is None


def test_nonpositive_budget_raises(image_root) -> None:
    """No silent fallback to unbounded."""
    with pytest.raises(ValueError, match="must be positive"):
        _ds(image_root, 0.0)
