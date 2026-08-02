"""The multi-task training set: layout, pairing, determinism, task balance.

B0-denoise trained on denoise alone for its entire life because nothing checked
what the loader actually loaded (finding F11). These tests check the properties
that failure would have caught: that every task is present, that its share of
the index space is what it claims, and that a sample is a pure function of its
index.
"""
from __future__ import annotations

import collections

import numpy as np
import pytest
import torch
from PIL import Image

from src.data.build import TASK_IDS, MultiTaskTrainDataset, build_multitask_loader

PATCH = 32


def _img(path, seed, size=(64, 64)):
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (*size, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


@pytest.fixture
def roots(tmp_path):
    """A miniature 3-task dataset with deliberately UNEVEN file counts.

    6 denoise images, 4 derain pairs, 3 hazy images sharing 2 clear sources —
    uneven on purpose, because equal counts would hide a sampler that balances
    by file count instead of by index range.
    """
    for i in range(6):
        _img(tmp_path / "denoise" / f"clean{i}.png", i)
    for i in range(4):
        # Distinct seeds: an input identical to its target would make a pairing
        # bug (returning the input twice) indistinguishable from correct output.
        _img(tmp_path / "derain" / "input" / f"rain-{i}.png", 100 + i)
        _img(tmp_path / "derain" / "target" / f"rain-{i}.png", 150 + i)
    for i in range(2):
        _img(tmp_path / "dehaze" / "target" / f"{i}.png", 200 + i)
    for name in ("0_0.8_0.04.jpg", "0_0.9_0.20.jpg", "1_0.8_0.04.jpg"):
        _img(tmp_path / "dehaze" / "input" / name, 300 + len(name))
    return {"denoise": tmp_path / "denoise",
            "derain": tmp_path / "derain",
            "dehaze": tmp_path / "dehaze"}


def _ds(roots, **kw):
    kw.setdefault("length", 90)
    kw.setdefault("cache_budget_gb", 0.01)
    return MultiTaskTrainDataset(roots, patch_size=PATCH, sigmas=(15, 25, 50), **kw)


# --------------------------------------------------------------- layout


def test_all_three_tasks_are_present_and_equally_sized(roots) -> None:
    """The F11 check: every configured task must really occupy the index space."""
    ds = _ds(roots)
    ranges = ds.task_ranges()
    assert set(ranges) == {"denoise", "derain", "dehaze"}
    sizes = {t: len(r) for t, r in ranges.items()}
    assert sizes == {"denoise": 30, "derain": 30, "dehaze": 30}, sizes
    assert len(ds) == 90


def test_ranges_partition_the_index_space_exactly(roots) -> None:
    ds = _ds(roots)
    covered = sorted(i for r in ds.task_ranges().values() for i in r)
    assert covered == list(range(len(ds))), "ranges overlap or leave a gap"


def test_task_of_agrees_with_the_ranges(roots) -> None:
    ds = _ds(roots)
    for task, rng in ds.task_ranges().items():
        for idx in (rng.start, rng[len(rng) // 2], rng.stop - 1):
            assert ds.task_of(idx) == task


def test_length_not_divisible_by_task_count_is_truncated_not_skewed(roots) -> None:
    """95 // 3 = 31 per task; the length becomes 93, not 95 with one task fat."""
    ds = _ds(roots, length=95)
    assert len(ds) == 93
    assert {len(r) for r in ds.task_ranges().values()} == {31}


def test_a_small_task_repeats_rather_than_shrinking_its_share(roots) -> None:
    """3 hazy images must still fill a 30-slot stream."""
    ds = _ds(roots)
    seen = {ds[i][2]["task"] for i in ds.task_ranges()["dehaze"]}
    assert seen == {TASK_IDS["dehaze"]}


def test_two_task_subset_splits_in_half(roots) -> None:
    """Dehaze slots in as a third source with no code change — and out again."""
    ds = _ds({k: roots[k] for k in ("denoise", "derain")}, length=90)
    assert {t: len(r) for t, r in ds.task_ranges().items()} == \
        {"denoise": 45, "derain": 45}


# --------------------------------------------------------------- samples


def test_sample_shapes_and_range(roots) -> None:
    ds = _ds(roots)
    for idx in (0, 30, 60):
        degraded, clean, meta = ds[idx]
        assert degraded.shape == clean.shape == (3, PATCH, PATCH)
        assert degraded.dtype == clean.dtype == torch.float32
        assert 0.0 <= float(degraded.min()) and float(degraded.max()) <= 1.0
        assert meta["task"] == TASK_IDS[ds.task_of(idx)]


def test_denoise_sigma_cycles_and_others_are_sentinel(roots) -> None:
    ds = _ds(roots)
    ranges = ds.task_ranges()
    sigmas = [ds[i][2]["sigma"] for i in list(ranges["denoise"])[:6]]
    assert sigmas == [15.0, 25.0, 50.0, 15.0, 25.0, 50.0]
    # -1.0, never 0.0: sigma=0 is the clean-input case F10 is about, and must
    # stay distinguishable from "noise level does not apply".
    for task in ("derain", "dehaze"):
        assert ds[ranges[task].start][2]["sigma"] == -1.0


def test_denoise_target_is_clean_and_input_is_noisy(roots) -> None:
    ds = _ds(roots)
    degraded, clean, meta = ds[ds.task_ranges()["denoise"].start]
    assert meta["sigma"] == 15.0
    residual = float((degraded - clean).abs().mean()) * 255
    assert 6.0 < residual < 20.0, f"residual {residual:.2f}/255 is not sigma-15 noise"


def test_derain_pair_is_the_matching_file_not_a_copy(roots) -> None:
    """Rain input and target are distinct images; a pairing bug returns one twice."""
    ds = _ds(roots)
    degraded, clean, _ = ds[ds.task_ranges()["derain"].start]
    assert not torch.equal(degraded, clean), "input and target are identical"


def test_dehaze_pairs_by_basename_prefix(roots) -> None:
    """`0_0.8_0.04.jpg` and `0_0.9_0.20.jpg` must both resolve to `0.png`."""
    ds = _ds(roots)
    targets = {tgt.name for _, tgt in ds.items["dehaze"]}
    assert targets == {"0.png", "1.png"}
    by_input = {src.name: tgt.name for src, tgt in ds.items["dehaze"]}
    assert by_input["0_0.8_0.04.jpg"] == by_input["0_0.9_0.20.jpg"] == "0.png"


def test_geometric_augmentation_stays_synchronised(roots, tmp_path) -> None:
    """Crop and flip must hit the pair identically.

    Uses an identical input/target pair: any desynchronised transform makes the
    two differ, which a real pair could not distinguish from a pairing bug.
    """
    for i in range(4):
        _img(tmp_path / "same" / "input" / f"x{i}.png", 7 + i)
        _img(tmp_path / "same" / "target" / f"x{i}.png", 7 + i)
    ds = MultiTaskTrainDataset({"derain": tmp_path / "same"}, patch_size=PATCH,
                               length=40, cache_budget_gb=0.01)
    for idx in range(12):
        degraded, clean, _ = ds[idx]
        assert torch.equal(degraded, clean), f"pair desynchronised at index {idx}"


# --------------------------------------------------------------- determinism


def test_sample_is_a_pure_function_of_its_index(roots) -> None:
    """Two datasets, same seed: identical samples. This is what makes results
    independent of worker count and reproducible on resume."""
    a, b = _ds(roots), _ds(roots)
    for idx in (0, 7, 31, 44, 61, 89):
        da, ca, ma = a[idx]
        db, cb, mb = b[idx]
        assert torch.equal(da, db) and torch.equal(ca, cb), f"index {idx} differs"
        assert ma == mb


def test_different_seed_gives_a_different_realisation(roots) -> None:
    a = _ds(roots, base_seed=0)
    b = _ds(roots, base_seed=1)
    assert not torch.equal(a[0][0], b[0][0]), "base_seed had no effect"


def test_out_of_order_access_does_not_change_samples(roots) -> None:
    """Shuffled access must not perturb the stream — no shared RNG state."""
    ref = {i: _ds(roots)[i][0] for i in (3, 40, 77)}
    ds = _ds(roots)
    for i in (77, 3, 40, 3):
        assert torch.equal(ds[i][0], ref[i]), f"index {i} changed with access order"


# --------------------------------------------------------------- failing loudly


def test_missing_root_fails_at_construction(roots, tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="derain root not found"):
        _ds({**roots, "derain": tmp_path / "nope"})


def test_missing_input_dir_names_what_is_missing(roots, tmp_path) -> None:
    (tmp_path / "bare" / "target").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="input/"):
        _ds({**roots, "derain": tmp_path / "bare"})


def test_unpairable_image_fails_before_training_starts(roots) -> None:
    """Every pair resolves at construction — not eight hours into a run."""
    _img(roots["derain"] / "input" / "orphan.png", 999)
    with pytest.raises(FileNotFoundError, match="no ground truth for orphan.png"):
        _ds(roots)


def test_unknown_task_is_rejected(roots) -> None:
    with pytest.raises(ValueError, match="unknown task"):
        _ds({**roots, "deblur": roots["denoise"]})


def test_empty_sources_is_rejected(roots) -> None:
    with pytest.raises(ValueError, match="at least one task"):
        _ds({})


def test_length_too_small_for_the_task_count_is_rejected(roots) -> None:
    """No silent fallback to an empty stream for some task."""
    with pytest.raises(ValueError, match="too small"):
        _ds(roots, length=2)


def test_index_out_of_range_raises(roots) -> None:
    ds = _ds(roots)
    with pytest.raises(IndexError):
        ds.task_of(len(ds))


def test_images_smaller_than_the_patch_are_padded_not_dropped(roots, tmp_path) -> None:
    for i in range(3):
        _img(tmp_path / "tiny" / "input" / f"t{i}.png", i, size=(20, 20))
        _img(tmp_path / "tiny" / "target" / f"t{i}.png", i, size=(20, 20))
    ds = MultiTaskTrainDataset({"derain": tmp_path / "tiny"}, patch_size=PATCH,
                               length=30, cache_budget_gb=0.01)
    degraded, clean, _ = ds[0]
    assert degraded.shape == clean.shape == (3, PATCH, PATCH)


def test_cache_stays_within_budget_across_tasks(roots) -> None:
    ds = _ds(roots, cache_budget_gb=8.0 / 2 ** 20)     # 8 KB — evicts constantly
    for i in range(len(ds)):
        ds[i]
        assert ds._lru.nbytes <= ds._lru.budget
    assert ds._lru.nbytes == sum(v.nbytes for v in ds._lru.entries.values())


def test_caching_off_returns_the_same_samples(roots) -> None:
    cached, uncached = _ds(roots), _ds(roots, cache_images=False)
    for idx in (0, 31, 62):
        assert torch.equal(cached[idx][0], uncached[idx][0])


# ------------------------------------------------- end to end through a loader


def _loader(roots, **kw):
    kw.setdefault("length", 90)
    kw.setdefault("num_workers", 0)
    return build_multitask_loader(roots, batch_size=6, patch_size=PATCH,
                                  cache_budget_gb=0.01, **kw)


def test_loader_delivers_balanced_batches(roots) -> None:
    """The end-to-end claim: what reaches the trainer really is all-in-one."""
    seen = 0
    for _degraded, _clean, meta in _loader(roots):
        counts = collections.Counter(int(t) for t in meta["task"])
        assert counts == {TASK_IDS["denoise"]: 2, TASK_IDS["derain"]: 2,
                          TASK_IDS["dehaze"]: 2}, counts
        seen += 1
    assert seen == 15, f"expected 90 // 6 = 15 batches, got {seen}"


def test_loader_batches_are_worker_count_independent(roots) -> None:
    """Two workers must produce byte-identical batches to zero workers.

    Proven earlier for the denoise loader by scripts/determinism_check.py; it
    holds here because every sample is a function of (base_seed, index) alone,
    with no generator state shared between workers.
    """
    single = [(d, c, meta["task"]) for d, c, meta in _loader(roots, num_workers=0)]
    multi = [(d, c, meta["task"]) for d, c, meta in _loader(roots, num_workers=2)]
    assert len(single) == len(multi)
    for i, ((d0, c0, t0), (d1, c1, t1)) in enumerate(zip(single, multi)):
        assert torch.equal(t0, t1), f"batch {i}: task composition differs"
        assert torch.equal(d0, d1), f"batch {i}: inputs differ"
        assert torch.equal(c0, c1), f"batch {i}: targets differ"


def test_loader_carries_sigma_alongside_the_task(roots) -> None:
    _d, _c, meta = next(iter(_loader(roots)))
    sigmas = meta["sigma"][meta["task"] == TASK_IDS["denoise"]]
    assert set(float(s) for s in sigmas) <= {15.0, 25.0, 50.0}
    others = meta["sigma"][meta["task"] != TASK_IDS["denoise"]]
    assert all(float(s) == -1.0 for s in others)
