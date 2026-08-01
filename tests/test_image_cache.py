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


def test_clamp_bound_is_read_at_construction_not_import() -> None:
    """Overriding AFFINE_CLAMP_BOUND must actually change the bound.

    Regression: the bound was a default argument (`bound=AFFINE_CLAMP_BOUND`),
    which Python binds once at def time. Reassigning the module constant had no
    effect, so a sweep over bounds silently returned identical results at every
    setting.
    """
    import torch as _torch

    import src.models.norms as norms

    original = norms.AFFINE_CLAMP_BOUND
    try:
        norms.AFFINE_CLAMP_BOUND = 2.0
        n = norms.build_norm("affine_clamp", 4)
        assert n.bound == 2.0, f"bound is {n.bound}, expected the override 2.0"
        out = n(_torch.full((1, 4, 2, 2), 1e6))
        assert float(out.abs().max()) == 2.0
    finally:
        norms.AFFINE_CLAMP_BOUND = original

    assert norms.build_norm("affine_clamp", 4).bound == original


def test_clamp_engagement_tracking_is_off_by_default_and_stays_out_of_state_dict() -> None:
    """Tracking must be opt-in, and must not change weight compatibility.

    The whole fix comparison rests on affine / affine_clamp / layernorm2d having
    identical state_dict shapes, so spike-state weights load into each. A buffer
    for the counters would break that.
    """
    import torch as _torch

    import src.models.norms as norms

    n = norms.build_norm("affine_clamp", 4, clamp_bound=1.0)
    n(_torch.rand(1, 4, 4, 4) * 100)
    assert not hasattr(n, "engaged"), "tracking ran while disabled"
    assert list(n.state_dict()) == ["weight", "bias"], list(n.state_dict())

    ln = norms.build_norm("layernorm2d", 4)
    af = norms.build_norm("affine", 4)
    shapes = lambda m: {k: tuple(v.shape) for k, v in m.state_dict().items()}
    assert shapes(n) == shapes(ln) == shapes(af)


def test_clamp_engagement_counts_only_when_it_fires() -> None:
    import torch as _torch

    import src.models.norms as norms

    original = norms.TRACK_CLAMP_ENGAGEMENT
    try:
        norms.TRACK_CLAMP_ENGAGEMENT = True
        n = norms.build_norm("affine_clamp", 4, clamp_bound=1.0)
        n(_torch.full((1, 4, 4, 4), 0.5))          # under the bound
        assert getattr(n, "forwards", 0) == 1
        assert getattr(n, "engaged", 0) == 0, "counted an engagement that did not happen"
        n(_torch.full((1, 4, 4, 4), 50.0))         # over the bound
        assert getattr(n, "engaged", 0) == 1
        assert getattr(n, "elements_clamped", 0) == 4 * 4 * 4
    finally:
        norms.TRACK_CLAMP_ENGAGEMENT = original


def test_clamp_engagement_excludes_validation_forwards(tmp_path, monkeypatch) -> None:
    """Engagement rate must measure TRAINING forwards only.

    Regression: the counters were read after activation_stats() and validate(),
    so one diagnostic forward plus one per BSD68 image — at full resolution
    rather than 128px crops — were folded into the rate. That measures
    engagement on a different input distribution than training actually sees,
    which is the entire point of the metric.
    """
    import torch as _torch

    import src.models.norms as norms
    from src.models.nafnet import NAFNet
    from src.train.trainer import Trainer

    original = norms.TRACK_CLAMP_ENGAGEMENT
    try:
        norms.TRACK_CLAMP_ENGAGEMENT = True
        model = NAFNet(width=4, enc_blk_nums=[1], middle_blk_num=1,
                       dec_blk_nums=[1], norm_type="layernorm2d",
                       full_res_norm_type="affine_clamp", clamp_bound=1e9)
        batches = [(_torch.rand(2, 3, 32, 32), _torch.rand(2, 3, 32, 32), 15)
                   for _ in range(4)]
        cfg = {"optim": {"lr": 1e-3}, "schedule": {"total_iters": 4,
                                                   "warmup_iters": 1},
               "train": {"accum_steps": 1, "amp": False, "val_every": 4,
                         "ckpt_every": 10 ** 9},
               "loss": {"name": "charbonnier"}}
        t = Trainer(model, batches, cfg, tmp_path, device="cpu")

        seen = {}
        real = t._clamp_stats

        def spy():
            out = real()
            seen.update(out)
            return out

        monkeypatch.setattr(t, "_clamp_stats", spy)
        state = t.train()

        row = state.history[-1]
        n_clamps = sum(1 for m in model.modules()
                       if isinstance(m, norms.AffineClampNorm2d))
        # 4 training steps x 1 micro-batch, and nothing else.
        assert row["clamp_engage_rate"] == 0.0
        assert "clamp_elements" in row
    finally:
        norms.TRACK_CLAMP_ENGAGEMENT = original
