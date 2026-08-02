"""Per-batch task balance, asserted rather than assumed.

B0-denoise trained on one task for a full run and nothing noticed (F11). The
answer is not vigilance, it is a property that can be checked: every batch holds
a known share of every task, and the check is cheap enough to run in CI.
"""
from __future__ import annotations

from collections import Counter

import pytest

from src.data.sampler import BalancedTaskBatchSampler

RANGES = {"denoise": range(0, 300), "derain": range(300, 600),
          "dehaze": range(600, 900)}


def _sampler(batch_size=15, num_batches=20, **kw):
    return BalancedTaskBatchSampler(RANGES, batch_size,
                                    num_batches=num_batches, **kw)


def _task_of(idx: int) -> str:
    for task, rng in RANGES.items():
        if idx in rng:
            return task
    raise AssertionError(f"index {idx} belongs to no task")


# ------------------------------------------------------------- balance


def test_every_batch_holds_an_equal_share_when_divisible() -> None:
    for batch in _sampler(batch_size=15):
        counts = Counter(_task_of(i) for i in batch)
        assert counts == {"denoise": 5, "derain": 5, "dehaze": 5}, counts


def test_every_batch_is_exactly_batch_size() -> None:
    for batch in _sampler(batch_size=16):
        assert len(batch) == 16


def test_indivisible_batch_shares_the_leftovers_fairly(  ) -> None:
    """16 over 3 tasks: 5 each plus one extra, which must ROTATE.

    A fixed recipient would give one task a permanent 6.7% advantage — small
    enough to survive review and large enough to matter over 300k iterations.
    """
    batches = list(_sampler(batch_size=16, num_batches=30))
    for batch in batches:
        counts = Counter(_task_of(i) for i in batch)
        assert sorted(counts.values()) == [5, 5, 6], counts
    totals = Counter(_task_of(i) for b in batches for i in b)
    assert len(set(totals.values())) == 1, f"unequal over the epoch: {totals}"


def test_leftover_rotation_over_two_extras() -> None:
    """17 over 3 tasks: two extras per batch, still equal across an epoch."""
    batches = list(_sampler(batch_size=17, num_batches=30))
    for batch in batches:
        assert sorted(Counter(_task_of(i) for i in batch).values()) == [5, 6, 6]
    totals = Counter(_task_of(i) for b in batches for i in b)
    assert len(set(totals.values())) == 1, f"unequal over the epoch: {totals}"


def test_quotas_sum_to_batch_size_for_every_batch() -> None:
    s = _sampler(batch_size=16, num_batches=40)
    for b in range(len(s)):
        assert sum(s.quotas(b).values()) == 16


def test_two_task_mix_splits_in_half() -> None:
    s = BalancedTaskBatchSampler({k: RANGES[k] for k in ("denoise", "derain")},
                                 10, num_batches=5)
    for batch in s:
        assert Counter(_task_of(i) for i in batch) == {"denoise": 5, "derain": 5}


def test_indices_stay_inside_their_task_range() -> None:
    for batch in _sampler(batch_size=16, num_batches=25):
        for idx in batch:
            _task_of(idx)                      # raises if outside every range


def test_length_is_the_batch_budget() -> None:
    s = _sampler(num_batches=37)
    assert len(s) == 37 == len(list(s))


# ------------------------------------------------------------- coverage


def test_walks_a_permutation_before_repeating_anything() -> None:
    """Coverage, not i.i.d. draws: every index of a task is used once before any
    is used twice. Independent draws would leave images unseen for a whole
    epoch while repeating others."""
    # 300 denoise indices, 5 per batch -> exactly 60 batches to cover the range.
    drawn = [i for b in _sampler(batch_size=15, num_batches=60) for i in b
             if _task_of(i) == "denoise"]
    assert len(drawn) == 300
    assert sorted(drawn) == list(RANGES["denoise"]), "coverage was not exhaustive"


def test_reshuffles_rather_than_repeating_the_same_order() -> None:
    batches = list(_sampler(batch_size=15, num_batches=120))
    first = [i for b in batches[:60] for i in b if _task_of(i) == "denoise"]
    second = [i for b in batches[60:] for i in b if _task_of(i) == "denoise"]
    assert sorted(first) == sorted(second), "second cycle changed the coverage"
    assert first != second, "the second cycle repeated the first order verbatim"


# ------------------------------------------------------------- determinism


def test_same_seed_gives_identical_batches() -> None:
    a = list(_sampler(base_seed=0))
    b = list(_sampler(base_seed=0))
    assert a == b


def test_different_seed_gives_different_batches() -> None:
    assert list(_sampler(base_seed=0)) != list(_sampler(base_seed=1))


def test_set_epoch_reshuffles_and_is_never_implicit() -> None:
    s = _sampler()
    first = list(s)
    assert list(s) == first, "iterating twice must not change the batches"
    s.set_epoch(1)
    assert list(s) != first, "set_epoch did not reshuffle"


def test_seeding_survives_a_fresh_process() -> None:
    """Regression: the permutation was seeded on hash(task_name).

    Python salts str hashing per process, so that seed differed on every launch
    — batches would have been irreproducible across runs while looking perfectly
    deterministic within one. Run the sampler in a subprocess with a different
    PYTHONHASHSEED and require the same first batch.
    """
    import os
    import subprocess
    import sys

    code = (
        "from src.data.sampler import BalancedTaskBatchSampler as S;"
        "r={'denoise':range(0,300),'derain':range(300,600),'dehaze':range(600,900)};"
        "print(next(iter(S(r,15,num_batches=3,base_seed=0))))"
    )
    outs = []
    for salt in ("1", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": salt}
        outs.append(subprocess.run([sys.executable, "-c", code], check=True,
                                   capture_output=True, text=True,
                                   env=env).stdout.strip())
    assert outs[0] == outs[1], f"batches depend on the hash salt: {outs}"
    assert outs[0] == str(next(iter(_sampler(batch_size=15, num_batches=3)))), (
        "subprocess disagrees with in-process sampling")


# ------------------------------------------------------------- failing loudly


def test_batch_too_small_to_hold_every_task_is_rejected() -> None:
    """No silent dropping of a task — that is the F11 failure in miniature."""
    with pytest.raises(ValueError, match="cannot hold all 3 tasks"):
        _sampler(batch_size=2)


def test_empty_task_ranges_rejected() -> None:
    with pytest.raises(ValueError, match="nothing to sample"):
        BalancedTaskBatchSampler({}, 4, num_batches=1)


def test_a_task_owning_no_indices_is_rejected() -> None:
    with pytest.raises(ValueError, match="owns no indices"):
        BalancedTaskBatchSampler({"denoise": range(0, 10), "derain": range(5, 5)},
                                 4, num_batches=1)


def test_nonpositive_num_batches_rejected() -> None:
    with pytest.raises(ValueError, match="num_batches must be"):
        _sampler(num_batches=0)
