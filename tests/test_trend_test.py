"""The trend test must be trustworthy before it is used to decide anything."""
from __future__ import annotations

import math

import pytest

from scripts.trend_test import ALPHA, mann_kendall, verdict


def test_monotonic_increase_is_detected() -> None:
    r = mann_kendall([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert r["S"] == 45 and r["tau"] == 1.0
    assert r["p"] < ALPHA and verdict(r).startswith("RISING")


def test_monotonic_decrease_is_detected() -> None:
    r = mann_kendall(list(range(10, 0, -1)))
    assert r["S"] == -45 and r["tau"] == -1.0
    assert r["p"] < ALPHA and verdict(r).startswith("FALLING")


def test_heavy_tailed_noise_is_not_called_a_trend() -> None:
    """The failure mode this exists to prevent.

    These are Q-A's own maxgn values, which swing three orders of magnitude
    between adjacent intervals and are known NOT to be a trend.
    """
    qa = [0.708, 0.381, 0.244, 0.270, 60.95, 5.394, 293.299, 0.283,
          135.185, 2360.571, 0.687, 1580.129, 3.335, 0.104, 1831.826, 5.747]
    r = mann_kendall([math.log(v) for v in qa])
    assert r["p"] >= ALPHA, (
        f"called a trend on known heavy-tailed noise (p={r['p']:.4f})")


def test_log_transform_cannot_change_the_verdict() -> None:
    """MK depends only on the sign of pairwise differences.

    log is strictly monotonic on positives, so MK(log x) == MK(x) exactly.
    Claiming the log made the test more robust would be wrong, and this pins it.
    """
    xs = [9.129, 8500.0, 3.2, 41.0, 0.5, 1200.0, 17.0, 250.0]
    raw = mann_kendall(xs)
    logged = mann_kendall([math.log(v) for v in xs])
    assert raw == logged


def test_ties_are_corrected() -> None:
    """clamp_engage_rate is frequently exactly 0.0, so ties are the norm."""
    tied = [0.0] * 6 + [0.001, 0.002]
    r = mann_kendall(tied)
    assert r["p"] < 1.0 and math.isfinite(r["Z"])
    all_tied = mann_kendall([0.0] * 8)
    assert all_tied["S"] == 0 and all_tied["Z"] == 0.0
    assert all_tied["p"] == pytest.approx(1.0)
    assert verdict(all_tied) == "no significant trend"


def test_refuses_too_few_points() -> None:
    with pytest.raises(ValueError, match="at least 4"):
        mann_kendall([1.0, 2.0, 3.0])
