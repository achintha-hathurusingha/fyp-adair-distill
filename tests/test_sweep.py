"""Tests for student-sweep family assignment and grid expansion."""
from __future__ import annotations

import pytest

from src.models.complexity import Complexity
from src.models.student_sweep import SweepRow, assign_family, build_grid


def _row(name: str, mac_reduction: float, params: int = 1_000_000) -> SweepRow:
    return SweepRow(
        name=name, width=16, block_name="b", enc_blk_nums=[1],
        middle_blk_num=1, dec_blk_nums=[1],
        complexity=Complexity(params=params, macs=10**9, resolution=(256, 256)),
        param_reduction=10.0, mac_reduction=mac_reduction,
    )


TARGETS = {"S": 30.0, "M": 10.0, "L": 4.0}


def test_family_is_monotonic_when_targets_reachable() -> None:
    rows = [_row("s", 30.0), _row("m", 10.0), _row("l", 4.0), _row("x", 55.0)]
    family, warnings = assign_family(rows, TARGETS)
    assert family["S"].mac_reduction == 30.0
    assert family["M"].mac_reduction == 10.0
    assert family["L"].mac_reduction == 4.0
    assert not warnings


def test_family_stays_monotonic_when_targets_unreachable() -> None:
    """Regression: greedy matching used to invert the family (L smaller than S).

    All candidates here sit far above every target, so a naive
    closest-to-target assignment can pick L more-reduced (smaller) than S.
    The ordered fit must preserve S >= M >= L in MAC reduction.
    """
    rows = [_row("a", 76.1), _row("b", 58.9), _row("c", 39.6),
            _row("d", 34.6), _row("e", 26.7)]
    family, warnings = assign_family(rows, TARGETS)
    assert family["S"].mac_reduction >= family["M"].mac_reduction
    assert family["M"].mac_reduction >= family["L"].mac_reduction
    assert warnings, "unreachable targets must be reported, not hidden"


def test_family_arms_are_distinct() -> None:
    rows = [_row("a", 76.1), _row("b", 58.9), _row("c", 39.6)]
    family, _ = assign_family(rows, TARGETS)
    assert len({r.name for r in family.values()}) == 3


def test_ineligible_rows_are_excluded() -> None:
    good = _row("good", 30.0)
    bad = _row("bad", 10.0)
    bad.notes.append("fails-compression-rule")
    family, warnings = assign_family([good, bad], TARGETS)
    assert all(r.name != "bad" for r in family.values())
    assert warnings  # not enough eligible candidates for three arms


def test_too_few_candidates_warns_rather_than_crashing() -> None:
    family, warnings = assign_family([_row("only", 30.0)], TARGETS)
    assert warnings
    assert len(family) <= 3


def test_build_grid_expands_widths_times_blocks() -> None:
    cfg = {
        "widths": [16, 32],
        "block_configs": [
            {"name": "b8", "enc_blk_nums": [1, 1, 1, 8], "middle_blk_num": 2,
             "dec_blk_nums": [1, 1, 1, 1]},
            {"name": "sidd", "enc_blk_nums": [2, 2, 4, 8], "middle_blk_num": 12,
             "dec_blk_nums": [2, 2, 2, 2]},
        ],
    }
    grid = build_grid(cfg)
    assert len(grid) == 4
    assert {g["name"] for g in grid} == {"w16_b8", "w16_sidd", "w32_b8", "w32_sidd"}


def test_build_grid_requires_keys() -> None:
    with pytest.raises(KeyError):
        build_grid({"widths": [16]})  # missing block_configs
