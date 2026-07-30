"""Tests for student-sweep family selection and grid expansion.

Family semantics (S smallest, L largest, wide enough span) are enforced
invariants, not emergent properties of the search — see :func:`validate_family`.
"""
from __future__ import annotations

import pytest

from src.export.aihub import DeviceJobResult
from src.models.complexity import Complexity
from src.models.student_sweep import (ARMS, FamilyInvariantError, assign_family,
                                      build_grid, validate_family)


def _row(name: str, params: float, macs: float, *, latency: float | None = None,
         teacher_params: float = 28.78e6, teacher_macs: float = 161.75e9):
    """Build a SweepRow with the given params (M) and MACs (G)."""
    from src.models.student_sweep import SweepRow

    p, m = int(params * 1e6), int(macs * 1e9)
    row = SweepRow(
        name=name, width=16, block_name="b", enc_blk_nums=[1],
        middle_blk_num=1, dec_blk_nums=[1],
        complexity=Complexity(params=p, macs=m, resolution=(256, 256)),
        param_reduction=teacher_params / p, mac_reduction=teacher_macs / m,
    )
    if latency is not None:
        row.device = DeviceJobResult(name=name, profiled=True,
                                     inference_latency_ms=latency)
    return row


TARGETS = {"S": 30.0, "M": 10.0, "L": 4.0}


# --------------------------------------------------------------------------
# validate_family: the invariants themselves
# --------------------------------------------------------------------------

def test_validate_accepts_well_ordered_family() -> None:
    fam = dict(zip(ARMS, [_row("s", 2.0, 2.0), _row("m", 5.0, 5.0),
                          _row("l", 9.0, 9.0)]))
    validate_family(fam)  # must not raise


def test_validate_rejects_param_inversion() -> None:
    """S larger than L in params is the exact failure reported by review."""
    fam = dict(zip(ARMS, [_row("s", 7.02, 2.0), _row("m", 5.44, 5.0),
                          _row("l", 4.35, 9.0)]))
    with pytest.raises(FamilyInvariantError, match="params must increase"):
        validate_family(fam)


def test_validate_rejects_mac_inversion() -> None:
    fam = dict(zip(ARMS, [_row("s", 2.0, 9.0), _row("m", 5.0, 5.0),
                          _row("l", 9.0, 2.0)]))
    with pytest.raises(FamilyInvariantError, match="MACs must increase"):
        validate_family(fam)


def test_validate_rejects_narrow_span() -> None:
    """1.48x span (the reported degenerate case) must be rejected."""
    fam = dict(zip(ARMS, [_row("s", 4.35, 4.09), _row("m", 5.44, 4.67),
                          _row("l", 7.02, 6.05)]))
    with pytest.raises(FamilyInvariantError, match="span too narrow"):
        validate_family(fam, min_mac_span=2.5)


def test_validate_rejects_missing_arm() -> None:
    with pytest.raises(FamilyInvariantError, match="missing arm"):
        validate_family({"S": _row("s", 2.0, 2.0)})


# --------------------------------------------------------------------------
# assign_family: selection under the params ceiling
# --------------------------------------------------------------------------

def test_selected_family_always_satisfies_invariants() -> None:
    rows = [_row("a", 2.44, 2.13), _row("b", 3.15, 2.74), _row("c", 4.35, 4.09),
            _row("d", 5.44, 4.67), _row("e", 7.02, 6.05), _row("f", 9.68, 9.05)]
    family, _ = assign_family(rows, TARGETS, params_ceiling=10e6)
    validate_family(family)


def test_params_ceiling_excludes_oversized_configs() -> None:
    rows = [_row("small", 2.0, 2.0), _row("mid", 5.0, 5.0),
            _row("big", 9.0, 9.0), _row("huge", 29.16, 16.05)]
    family, warnings = assign_family(rows, TARGETS, params_ceiling=10e6)
    assert all(r.name != "huge" for r in family.values())
    assert any("ceiling" in w for w in warnings)


def test_unreachable_targets_do_not_break_ordering() -> None:
    """Regression: the fallback path (no target reachable) previously inverted
    the family and picked a non-nearest candidate for L.

    Every candidate here is far above every target, which is exactly the branch
    the old greedy/joint-score code mishandled.
    """
    rows = [_row("a", 2.44, 2.13), _row("b", 3.15, 2.74), _row("c", 4.35, 4.09),
            _row("d", 5.44, 4.67), _row("e", 7.02, 6.05)]
    family, warnings = assign_family(rows, TARGETS, params_ceiling=10e6,
                                     min_mac_span=2.5)
    validate_family(family, min_mac_span=2.5)
    assert any("advisory" in w for w in warnings), \
        "unmet advisory targets must be reported, not silently accepted"


def test_latency_ordering_is_enforced_when_measured() -> None:
    """A family whose 'small' arm is slower than its 'medium' arm is invalid."""
    rows = [
        _row("a", 2.0, 2.0, latency=2.5),
        _row("b", 4.0, 4.0, latency=9.0),   # MAC-ordered but pathologically slow
        _row("c", 6.0, 6.0, latency=3.0),
        _row("d", 8.0, 8.0, latency=4.0),
    ]
    family, _ = assign_family(rows, TARGETS, params_ceiling=10e6)
    lats = [family[a].device.inference_latency_ms for a in ARMS]
    assert lats == sorted(lats), f"latency must increase across arms, got {lats}"
    assert family["M"].name != "b"


def test_no_valid_family_reports_rather_than_inventing_one() -> None:
    """Too-narrow candidate pool must warn and return empty, not a bad family."""
    rows = [_row("a", 2.0, 2.0), _row("b", 2.1, 2.1), _row("c", 2.2, 2.2)]
    family, warnings = assign_family(rows, TARGETS, params_ceiling=10e6,
                                     min_mac_span=2.5)
    assert family == {}
    assert any("no S/M/L family" in w for w in warnings)


def test_wider_span_is_preferred() -> None:
    """Given the choice, the selector maximises the capacity gap."""
    rows = [_row("a", 2.0, 2.0), _row("b", 3.0, 3.0), _row("c", 4.0, 4.0),
            _row("d", 9.0, 9.0)]
    family, _ = assign_family(rows, TARGETS, params_ceiling=10e6)
    span = family["L"].complexity.macs / family["S"].complexity.macs
    assert span == pytest.approx(4.5), f"expected widest span 4.5x, got {span}"


# --------------------------------------------------------------------------
# grid expansion
# --------------------------------------------------------------------------

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
