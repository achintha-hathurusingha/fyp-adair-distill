"""Regression guard on the LOCKED architecture (Task 1.5b, 2026-07-31).

The normalization scheme and the S/M/L family are the outcome of an expensive
ablation — six training arms and 36 on-device profiling jobs. Changing either
silently would invalidate every number measured against them, including B0 and
every Phase 02 distillation delta.

These tests fail loudly if the lock moves.
"""
from __future__ import annotations

import pytest

from src.models.nafnet import NAFNet
from src.models.norms import AffineNorm2d, LayerNorm2d
from src.utils.config import load_yaml

LOCKED = "configs/model/nafnet_locked.yaml"

#: The locked family, re-selected on N-F latency (reports/family_reselection.md).
EXPECTED_FAMILY = {
    "S": ("w16_b8", 16, [1, 1, 1, 8], 2, [1, 1, 1, 1]),
    "M": ("w16_sidd", 16, [2, 2, 4, 8], 12, [2, 2, 2, 2]),
    "L": ("w24_b28", 24, [1, 1, 1, 28], 1, [1, 1, 1, 1]),
}


def test_locked_norm_is_N_F() -> None:
    """N-F: LayerNorm2d deeper, affine at full resolution.

    Q-A 31.019 dB / Q-F 31.014 dB / Q-E diverged. N-F costs 0.005 dB for a
    1.59x speedup; N-E's 2.34x is unreachable (findings F6).
    """
    cfg = load_yaml(LOCKED)
    assert cfg["norm_type"] == "layernorm2d"
    assert cfg["full_res_norm_type"] == "affine"


def test_built_model_places_norms_as_locked() -> None:
    """The config must actually produce the intended module layout."""
    cfg = load_yaml(LOCKED)
    model = NAFNet(width=cfg["width"], enc_blk_nums=cfg["enc_blk_nums"],
                   middle_blk_num=cfg["middle_blk_num"],
                   dec_blk_nums=cfg["dec_blk_nums"],
                   norm_type=cfg["norm_type"],
                   full_res_norm_type=cfg["full_res_norm_type"])
    # full resolution -> affine only
    assert isinstance(model.encoders[0][0].norm1, AffineNorm2d)
    assert isinstance(model.decoders[-1][0].norm1, AffineNorm2d)
    # deeper stages -> full LayerNorm
    assert isinstance(model.encoders[1][0].norm1, LayerNorm2d)
    assert isinstance(model.middle_blks[0].norm1, LayerNorm2d)


def test_default_geometry_is_the_M_arm() -> None:
    cfg = load_yaml(LOCKED)
    _, width, enc, middle, dec = EXPECTED_FAMILY["M"]
    assert (cfg["width"], cfg["enc_blk_nums"], cfg["middle_blk_num"],
            cfg["dec_blk_nums"]) == (width, enc, middle, dec)


@pytest.mark.parametrize("arm", ["S", "M", "L"])
def test_family_arm_matches_reselection(arm: str) -> None:
    cfg = load_yaml(LOCKED)["family"][arm]
    name, width, enc, middle, dec = EXPECTED_FAMILY[arm]
    assert cfg["name"] == name
    assert (cfg["width"], cfg["enc"], cfg["middle"], cfg["dec"]) == \
        (width, enc, middle, dec)


def test_family_invariants_hold() -> None:
    """params and MACs must increase strictly S < M < L, with span >= 2.5x."""
    from src.models.complexity import measure

    cfg = load_yaml(LOCKED)["family"]
    measured = {}
    for arm in ("S", "M", "L"):
        a = cfg[arm]
        model = NAFNet(width=a["width"], enc_blk_nums=a["enc"],
                       middle_blk_num=a["middle"], dec_blk_nums=a["dec"])
        measured[arm] = measure(model, (1, 3, 256, 256))

    assert measured["S"].params < measured["M"].params < measured["L"].params
    assert measured["S"].macs < measured["M"].macs < measured["L"].macs
    span = measured["L"].macs / measured["S"].macs
    assert span >= 2.5, f"family MAC span {span:.2f}x below the 2.5x minimum"


def test_params_stay_under_the_ceiling() -> None:
    from src.models.complexity import count_params

    cfg = load_yaml(LOCKED)["family"]
    for arm in ("S", "M", "L"):
        a = cfg[arm]
        n = count_params(NAFNet(width=a["width"], enc_blk_nums=a["enc"],
                                middle_blk_num=a["middle"], dec_blk_nums=a["dec"]))
        assert n <= 10_000_000, f"{arm} ({a['name']}) has {n:,} params, over 10M"
