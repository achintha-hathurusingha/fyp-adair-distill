"""Shape/behaviour tests for the student and the export-critical gate."""
from __future__ import annotations

import torch

from src.models.gate import ChannelGate
from src.models.nafnet import NAFNet, build_nafnet
from src.utils.config import load_yaml


def test_gate_preserves_shape_and_scales() -> None:
    gate = ChannelGate(32, reduction=4).eval()
    x = torch.randn(2, 32, 16, 16)
    y = gate(x)
    assert y.shape == x.shape
    # Sigmoid gating is strictly positive and < 1, so |y| < |x| elementwise.
    assert torch.all(y.abs() <= x.abs() + 1e-6)


def test_gate_rejects_bad_channels() -> None:
    try:
        ChannelGate(0)
    except ValueError:
        return
    raise AssertionError("ChannelGate must reject non-positive channels")


def test_nafnet_roundtrip_shape() -> None:
    model = NAFNet(width=8, enc_blk_nums=[1, 1], middle_blk_num=1,
                   dec_blk_nums=[1, 1]).eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        y = model(x)
    assert y.shape == x.shape


def test_nafnet_handles_non_multiple_resolution() -> None:
    """Padding must let non-multiple-of-4 inputs through and restore size."""
    model = NAFNet(width=8, enc_blk_nums=[1, 1], middle_blk_num=1,
                   dec_blk_nums=[1, 1]).eval()
    x = torch.randn(1, 3, 50, 54)
    with torch.no_grad():
        y = model(x)
    assert y.shape == x.shape


def test_build_nafnet_from_config_width32() -> None:
    cfg = load_yaml("configs/model/nafnet_w32.yaml")
    model = build_nafnet(cfg)
    assert model.intro.out_channels == 32
    assert model.gate is None, "gate must be off by default in the config"

    gated = build_nafnet(cfg, use_gate=True)
    assert isinstance(gated.gate, ChannelGate)


def test_enc_dec_stage_mismatch_raises() -> None:
    try:
        NAFNet(enc_blk_nums=[1, 1, 1], dec_blk_nums=[1, 1])
    except ValueError:
        return
    raise AssertionError("NAFNet must reject mismatched enc/dec stage counts")
