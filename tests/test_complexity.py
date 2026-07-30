"""Tests pinning the MAC-counting convention (rule 8: test anything numeric)."""
from __future__ import annotations

import torch
from torch import nn

from src.models.complexity import count_macs, count_params, measure
from src.models.nafnet import NAFNet


def test_conv_macs_match_hand_calculation() -> None:
    """Known answer: Cout*Hout*Wout*(Cin/groups)*Kh*Kw."""
    conv = nn.Conv2d(3, 8, 3, padding=1, bias=False)
    expected = 8 * 32 * 32 * (3 * 3 * 3)
    assert count_macs(conv, (1, 3, 32, 32)) == expected


def test_depthwise_conv_macs_account_for_groups() -> None:
    """Groups must divide the per-output cost."""
    dw = nn.Conv2d(16, 16, 3, padding=1, groups=16, bias=False)
    expected = 16 * 16 * 16 * (1 * 3 * 3)  # Cin/groups == 1
    assert count_macs(dw, (1, 16, 16, 16)) == expected


def test_macs_scale_quadratically_with_resolution() -> None:
    """Doubling H and W should ~4x the MACs for a conv net."""
    conv = nn.Conv2d(3, 8, 3, padding=1)
    small = count_macs(conv, (1, 3, 32, 32))
    large = count_macs(conv, (1, 3, 64, 64))
    assert large == 4 * small


def test_deep_blocks_cost_less_at_low_resolution() -> None:
    """The core argument for choosing by MACs, not params.

    A block at H/8 costs ~1/64 the MACs of the same block at full resolution,
    so parameter count alone cannot rank architectures by cost.
    """
    block_hi = nn.Conv2d(32, 32, 3, padding=1, bias=False)
    macs_full = count_macs(block_hi, (1, 32, 64, 64))
    macs_eighth = count_macs(block_hi, (1, 32, 8, 8))
    assert macs_full == 64 * macs_eighth


def test_measure_returns_params_and_macs() -> None:
    model = NAFNet(width=8, enc_blk_nums=[1, 1], middle_blk_num=1,
                   dec_blk_nums=[1, 1])
    c = measure(model, (1, 3, 64, 64))
    assert c.params == count_params(model)
    assert c.macs > 0
    assert c.resolution == (64, 64)
    assert c.gmacs == c.macs / 1e9


def test_params_are_resolution_independent() -> None:
    """Sanity: parameter count must not depend on input size."""
    model = NAFNet(width=8, enc_blk_nums=[1, 1], middle_blk_num=1,
                   dec_blk_nums=[1, 1])
    a = measure(model, (1, 3, 64, 64))
    b = measure(model, (1, 3, 128, 128))
    assert a.params == b.params
    assert b.macs > a.macs


def test_no_grad_during_counting() -> None:
    """Counting must not leave the model in training mode or build graphs."""
    model = NAFNet(width=8, enc_blk_nums=[1, 1], middle_blk_num=1,
                   dec_blk_nums=[1, 1]).train()
    count_macs(model, (1, 3, 64, 64))
    assert not model.training
    assert all(p.grad is None for p in model.parameters())


def test_torch_flop_counter_is_two_flops_per_mac() -> None:
    """Pin the upstream convention this module depends on.

    If a future torch release changes to counting 1 FLOP per MAC, this test
    fails loudly instead of silently halving every reported number.
    """
    from torch.utils.flop_counter import FlopCounterMode

    conv = nn.Conv2d(3, 8, 3, padding=1, bias=False)
    counter = FlopCounterMode(display=False)
    with counter, torch.no_grad():
        conv(torch.randn(1, 3, 32, 32))
    hand_macs = 8 * 32 * 32 * (3 * 3 * 3)
    assert counter.get_total_flops() == 2 * hand_macs
