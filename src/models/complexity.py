"""Parameter and MAC accounting for architecture selection.

MAC CONVENTION (stated explicitly — rule 4):
  * A "MAC" is one multiply-accumulate. A conv contributing
    ``Cout*Hout*Wout*(Cin/groups)*Kh*Kw`` multiply-accumulates counts as that
    many MACs.
  * ``torch.utils.flop_counter.FlopCounterMode`` reports **2 FLOPs per MAC**
    (verified empirically against a known-answer conv in tests/test_complexity.py).
    We therefore report ``MACs = flops / 2``.
  * MACs are measured at a stated input resolution (default 256x256, matching
    the fixed-resolution protocol) with batch size 1.
  * Only ops the counter models are included (conv/matmul/attention). Cheap
    elementwise work (LayerNorm arithmetic, SimpleGate multiplies, sigmoid) is
    NOT counted — it is bandwidth-bound, not MAC-bound. So MACs understate the
    true cost of normalisation-heavy graphs; on-device latency is the arbiter.
  * Papers in this field often print "FLOPs" when they mean MACs. When comparing
    to a published number, confirm which convention and which resolution it used.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class Complexity:
    """Measured complexity of one model at one input resolution."""

    params: int
    macs: int
    resolution: tuple[int, int]

    @property
    def gmacs(self) -> float:
        """MACs in billions."""
        return self.macs / 1e9

    @property
    def mparams(self) -> float:
        """Parameters in millions."""
        return self.params / 1e6


def count_params(model: nn.Module, *, trainable_only: bool = False) -> int:
    """Count model parameters."""
    ps = model.parameters()
    if trainable_only:
        return sum(p.numel() for p in ps if p.requires_grad)
    return sum(p.numel() for p in ps)


def count_macs(model: nn.Module, input_shape: tuple[int, int, int, int]) -> int:
    """Count MACs for one forward pass at ``input_shape`` (N, C, H, W).

    Uses :class:`torch.utils.flop_counter.FlopCounterMode` and halves the
    result, since that counter reports 2 FLOPs per multiply-accumulate.

    Raises:
        ValueError: If the counter reports an odd total, which would mean the
            flops-per-MAC assumption no longer holds.
    """
    from torch.utils.flop_counter import FlopCounterMode

    model = model.eval()
    x = torch.randn(*input_shape)
    counter = FlopCounterMode(display=False)
    with counter, torch.no_grad():
        model(x)
    flops = counter.get_total_flops()
    if flops % 2 != 0:
        raise ValueError(
            f"FlopCounterMode returned an odd FLOP total ({flops}); the "
            "2-FLOPs-per-MAC assumption in this module may be invalid."
        )
    return flops // 2


def measure(model: nn.Module, input_shape: tuple[int, int, int, int]) -> Complexity:
    """Measure params and MACs together at ``input_shape``."""
    return Complexity(
        params=count_params(model),
        macs=count_macs(model, input_shape),
        resolution=(input_shape[2], input_shape[3]),
    )
