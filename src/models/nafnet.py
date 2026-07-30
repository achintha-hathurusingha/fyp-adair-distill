"""Clean-room NAFNet (width-32) student.

Self-contained reimplementation of NAFNet (Chen et al., ECCV 2022) with no
basicsr dependency, so the graph exports cleanly. Block/width defaults follow
the official NAFNet-width32 configuration.

Export notes (relevant to Gate G1):
  * ``LayerNorm2d`` expands to ReduceMean/Sub/Pow/Sqrt/Div — watch on NPU INT8.
  * ``SimpleGate`` is a channel chunk + elementwise multiply (no nonlinearity).
  * Up-sampling uses PixelShuffle -> ONNX ``DepthToSpace``.
Optionally inserts a :class:`ChannelGate` after the stem to smoke-test the
export-critical gate op set (see gate.py).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.models.gate import ChannelGate


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for NCHW tensors."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=1, keepdim=True)
        var = (x - mu).pow(2).mean(dim=1, keepdim=True)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class SimpleGate(nn.Module):
    """Split channels in half and multiply the halves (NAFNet's gating)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """NAFNet residual block: gated depthwise conv + simplified channel attn."""

    def __init__(self, c: int, dw_expand: int = 2, ffn_expand: int = 2) -> None:
        super().__init__()
        dw = c * dw_expand
        self.norm1 = LayerNorm2d(c)
        self.conv1 = nn.Conv2d(c, dw, 1)
        self.conv2 = nn.Conv2d(dw, dw, 3, padding=1, groups=dw)
        self.sg = SimpleGate()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw // 2, dw // 2, 1),
        )
        self.conv3 = nn.Conv2d(dw // 2, c, 1)

        self.norm2 = LayerNorm2d(c)
        ffn = c * ffn_expand
        self.conv4 = nn.Conv2d(c, ffn, 1)
        self.conv5 = nn.Conv2d(ffn // 2, c, 1)

        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = self.norm1(inp)
        x = self.conv2(self.conv1(x))
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        return y + x * self.gamma


class NAFNet(nn.Module):
    """NAFNet U-shaped restoration network.

    Args:
        img_channels: Input/output channels (3 for RGB).
        width: Base feature width (32 for the width-32 student).
        enc_blk_nums: Number of NAFBlocks per encoder stage.
        middle_blk_num: Number of NAFBlocks in the bottleneck.
        dec_blk_nums: Number of NAFBlocks per decoder stage.
        use_gate: If True, insert a :class:`ChannelGate` after the stem conv
            (Gate G1 smoke-test hook; a placeholder for Phase-02 gating).
        gate_reduction: Reduction factor for the inserted gate.
    """

    def __init__(self, img_channels: int = 3, width: int = 32,
                 enc_blk_nums: list[int] | None = None, middle_blk_num: int = 12,
                 dec_blk_nums: list[int] | None = None, *, use_gate: bool = False,
                 gate_reduction: int = 4) -> None:
        super().__init__()
        enc_blk_nums = enc_blk_nums or [2, 2, 4, 8]
        dec_blk_nums = dec_blk_nums or [2, 2, 2, 2]
        if len(enc_blk_nums) != len(dec_blk_nums):
            raise ValueError("enc and dec must have equal stage counts, "
                             f"got {len(enc_blk_nums)} vs {len(dec_blk_nums)}")

        self.intro = nn.Conv2d(img_channels, width, 3, padding=1)
        self.gate = ChannelGate(width, gate_reduction) if use_gate else None
        self.ending = nn.Conv2d(width, img_channels, 3, padding=1)

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()

        chan = width
        for n in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(n)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, stride=2))
            chan *= 2

        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])

        for n in dec_blk_nums:
            self.ups.append(nn.Sequential(
                nn.Conv2d(chan, chan * 2, 1, bias=False),
                nn.PixelShuffle(2),
            ))
            chan //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(n)]))

        self.padder_size = 2 ** len(enc_blk_nums)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Restore ``inp`` of shape ``(N, C, H, W)``; global residual applied."""
        _, _, h, w = inp.shape
        inp = self._pad(inp)

        x = self.intro(inp)
        if self.gate is not None:
            x = self.gate(x)

        skips = []
        for enc, down in zip(self.encoders, self.downs):
            x = enc(x)
            skips.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for dec, up, skip in zip(self.decoders, self.ups, reversed(skips)):
            x = up(x)
            x = x + skip
            x = dec(x)

        x = self.ending(x)
        x = x + inp
        return x[:, :, :h, :w]

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        """Zero-pad H,W up to a multiple of ``padder_size`` (no-op at 256)."""
        _, _, h, w = x.shape
        ph = (self.padder_size - h % self.padder_size) % self.padder_size
        pw = (self.padder_size - w % self.padder_size) % self.padder_size
        return F.pad(x, (0, pw, 0, ph))


def build_nafnet(cfg: dict, *, use_gate: bool = False) -> NAFNet:
    """Construct a :class:`NAFNet` from a model-config dict (see configs/model)."""
    gate_cfg = cfg.get("gate", {})
    return NAFNet(
        img_channels=cfg.get("img_channels", 3),
        width=cfg.get("width", 32),
        enc_blk_nums=cfg.get("enc_blk_nums"),
        middle_blk_num=cfg.get("middle_blk_num", 12),
        dec_blk_nums=cfg.get("dec_blk_nums"),
        use_gate=use_gate or gate_cfg.get("enabled", False),
        gate_reduction=gate_cfg.get("reduction", 4),
    )
