"""Export-critical channel gate.

Deliberately restricted to a minimal, NPU-friendly op set so it survives INT8
export to QNN / TFLite / TensorRT:

    Global Average Pool -> 1x1 conv -> ReLU -> 1x1 conv -> sigmoid -> multiply

Nothing else. Do not add LayerNorm, GELU, softmax, or dynamic-shape ops here —
this module is the one whose exportability gates the whole architecture choice
(Gate G1).
"""
from __future__ import annotations

import torch
from torch import nn


class ChannelGate(nn.Module):
    """Squeeze-and-excitation-style channel gate, minimal op set.

    Args:
        channels: Number of input/output channels (gate is channel-preserving).
        reduction: Channel-reduction factor for the bottleneck. The reduced
            width is ``max(1, channels // reduction)``.
    """

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)          # -> GlobalAveragePool
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel-wise gating. Input/output shape ``(N, C, H, W)``."""
        w = self.pool(x)
        w = self.act(self.fc1(w))
        w = self.gate(self.fc2(w))
        return x * w
