"""Feature adapter bridging the student's `middle_blks` bottleneck to the
teacher's `latent_pre` representation space (kd_feature experiment — see
reports/kd_feature/plan.md).

Neither channel count nor spatial resolution match between the two:

    student (model.middle_blks, M-DEHAZE-KD-FREQ arch): 256ch @ 1/16
    teacher (AdaIR self.latent, dim=48 default):         384ch @ 1/8

The student's own representation is pulled UP toward the teacher's space
(channel-projected and spatially upsampled) rather than compressing the
teacher down — standard FitNets-style adapter practice: don't throw away
resolution or channels the teacher representation actually carries.

This module is training-time only. It is never part of the exported student
graph — same principle already established for kd_freq's spectrum loss and
verified there by an export smoke test (F7) — so it does not reopen the
export-gate problem.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class FeatureAdapter(nn.Module):
    """1x1 conv channel projection + bilinear spatial upsample.

    Args:
        in_channels: student `middle_blks` output channels.
        out_channels: teacher `latent_pre` channels to match.
        scale_factor: spatial upsample factor from student resolution to
            teacher resolution (e.g. 2.0 for 1/16 -> 1/8). Passed explicitly
            rather than inferred from shapes at call time, so a config
            mismatch raises here instead of silently interpolating to the
            wrong size.
    """

    def __init__(self, in_channels: int, out_channels: int, scale_factor: float) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError(
                f"in_channels/out_channels must be positive, got "
                f"{in_channels}/{out_channels}")
        if scale_factor <= 0:
            raise ValueError(f"scale_factor must be positive, got {scale_factor}")
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        self.scale_factor = scale_factor
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected NCHW input, got shape {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} input channels, got {x.shape[1]}")
        x = self.proj(x)
        if self.scale_factor != 1.0:
            x = F.interpolate(x, scale_factor=self.scale_factor, mode="bilinear",
                              align_corners=False)
        return x

    def match_target(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Adapt ``x`` and, if rounding left a 1-2px spatial mismatch against
        ``target`` (bilinear scale_factor doesn't always land exactly), crop
        or pad the trailing edge to match exactly. Raises rather than
        silently interpolating again if the mismatch is larger than that —
        a large mismatch means the configured scale_factor is wrong, not a
        rounding artifact, and should be fixed in config, not papered over.
        """
        out = self.forward(x)
        dh = target.shape[-2] - out.shape[-2]
        dw = target.shape[-1] - out.shape[-1]
        if abs(dh) > 2 or abs(dw) > 2:
            raise ValueError(
                f"adapter output {tuple(out.shape)} vs target "
                f"{tuple(target.shape)} — mismatch too large for a rounding "
                f"fix; check scale_factor")
        if dh > 0 or dw > 0:
            out = F.pad(out, (0, max(dw, 0), 0, max(dh, 0)), mode="replicate")
        if dh < 0 or dw < 0:
            out = out[..., :target.shape[-2], :target.shape[-1]]
        return out
