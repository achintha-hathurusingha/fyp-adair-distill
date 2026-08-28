"""Auxiliary degradation-classification head + FiLM conditioning
(kd_feature_multitask experiment — see reports/kd_feature_multitask/plan.md).

Predicts which of {denoise, derain, dehaze} the current input is, from the
student's OWN `middle_blks` features (no teacher involved), and uses that
same prediction to condition (FiLM: scale+shift) the features it was
computed from. Ground truth for the auxiliary loss is
`_provenance["task"]` — already flowing through `build_multitask_loader`
unused (`trainer.py:528`), matching `TASK_IDS = {"denoise": 0, "derain": 1,
"dehaze": 2}` (`src/data/build.py:42`) — no new data-pipeline work.

Deployment-safe by construction: a linear layer (GAP -> logits) and an
elementwise scale/shift, both ops any NPU toolchain this project has
already validated can export. At inference the student conditions on its
own guess; no teacher, no extra runtime dependency, still a plain-conv
graph.

This is deliberately the SIMPLER of two designs considered (the other:
distill the teacher's continuous PCA-16 `e_D` code instead of a discrete
ground-truth label). Ground-truth-label conditioning isolates the actual
variable the multitask literature review flagged — does explicit
degradation-awareness prevent interference, however it's provided — from
the separate question of whether the teacher's specific code is worth
reproducing (kd_feat's existing latent_pre loss already covers that). If
this doesn't help, the teacher-distilled version is the natural escalation,
not the starting point.
"""
from __future__ import annotations

import torch
from torch import nn

N_TASKS = 3  # denoise, derain, dehaze -- matches src/data/build.py's TASK_IDS


class DegradationHead(nn.Module):
    """Args:
        channels: channel count of the feature map to condition (middle_blks').
        n_tasks: number of degradation classes (default 3, matching TASK_IDS).
    """

    def __init__(self, channels: int, n_tasks: int = N_TASKS) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(channels, n_tasks)
        self.film = nn.Linear(n_tasks, 2 * channels)
        self.channels = channels
        self.n_tasks = n_tasks

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Args:
            x: (B, C, H, W) feature map -- C must equal ``self.channels``.

        Returns:
            ``(conditioned_x, logits)`` — ``conditioned_x`` is the same
            shape as ``x``, ``logits`` is ``(B, n_tasks)`` for the auxiliary
            cross-entropy loss (raw logits, not softmax — callers pass
            these straight to ``nn.functional.cross_entropy``).
        """
        if x.ndim != 4:
            raise ValueError(f"expected NCHW input, got shape {tuple(x.shape)}")
        if x.shape[1] != self.channels:
            raise ValueError(
                f"expected {self.channels} channels, got {x.shape[1]}")
        pooled = self.pool(x).flatten(1)          # (B, C)
        logits = self.classifier(pooled)          # (B, n_tasks)
        probs = torch.softmax(logits, dim=-1)     # FiLM conditions on the
                                                   # distribution, not raw
                                                   # logits, so its scale
                                                   # doesn't drift with the
                                                   # classifier's own training
        film = self.film(probs)                   # (B, 2C)
        scale, shift = film.chunk(2, dim=-1)
        scale = scale.view(-1, self.channels, 1, 1)
        shift = shift.view(-1, self.channels, 1, 1)
        return x * (1 + scale) + shift, logits
