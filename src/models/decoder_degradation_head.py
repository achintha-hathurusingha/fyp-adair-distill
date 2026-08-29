"""Decoder-stage FiLM conditioning (kd_feature_multitask v2 -- see
reports/kd_feature_multitask/plan_v2_decoder_film.md).

Fixes the B0V2-KD-FEAT-COND regression
(reports/kd_feature_multitask/cond_regression.md): that design's FiLM
modulated `middle_blks`'s own output -- the exact tensor the feature-KD loss
also reads (via the adapter, against the teacher's `latent_pre`). Every task
got worse, and the gap widened over training, concentrated on dehaze
(feature-KD's strongest task) -- consistent with two objectives fighting
over one representation.

This version classifies off `middle_blks` (unchanged -- that tensor carries
the clearest degradation signal, TEST19: 99.0% leave-scene-out accuracy on
the teacher's equivalent representation) but never writes to it. Conditioning
moves to the decoder instead: one small FiLM head per decoder stage, matching
PromptIR's own ablation-validated placement (decoder-only, multi-level
injection beats single-point latent/bottleneck injection: 36.76dB vs
37.04dB on their own Rain100L ablation).
"""
from __future__ import annotations

import torch
from torch import nn

N_TASKS = 3  # denoise, derain, dehaze -- matches src/data/build.py's TASK_IDS


class DecoderDegradationHead(nn.Module):
    """Args:
        middle_channels: channel count of middle_blks' output (read-only).
        decoder_channels: channel count of each decoder stage, in the same
            order NAFNet.decoders iterates them (e.g. [128, 64, 32, 16] for
            W16 SIDD's dec_blk_nums=[2,2,2,2]).
        n_tasks: number of degradation classes (default 3, matching TASK_IDS).
    """

    def __init__(self, middle_channels: int, decoder_channels: list[int],
                 n_tasks: int = N_TASKS) -> None:
        super().__init__()
        if middle_channels <= 0:
            raise ValueError(f"middle_channels must be positive, got {middle_channels}")
        if not decoder_channels or any(c <= 0 for c in decoder_channels):
            raise ValueError(f"decoder_channels must be a non-empty list of positive "
                             f"ints, got {decoder_channels}")
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(middle_channels, n_tasks)
        self.films = nn.ModuleList(
            nn.Linear(n_tasks, 2 * c) for c in decoder_channels)
        self.middle_channels = middle_channels
        self.decoder_channels = decoder_channels
        self.n_tasks = n_tasks

    def classify(self, middle_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Read-only: pools and classifies `middle_feat`, returns it UNCHANGED
        to the caller alongside the prediction. Never modulates `middle_feat`
        itself -- that is exactly the behaviour that caused the v1 regression.

        Returns:
            ``(logits, probs)`` -- ``logits`` is ``(B, n_tasks)`` raw logits
            for the auxiliary cross-entropy loss, ``probs`` is the softmax
            distribution FiLM conditions on (stable scale, doesn't drift with
            the classifier's own training, same rationale as v1).
        """
        if middle_feat.ndim != 4:
            raise ValueError(f"expected NCHW input, got shape {tuple(middle_feat.shape)}")
        if middle_feat.shape[1] != self.middle_channels:
            raise ValueError(
                f"expected {self.middle_channels} middle channels, got {middle_feat.shape[1]}")
        pooled = self.pool(middle_feat).flatten(1)
        logits = self.classifier(pooled)
        probs = torch.softmax(logits, dim=-1)
        return logits, probs

    def modulate(self, x: torch.Tensor, probs: torch.Tensor, stage: int) -> torch.Tensor:
        """FiLM-condition decoder stage ``stage``'s output on ``probs``.

        Args:
            x: (B, C, H, W) decoder-stage output, C must equal
                ``decoder_channels[stage]``.
            probs: (B, n_tasks) softmax distribution from ``classify``.
            stage: index into ``decoder_channels`` / ``self.films``.
        """
        if not (0 <= stage < len(self.films)):
            raise ValueError(f"stage {stage} out of range for "
                             f"{len(self.films)} decoder stages")
        c = self.decoder_channels[stage]
        if x.ndim != 4 or x.shape[1] != c:
            raise ValueError(
                f"stage {stage}: expected {c} channels, got shape {tuple(x.shape)}")
        film = self.films[stage](probs)
        scale, shift = film.chunk(2, dim=-1)
        scale = scale.view(-1, c, 1, 1)
        shift = shift.view(-1, c, 1, 1)
        return x * (1 + scale) + shift
