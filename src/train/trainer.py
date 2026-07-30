"""Training loop — implemented in Task 5 (B0 baseline).

Requirements when implemented: AdamW, cosine schedule, 256x256 patches,
gradient clipping, EMA, periodic validation, best-checkpoint tracking, and
fully resumable state (optimizer, scheduler, epoch/iter, RNG state).
"""
from __future__ import annotations

_TASK = "Task 5 — B0 baseline"


class Trainer:
    """Placeholder — implemented in Task 5."""

    def __init__(self, *_args, **_kwargs) -> None:
        raise NotImplementedError(f"Trainer is scaffolded for {_TASK}.")
