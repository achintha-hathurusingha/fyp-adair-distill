"""Frozen AdaIR teacher wrapper — inference and caching ONLY (Phase 01).

Implemented in Task 3. Must: load the released 3-degradation checkpoint, force
eval mode, disable gradients, and assert an exact weight load (no partial-load
warnings tolerated).

Explicitly out of scope for Phase 01: feature-extraction hooks into AdaIR
internals, adapters/projectors, any distillation plumbing.
"""
from __future__ import annotations

_TASK = "Task 3 — teacher integration"


def load_teacher(*_args, **_kwargs):
    """Placeholder — implemented in Task 3."""
    raise NotImplementedError(f"teacher_wrapper.load_teacher is scaffolded for {_TASK}.")
