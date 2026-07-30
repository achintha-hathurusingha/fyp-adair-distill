"""Determinism helpers. Every entry point calls :func:`seed_everything`."""
from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class RNGState:
    """Snapshot of all RNG states, for exact checkpoint resume."""

    python: object
    numpy: object
    torch: torch.Tensor
    torch_cuda: list | None


def seed_everything(seed: int, *, deterministic: bool = True) -> int:
    """Seed Python, NumPy and PyTorch (CPU+CUDA) and return the resolved seed.

    Args:
        seed: The seed to apply across all libraries.
        deterministic: If True, force cuDNN into deterministic mode and disable
            its autotuner. Required for reproducible numbers; slightly slower.

    Returns:
        The seed that was applied (echoed for logging).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


def capture_rng_state() -> RNGState:
    """Capture current RNG state for all libraries (for --resume)."""
    return RNGState(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch=torch.get_rng_state(),
        torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    )


def restore_rng_state(state: RNGState) -> None:
    """Restore RNG state captured by :func:`capture_rng_state`."""
    random.setstate(state.python)
    np.random.set_state(state.numpy)
    torch.set_rng_state(state.torch)
    if state.torch_cuda is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state.torch_cuda)
