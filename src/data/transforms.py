"""Paired geometric augmentation.

Every transform here applies the **identical** geometric operation to the
degraded image and its ground truth. A desynchronised pair trains the model to
predict a shifted or flipped target, which does not crash and does not look
obviously wrong in the loss curve — it just quietly costs several dB. Hence the
pairing is unit-tested rather than assumed.

Augmentation follows the standard restoration recipe used by AdaIR's training
path: random crop, horizontal flip, and 90-degree rotations.
"""
from __future__ import annotations

import numpy as np


def paired_random_crop(degraded: np.ndarray, clean: np.ndarray, size: int,
                       rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Crop both images at the same random location.

    Args:
        degraded: HWC array.
        clean: HWC array, same shape as ``degraded``.
        size: square patch size.
        rng: seeded generator.

    Raises:
        ValueError: on shape mismatch, or if the patch does not fit.
    """
    if degraded.shape != clean.shape:
        raise ValueError(
            f"paired inputs must match: {degraded.shape} vs {clean.shape}")
    h, w = degraded.shape[:2]
    if h < size or w < size:
        raise ValueError(
            f"cannot crop {size}x{size} from a {h}x{w} image")

    top = int(rng.integers(0, h - size + 1))
    left = int(rng.integers(0, w - size + 1))
    return (degraded[top:top + size, left:left + size],
            clean[top:top + size, left:left + size])


def paired_augment(degraded: np.ndarray, clean: np.ndarray,
                   rng: np.random.Generator, *,
                   flip: bool = True, rotate: bool = True
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Apply the same random flip and 90-degree rotation to both images."""
    if degraded.shape != clean.shape:
        raise ValueError(
            f"paired inputs must match: {degraded.shape} vs {clean.shape}")

    if flip and rng.random() < 0.5:
        degraded, clean = np.fliplr(degraded), np.fliplr(clean)
    if rotate:
        k = int(rng.integers(0, 4))
        if k:
            degraded = np.rot90(degraded, k, axes=(0, 1))
            clean = np.rot90(clean, k, axes=(0, 1))
    return np.ascontiguousarray(degraded), np.ascontiguousarray(clean)


def paired_transform(degraded: np.ndarray, clean: np.ndarray, *,
                     patch_size: int, rng: np.random.Generator,
                     flip: bool = True, rotate: bool = True
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Crop then augment, keeping the pair synchronised throughout."""
    degraded, clean = paired_random_crop(degraded, clean, patch_size, rng)
    return paired_augment(degraded, clean, rng, flip=flip, rotate=rotate)
