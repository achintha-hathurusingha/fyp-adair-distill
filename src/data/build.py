"""Training dataloaders and the mixed-task sampler.

**Deliberate deviation from AdaIR.** AdaIR performs no per-batch task balancing
at all: ``sample_ids`` is a flat concatenation of per-task streams with wildly
asymmetric repeat multipliers (derain x120, denoise x3 per sigma, dehaze x1;
``dataset_utils.py:155-168``). A task's share of training is therefore just its
list length. We balance *within* each batch instead, which is recorded wherever
our training mix is compared with theirs.

For Task 1.5b only the denoising stream is used, so balancing is across the
three sigmas.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.datasets import load_rgb_uint8, to_tensor
from src.data.degradations import add_gaussian_noise
from src.data.transforms import paired_transform

_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")


class DenoiseTrainDataset(Dataset):
    """Clean images + on-the-fly Gaussian noise at one of several sigmas.

    Each ``__getitem__`` draws a sigma cyclically from ``sigmas``, so a batch of
    size B contains a balanced mix rather than whatever the shuffle produced.

    Noise here is **iteration-seeded**, not filename-seeded: training wants a
    fresh realisation per epoch, whereas evaluation wants a fixed one. The
    generator is seeded from ``(base_seed, index)`` so a resumed run reproduces
    the same stream.
    """

    def __init__(self, roots: list[str | Path], *, sigmas=(15, 25, 50),
                 patch_size: int = 256, base_seed: int = 0,
                 cache_images: bool = True, length: int | None = None,
                 cache_budget_gb: float = 0.75) -> None:
        self.files: list[Path] = []
        for root in roots:
            root = Path(root)
            if not root.exists():
                raise FileNotFoundError(f"training root not found: {root}")
            self.files.extend(sorted(
                p for p in root.rglob("*")
                if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES))
        if not self.files:
            raise FileNotFoundError(f"no images under {roots}")

        self.sigmas = tuple(sigmas)
        self.patch_size = patch_size
        self.base_seed = base_seed
        self.length = length or len(self.files) * len(self.sigmas)
        # Decoding dominates for small models; cache to keep the GPU fed.
        #
        # The cache MUST be bounded. Workers are persistent and each holds its
        # own copy, so an unbounded dict converges on num_workers x the full
        # decoded training set — 18.4 GB for 6 workers on 5,144 images, against
        # 15.7 GB of RAM. That does not fail fast: the machine pages, the GPU
        # starves, and the run dies hours in. Evict LRU against a byte budget.
        if cache_budget_gb <= 0:
            raise ValueError(
                f"cache_budget_gb must be positive, got {cache_budget_gb}. "
                "Pass cache_images=False to disable caching instead.")
        self._cache: OrderedDict[int, np.ndarray] | None = (
            OrderedDict() if cache_images else None)
        self._cache_budget = int(cache_budget_gb * 2 ** 30)
        self._cache_bytes = 0

    def __len__(self) -> int:
        return self.length

    def _image(self, idx: int) -> np.ndarray:
        file_idx = idx % len(self.files)
        if self._cache is not None and file_idx in self._cache:
            self._cache.move_to_end(file_idx)      # mark as recently used
            return self._cache[file_idx]
        img = load_rgb_uint8(self.files[file_idx], base=1)  # no crop for training
        if self._cache is not None:
            # A single image larger than the whole budget would otherwise be
            # inserted and immediately evicted every time; skip caching it.
            if img.nbytes <= self._cache_budget:
                self._cache[file_idx] = img
                self._cache_bytes += img.nbytes
                while self._cache_bytes > self._cache_budget:
                    _, evicted = self._cache.popitem(last=False)   # LRU
                    self._cache_bytes -= evicted.nbytes
        return img

    def __getitem__(self, idx: int):
        clean_full = self._image(idx)
        # Balanced across sigmas by construction rather than by chance.
        sigma = self.sigmas[idx % len(self.sigmas)]
        rng = np.random.default_rng((self.base_seed, idx))

        h, w = clean_full.shape[:2]
        if h < self.patch_size or w < self.patch_size:
            pad_h = max(0, self.patch_size - h)
            pad_w = max(0, self.patch_size - w)
            clean_full = np.pad(clean_full, ((0, pad_h), (0, pad_w), (0, 0)),
                                mode="reflect")

        # Crop first, then synthesise noise on the crop: noise is i.i.d. so this
        # is equivalent to cropping a noised image, and is much cheaper.
        clean, _ = paired_transform(clean_full, clean_full,
                                    patch_size=self.patch_size, rng=rng)
        noise_state = np.random.RandomState(
            int(rng.integers(0, 2 ** 31 - 1)))  # noqa: NPY002 - legacy stream
        degraded = add_gaussian_noise(clean, sigma, rng=noise_state)
        return to_tensor(degraded), to_tensor(clean), sigma


def build_train_loader(roots: list[str | Path], *, batch_size: int = 32,
                       patch_size: int = 256, sigmas=(15, 25, 50),
                       num_workers: int = 8, seed: int = 0,
                       length: int | None = None,
                       cache_budget_gb: float = 0.75) -> torch.utils.data.DataLoader:
    """Build the training loader, tuned to avoid being dataloader-bound.

    ``cache_budget_gb`` is PER WORKER — total resident cache is roughly
    ``num_workers * cache_budget_gb``. Size it against real RAM, not the
    dataset.
    """
    dataset = DenoiseTrainDataset(roots, sigmas=sigmas, patch_size=patch_size,
                                  base_seed=seed, length=length,
                                  cache_budget_gb=cache_budget_gb)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )
