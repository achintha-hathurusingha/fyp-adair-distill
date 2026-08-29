"""Reads a teacher cache built by scripts/build_teacher_cache.py (see
reports/kd_feature_multitask/plan_cached_teacher.md) and yields the same
5-tuple shape a live training step currently assembles by hand: (degraded,
clean, response, latent_pre, provenance).

NO geometric re-augmentation is applied. The plan originally proposed D4
(flip/rotation) re-augmentation here, reasoning that a deterministic
teacher's output must transform identically to its input -- sound for a
generic CNN, wrong for THIS teacher. Verified directly in
smoke_cached_teacher_dataset.py (not assumed): of the 8 dihedral
transforms, only the identity matched a live re-run within quantization
tolerance. Even a plain horizontal flip broke completely (latent diff ~15
against a signal scale of ~2.5) -- AdaIR's FFT-based FreModule is not
flip/rotation-equivariant in practice, consistent with this project's own
prior findings of non-obvious asymmetric behaviour in this exact
frequency-domain machinery (TEST01/06/18, TEST20). `_d4` is kept as a
module-level function purely so the smoke test can keep demonstrating (and
guarding against regressing) this finding -- it is never called from
`__getitem__`.

Task balancing reuses `BalancedTaskBatchSampler` UNCHANGED -- the cache is
built with each task occupying a contiguous row range specifically so this
already-proven sampler needs no modification.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import sys
sys.path.insert(0, ".")
from src.data.sampler import BalancedTaskBatchSampler

TASK_NAMES = {0: "denoise", 1: "derain", 2: "dehaze"}


def _d4(tensors: list[torch.Tensor], k: int) -> list[torch.Tensor]:
    """Dihedral transform k (0-7): k % 4 = 90-degree rotation count, k // 4
    = whether to also mirror. NOT used by CachedTeacherDataset -- kept only
    for smoke_cached_teacher_dataset.py, which uses it to demonstrate (and
    would catch a regression of) the finding that this teacher is not D4-
    equivariant. See the module docstring."""
    rot, flip = k % 4, k // 4
    out = []
    for t in tensors:
        x = torch.rot90(t, rot, dims=(-2, -1))
        if flip:
            x = torch.flip(x, dims=(-1,))
        out.append(x)
    return out


class CachedTeacherDataset(Dataset):
    def __init__(self, cache_dir: str | Path) -> None:
        cache_dir = Path(cache_dir)
        index = json.loads((cache_dir / "index.json").read_text())
        self.n = index["n"]
        self.patch = index["patch"]
        self.latent_shape = tuple(index["latent_shape"])
        self._task_ranges = index["task_ranges"]  # {"denoise": [a, b], ...}
        self.sigma = index["sigma"]

        self.degraded = np.memmap(cache_dir / "degraded.dat", dtype=np.uint8,
                                  mode="r", shape=(self.n, 3, self.patch, self.patch))
        self.clean = np.memmap(cache_dir / "clean.dat", dtype=np.uint8,
                               mode="r", shape=(self.n, 3, self.patch, self.patch))
        self.response = np.memmap(cache_dir / "response.dat", dtype=np.uint8,
                                  mode="r", shape=(self.n, 3, self.patch, self.patch))
        self.latent = np.memmap(cache_dir / "latent_pre.dat", dtype=np.float16,
                                mode="r", shape=(self.n, *self.latent_shape))

    def task_ranges(self) -> dict[str, range]:
        """For BalancedTaskBatchSampler -- unchanged interface."""
        return {t: range(a, b) for t, (a, b) in self._task_ranges.items()}

    def _task_id_for(self, idx: int) -> int:
        for name, (a, b) in self._task_ranges.items():
            if a <= idx < b:
                return {"denoise": 0, "derain": 1, "dehaze": 2}[name]
        raise IndexError(f"index {idx} not covered by any task range")

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        degraded = torch.from_numpy(np.array(self.degraded[idx])).float() / 255.0
        clean = torch.from_numpy(np.array(self.clean[idx])).float() / 255.0
        response = torch.from_numpy(np.array(self.response[idx])).float() / 255.0
        latent = torch.from_numpy(np.array(self.latent[idx]).astype(np.float32))
        prov = {"task": self._task_id_for(idx), "sigma": self.sigma[idx]}
        return degraded, clean, response, latent, prov


def build_cached_teacher_loader(cache_dir: str | Path, *, batch_size: int,
                                num_batches: int, num_workers: int = 4,
                                seed: int = 0) -> DataLoader:
    dataset = CachedTeacherDataset(cache_dir)
    sampler = BalancedTaskBatchSampler(dataset.task_ranges(), batch_size,
                                       num_batches=num_batches, base_seed=seed)
    return DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers,
                      pin_memory=True, persistent_workers=num_workers > 0)
