"""Test datasets for the 3-degradation protocol, replicating AdaIR's loading.

Loading conventions (traced in ``reports/eval_conventions.md``):

* ``PIL.Image.open(...).convert('RGB')`` — RGB order, not BGR
  (``dataset_utils.py:252, 348, 351``)
* ``crop_img(..., base=16)`` — centre-crop to a multiple of 16, applied to the
  degraded image *and* the ground truth (``image_utils.py:59-64``)
* ``ToTensor()`` — HWC->CHW, divide by 255, giving ``[0, 1]`` float

Denoise noise is synthesised **after** the crop, matching AdaIR.

Unlike AdaIR these datasets **sort** their file listings. AdaIR uses
``os.listdir`` unsorted (``dataset_utils.py:238``), which makes its denoise noise
realisation filesystem-dependent; combined with filename-derived seeding
(default) our inputs are reproducible across machines. The G3 report quantifies
the resulting delta.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch

from src.data.degradations import add_gaussian_noise
from src.eval.metrics import crop_to_multiple

_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")
#: AdaIR crops every test image to a multiple of this before inference.
CROP_BASE = 16


def _list_images(directory: Path) -> list[Path]:
    """Sorted image listing. Sorting is a deliberate deviation — see module doc."""
    if not directory.exists():
        raise FileNotFoundError(f"dataset directory not found: {directory}")
    files = sorted(p for p in directory.iterdir()
                   if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES)
    if not files:
        raise FileNotFoundError(f"no images found in {directory}")
    return files


def load_rgb_uint8(path: Path, base: int = CROP_BASE) -> np.ndarray:
    """Load an image as cropped uint8 HWC RGB, exactly as AdaIR does."""
    from PIL import Image

    with Image.open(path) as im:
        arr = np.array(im.convert("RGB"))
    return crop_to_multiple(arr, base=base)


def to_tensor(img_uint8: np.ndarray) -> torch.Tensor:
    """uint8 HWC -> float32 CHW in [0, 1] (torchvision ``ToTensor`` semantics)."""
    return torch.from_numpy(
        np.ascontiguousarray(img_uint8.transpose(2, 0, 1))).float() / 255.0


class DenoiseTestDataset:
    """BSD68 with synthesised Gaussian noise at a fixed sigma.

    Yields ``(name, degraded, clean)`` with CHW float tensors in ``[0, 1]``.
    """

    def __init__(self, root: str | Path, sigma: int, *,
                 seed_mode: str = "filename", crop_base: int = CROP_BASE) -> None:
        if seed_mode not in ("filename", "global"):
            raise ValueError(
                f"seed_mode must be 'filename' or 'global', got {seed_mode!r}")
        self.files = _list_images(Path(root))
        self.sigma = sigma
        self.seed_mode = seed_mode
        self.crop_base = crop_base
        self._rng = None
        if seed_mode == "global":
            from src.data.degradations import legacy_rng
            self._rng = legacy_rng(0)

    def __len__(self) -> int:
        return len(self.files)

    def __iter__(self) -> Iterator[tuple[str, torch.Tensor, torch.Tensor]]:
        for path in self.files:
            clean = load_rgb_uint8(path, self.crop_base)
            if self.seed_mode == "filename":
                noisy = add_gaussian_noise(clean, self.sigma, filename=path.name)
            else:
                noisy = add_gaussian_noise(clean, self.sigma, rng=self._rng)
            yield path.stem, to_tensor(noisy), to_tensor(clean)


class PairedTestDataset:
    """Derain / dehaze: paired ``input/`` and ``target/`` directories.

    Ground truth is resolved the way AdaIR does (``dataset_utils.py:325-338``):
    by identical basename for deraining, and by ``name.split('_')[0] + '.png'``
    for dehazing, where several hazy variants may share one target.
    """

    def __init__(self, root: str | Path, *, task: str = "derain",
                 crop_base: int = CROP_BASE) -> None:
        if task not in ("derain", "dehaze"):
            raise ValueError(f"task must be 'derain' or 'dehaze', got {task!r}")
        root = Path(root)
        self.task = task
        self.crop_base = crop_base
        self.inputs = _list_images(root / "input")
        self.target_dir = root / "target"
        if not self.target_dir.exists():
            raise FileNotFoundError(f"missing target directory: {self.target_dir}")
        # Resolve every pair up front so a broken pairing fails immediately
        # rather than midway through an evaluation run.
        self.pairs = [(p, self._target_for(p)) for p in self.inputs]

    def _target_for(self, degraded: Path) -> Path:
        if self.task == "dehaze":
            candidate = self.target_dir / f"{degraded.stem.split('_')[0]}.png"
        else:
            candidate = self.target_dir / degraded.name
        if not candidate.exists():
            # Fall back across suffixes before giving up.
            for suffix in _IMAGE_SUFFIXES:
                alt = candidate.with_suffix(suffix)
                if alt.exists():
                    return alt
            raise FileNotFoundError(
                f"no ground truth for {degraded.name} (looked for {candidate.name})")
        return candidate

    def __len__(self) -> int:
        return len(self.pairs)

    def __iter__(self) -> Iterator[tuple[str, torch.Tensor, torch.Tensor]]:
        for deg_path, tgt_path in self.pairs:
            degraded = load_rgb_uint8(deg_path, self.crop_base)
            clean = load_rgb_uint8(tgt_path, self.crop_base)
            if degraded.shape != clean.shape:
                raise ValueError(
                    f"{deg_path.name}: degraded {degraded.shape} != "
                    f"clean {clean.shape} after cropping")
            yield deg_path.stem, to_tensor(degraded), to_tensor(clean)


def build_dataset(task: str, root: str | Path, **kwargs):
    """Construct the test dataset for ``task``."""
    if task == "denoise":
        return DenoiseTestDataset(root, **kwargs)
    if task in ("derain", "dehaze"):
        kwargs.pop("seed_mode", None)
        kwargs.pop("sigma", None)
        return PairedTestDataset(root, task=task, **kwargs)
    raise ValueError(f"unknown task {task!r}; expected denoise/derain/dehaze")
