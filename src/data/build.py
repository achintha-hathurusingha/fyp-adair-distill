"""Training dataloaders and the mixed-task sampler.

**Deliberate deviation from AdaIR.** AdaIR performs no per-batch task balancing
at all: ``sample_ids`` is a flat concatenation of per-task streams with wildly
asymmetric repeat multipliers (derain x120, denoise x3 per sigma, dehaze x1;
``dataset_utils.py:155-168``). A task's share of training is therefore just its
list length. We balance *within* each batch instead, which is recorded wherever
our training mix is compared with theirs.

Two training datasets live here, and the distinction matters for how results are
labelled:

``DenoiseTrainDataset``
    Denoising only, balanced across the three sigmas. This is what Task 1.5b's
    normalization ablation and **B0-denoise** were trained on — see finding F11,
    where the single-task scope went unnoticed because ``mixed_task: true`` sat
    in the config while no code read it.

``MultiTaskTrainDataset``
    The real 3-degradation set, one index space over denoise + derain + dehaze.
    Feeds B0-v2.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.datasets import load_rgb_uint8, resolve_pair_target, to_tensor
from src.data.degradations import add_gaussian_noise
from src.data.sampler import BalancedTaskBatchSampler
from src.data.transforms import paired_transform

_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")

#: Task codes carried alongside every sample. Fixed and ordered: they are
#: written into checkpoints and batch-composition assertions, so appending is
#: safe but reordering is not.
TASK_IDS = {"denoise": 0, "derain": 1, "dehaze": 2}


def _list_images(root: Path) -> list[Path]:
    """Sorted recursive image listing. Sorted so index -> file is stable."""
    return sorted(p for p in root.rglob("*")
                  if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES)


def resolve_task_sources(tasks: dict, data_root: Path) -> dict:
    """Turn config-relative task entries into absolute sources for the dataset.

    A task entry is either a relative path, or a ``{"input": ..., "target": ...}``
    mapping whose values are each made absolute. Both forms exist because RESIDE
    OTS ships ``synthetic/part1..4/`` beside ``clear/`` rather than
    ``input/``/``target/`` — see ``_paired_dirs``.

    Shared by the training entry point and the validation script so the two
    cannot resolve the same config differently, which is the sort of divergence
    that let B0-denoise train on a scope its config did not describe (F11).
    """
    from src.utils.config import REPO_ROOT

    out: dict = {}
    for task, spec in tasks.items():
        if isinstance(spec, dict):
            # Directories are data and resolve against data_root; a `list` is a
            # recorded artifact and lives with the repo, so it resolves against
            # REPO_ROOT. Silently resolving both the same way would put subset
            # manifests inside the dataset, where they are not version-tracked.
            out[task] = {k: ((REPO_ROOT / v) if k == "list" else (data_root / v))
                         for k, v in spec.items()}
        else:
            out[task] = data_root / spec
    return out


def _paired_dirs(task: str, spec) -> tuple[Path, Path]:
    """Resolve a derain/dehaze source to its (degraded, ground-truth) directories.

    Two forms, both explicit — there is no probing of one layout then falling
    back to another, because a fallback turns a mistyped path into a silently
    smaller dataset:

    ``Path``
        The ``input/`` + ``target/`` convention, as ``PairedTestDataset`` uses.
    ``{"input": ..., "target": ...}``
        Directories named outright. Needed for the published RESIDE OTS layout,
        which is ``synthetic/part1..4/`` beside ``clear/`` and has no ``input/``
        at all. Listing is recursive, so the ``partN`` split survives.
    """
    if isinstance(spec, dict):
        missing = {"input", "target"} - set(spec)
        if missing:
            raise ValueError(
                f"{task} source dict must give 'input' and 'target'; "
                f"missing {sorted(missing)}")
        input_dir, target_dir = Path(spec["input"]), Path(spec["target"])
        for role, d in (("input", input_dir), ("target", target_dir)):
            if not d.exists():
                raise FileNotFoundError(f"{task} {role} directory not found: {d}")
        return input_dir, target_dir

    root = Path(spec)
    input_dir, target_dir = root / "input", root / "target"
    for d in (input_dir, target_dir):
        if not d.exists():
            raise FileNotFoundError(
                f"{task} root {root} must contain input/ and target/; missing "
                f"{d.name}/. If the data uses a different layout, pass "
                f'{{"input": ..., "target": ...}} instead of a bare path.')
    return input_dir, target_dir


def _listed_images(spec, input_dir: Path) -> list[Path]:
    """Input images for a paired task: the whole directory, or an explicit list.

    A ``list`` key names a text file of paths relative to ``input_dir``. It
    exists so a run can train on a *recorded, reproducible subset* — the demo
    dehaze runs use a few thousand of RESIDE-OTS's 72,135 pairs, and "a few
    thousand chosen somehow" is not a result anyone can repeat. The file is the
    record; the script that wrote it records the seed.

    Every listed path must exist. A list that has drifted from the data on disk
    silently shrinks the training set, which is unrecoverable after the fact.
    """
    if not (isinstance(spec, dict) and spec.get("list")):
        return _list_images(input_dir)

    list_path = Path(spec["list"])
    if not list_path.exists():
        raise FileNotFoundError(f"subset list not found: {list_path}")
    # `#` comments carry the seed and counts that make the subset reproducible,
    # so the manifests are self-describing and the reader must skip them.
    names = [ln.strip() for ln in
             list_path.read_text(encoding="utf-8").splitlines()
             if ln.strip() and not ln.lstrip().startswith("#")]
    if not names:
        raise ValueError(f"subset list {list_path} is empty")
    files, missing = [], []
    for n in names:
        p = input_dir / n
        (files if p.exists() else missing).append(p)
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} of {len(names)} files in {list_path.name} are "
            f"missing under {input_dir}, e.g. {missing[0].name}")
    return files


def _pad_to_patch(img: np.ndarray, patch_size: int) -> np.ndarray:
    """Reflect-pad an image up to ``patch_size`` if it is smaller."""
    h, w = img.shape[:2]
    if h >= patch_size and w >= patch_size:
        return img
    return np.pad(img, ((0, max(0, patch_size - h)),
                        (0, max(0, patch_size - w)), (0, 0)), mode="reflect")


class _LRUImageCache:
    """Decoded images held under a hard byte budget, evicted least-recent-first.

    The cache MUST be bounded. Workers are persistent and each holds its own
    copy, so an unbounded dict converges on ``num_workers`` x the full decoded
    training set — 18.4 GB for 6 workers on 5,144 images, against 15.7 GB of
    RAM. That does not fail fast: the machine pages, the GPU starves, and the
    run dies hours in.

    Shared by both training datasets so the eviction policy is defined once.
    Keys are arbitrary hashables (a file index for denoise, a ``(task, index)``
    pair for the multi-task set).
    """

    def __init__(self, budget_gb: float) -> None:
        if budget_gb <= 0:
            raise ValueError(
                f"cache_budget_gb must be positive, got {budget_gb}. "
                "Pass cache_images=False to disable caching instead.")
        self.entries: OrderedDict[object, np.ndarray] = OrderedDict()
        self.budget = int(budget_gb * 2 ** 30)
        self.nbytes = 0

    def get(self, key, load) -> np.ndarray:
        """Return the cached array for ``key``, calling ``load()`` on a miss."""
        if key in self.entries:
            self.entries.move_to_end(key)          # mark as recently used
            return self.entries[key]
        img = load()
        # A single image larger than the whole budget would otherwise be
        # inserted and immediately evicted every time; skip caching it.
        if img.nbytes <= self.budget:
            self.entries[key] = img
            self.nbytes += img.nbytes
            while self.nbytes > self.budget:
                _, evicted = self.entries.popitem(last=False)   # LRU
                self.nbytes -= evicted.nbytes
        return img


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
            self.files.extend(_list_images(root))
        if not self.files:
            raise FileNotFoundError(f"no images under {roots}")

        self.sigmas = tuple(sigmas)
        self.patch_size = patch_size
        self.base_seed = base_seed
        self.length = length or len(self.files) * len(self.sigmas)
        # Decoding dominates for small models; cache to keep the GPU fed.
        self._lru = _LRUImageCache(cache_budget_gb) if cache_images else None

    #: Kept as properties because the cache internals are asserted on directly
    #: by tests/test_image_cache.py, which pins the eviction bound.
    @property
    def _cache(self):
        return None if self._lru is None else self._lru.entries

    @property
    def _cache_bytes(self) -> int:
        return 0 if self._lru is None else self._lru.nbytes

    @property
    def _cache_budget(self) -> int:
        return 0 if self._lru is None else self._lru.budget

    def __len__(self) -> int:
        return self.length

    def _image(self, idx: int) -> np.ndarray:
        file_idx = idx % len(self.files)
        # base=1: no crop-to-multiple for training, unlike evaluation.
        load = lambda: load_rgb_uint8(self.files[file_idx], base=1)
        return load() if self._lru is None else self._lru.get(file_idx, load)

    def __getitem__(self, idx: int):
        clean_full = self._image(idx)
        # Balanced across sigmas by construction rather than by chance.
        sigma = self.sigmas[idx % len(self.sigmas)]
        rng = np.random.default_rng((self.base_seed, idx))
        clean_full = _pad_to_patch(clean_full, self.patch_size)

        # Crop first, then synthesise noise on the crop: noise is i.i.d. so this
        # is equivalent to cropping a noised image, and is much cheaper.
        clean, _ = paired_transform(clean_full, clean_full,
                                    patch_size=self.patch_size, rng=rng)
        noise_state = np.random.RandomState(
            int(rng.integers(0, 2 ** 31 - 1)))  # noqa: NPY002 - legacy stream
        degraded = add_gaussian_noise(clean, sigma, rng=noise_state)
        return to_tensor(degraded), to_tensor(clean), sigma


class MultiTaskTrainDataset(Dataset):
    """The 3-degradation training set: denoise + derain + dehaze in one index space.

    Map-style rather than iterable so ``DataLoader(shuffle=True)`` and the
    balanced sampler (B2) can index it. The index space is a **concatenation of
    equal-length per-task streams**::

        [ 0 .. L )        denoise
        [ L .. 2L )       derain
        [ 2L .. 3L )      dehaze

    so a sampler balances tasks purely by which ranges it draws from, and never
    needs to ask the dataset what a given index will turn out to be. Stream
    length is ``length // n_tasks``, independent of how many files each task
    owns; a task with 200 pairs and a task with 5,144 images therefore
    contribute equally, with the smaller one repeating. This is the deliberate
    deviation from AdaIR described in the module docstring — AdaIR instead lets
    list length times a hand-tuned repeat multiplier decide each task's share.

    Every sample is a pure function of ``(base_seed, global index)``: the file,
    the crop, the flip/rotation and the noise realisation all derive from
    ``np.random.default_rng((base_seed, idx))``. That is what makes results
    independent of worker count and reproducible on resume.

    Yields ``(degraded, clean, meta)`` with CHW float tensors in ``[0, 1]`` and
    ``meta = {"task": int, "sigma": float}``. ``sigma`` is ``-1.0`` for
    non-denoise tasks — a value no real noise level can take, so it can never be
    mistaken for the clean-input case (sigma = 0) that F10 concerns.
    """

    def __init__(self, sources: dict[str, str | Path], *,
                 sigmas=(15, 25, 50), sigma_range: tuple[float, float] | None = None,
                 clean_prob: float = 0.0, patch_size: int = 256,
                 base_seed: int = 0, cache_images: bool = True,
                 length: int | None = None,
                 cache_budget_gb: float = 0.75) -> None:
        """Args:
            sources: task name -> source. ``denoise`` points at a directory of
                clean images. ``derain``/``dehaze`` point either at a directory
                holding ``input/`` and ``target/``, or at an explicit
                ``{"input": ..., "target": ...}`` mapping for datasets published
                in another layout — RESIDE OTS ships ``synthetic/part1..4/``
                beside ``clear/``. See ``_paired_dirs``.
            sigmas: discrete noise levels, cycled so each is equally
                represented. Used only when ``sigma_range`` is ``None``.
            sigma_range: ``(lo, hi)`` for CONTINUOUS noise sampling, which is the
                F10 fix — training on {15, 25, 50} alone left the model with no
                coverage below 15, and it degraded a nearly-clean input to
                125/255 mean absolute error. Overrides ``sigmas`` entirely.
            clean_prob: probability of drawing exactly sigma = 0. Uniform
                sampling gives the clean case measure zero, and sigma = 0 is the
                worst case F10 actually exhibited, so it gets explicit mass.
                Requires ``sigma_range``; meaningless with discrete cycling.
            length: total samples per epoch across all tasks. Defaults to the
                natural denoise stream length times the number of tasks, so
                adding a task widens the epoch rather than starving denoise.
        """
        if sigma_range is not None:
            lo, hi = (float(v) for v in sigma_range)
            if lo < 0 or hi <= lo:
                raise ValueError(
                    f"sigma_range must be (lo, hi) with 0 <= lo < hi, got {sigma_range}")
            sigma_range = (lo, hi)
        if not 0.0 <= clean_prob < 1.0:
            raise ValueError(f"clean_prob must be in [0, 1), got {clean_prob}")
        if clean_prob > 0 and sigma_range is None:
            raise ValueError(
                "clean_prob requires sigma_range: with discrete `sigmas` the "
                "noise level is cycled deterministically and there is no draw "
                "for a clean sample to displace.")
        self.sigma_range = sigma_range
        self.clean_prob = clean_prob

        if not sources:
            raise ValueError("sources is empty; pass at least one task")
        unknown = set(sources) - set(TASK_IDS)
        if unknown:
            raise ValueError(
                f"unknown task(s) {sorted(unknown)}; expected {sorted(TASK_IDS)}")

        self.tasks = [t for t in TASK_IDS if t in sources]   # fixed order
        self.sigmas = tuple(sigmas)
        self.patch_size = patch_size
        self.base_seed = base_seed

        # Resolve every file and every pair up front. A missing directory or an
        # unpairable image must stop the run now, not eight hours in.
        self.items: dict[str, list] = {}
        for task in self.tasks:
            spec = sources[task]
            root = Path(spec["input"] if isinstance(spec, dict) else spec)
            if not root.exists():
                raise FileNotFoundError(f"{task} root not found: {root}")
            if task == "denoise":
                items = [(p, None) for p in _list_images(root)]
            else:
                input_dir, target_dir = _paired_dirs(task, sources[task])
                inputs = _listed_images(sources[task], input_dir)
                items = [(p, resolve_pair_target(p, target_dir, task))
                         for p in inputs]
            if not items:
                raise FileNotFoundError(f"no images for task {task!r} under {root}")
            self.items[task] = items

        natural = len(self.items["denoise"]) * len(self.sigmas) \
            if "denoise" in self.items else len(self.items[self.tasks[0]])
        total = length or natural * len(self.tasks)
        self.stream_length = total // len(self.tasks)
        if self.stream_length < 1:
            raise ValueError(
                f"length={total} is too small for {len(self.tasks)} tasks; "
                "each task stream would be empty")
        self.length = self.stream_length * len(self.tasks)   # exact, no remainder

        self._lru = _LRUImageCache(cache_budget_gb) if cache_images else None

    def __len__(self) -> int:
        return self.length

    def task_ranges(self) -> dict[str, range]:
        """Index range owned by each task. The sampler's only view of layout."""
        return {t: range(i * self.stream_length, (i + 1) * self.stream_length)
                for i, t in enumerate(self.tasks)}

    def task_of(self, idx: int) -> str:
        """Which task global index ``idx`` belongs to."""
        if not 0 <= idx < self.length:
            raise IndexError(f"index {idx} out of range for length {self.length}")
        return self.tasks[idx // self.stream_length]

    def _image(self, key, path: Path) -> np.ndarray:
        # base=1: no crop-to-multiple for training, unlike evaluation.
        load = lambda: load_rgb_uint8(path, base=1)
        return load() if self._lru is None else self._lru.get(key, load)

    def _sigma_for(self, local: int, rng: np.random.Generator) -> float:
        """Noise level for one denoise sample.

        Discrete mode cycles on the LOCAL index so each sigma is equally
        represented regardless of task count, and draws nothing from ``rng`` —
        which keeps this path byte-identical to the pre-F10 behaviour.
        """
        if self.sigma_range is None:
            return float(self.sigmas[local % len(self.sigmas)])
        if self.clean_prob and rng.random() < self.clean_prob:
            return 0.0                          # exact identity task, not "low"
        return float(rng.uniform(*self.sigma_range))

    def __getitem__(self, idx: int):
        task = self.task_of(idx)
        local = idx % self.stream_length
        items = self.items[task]
        item_idx = local % len(items)
        deg_path, tgt_path = items[item_idx]
        rng = np.random.default_rng((self.base_seed, idx))

        if task == "denoise":
            clean_full = self._image((task, item_idx), deg_path)
            degraded_full = clean_full                      # noise added post-crop
        else:
            degraded_full = self._image((task, "in", item_idx), deg_path)
            clean_full = self._image((task, "gt", item_idx), tgt_path)
            if degraded_full.shape != clean_full.shape:
                raise ValueError(
                    f"{deg_path.name}: degraded {degraded_full.shape} != "
                    f"clean {clean_full.shape}")

        degraded_full = _pad_to_patch(degraded_full, self.patch_size)
        clean_full = _pad_to_patch(clean_full, self.patch_size)
        degraded, clean = paired_transform(degraded_full, clean_full,
                                           patch_size=self.patch_size, rng=rng)

        if task == "denoise":
            sigma = self._sigma_for(local, rng)
            noise_state = np.random.RandomState(
                int(rng.integers(0, 2 ** 31 - 1)))  # noqa: NPY002 - legacy stream
            degraded = add_gaussian_noise(clean, sigma, rng=noise_state)
        else:
            sigma = -1.0                            # not applicable, never 0

        return (to_tensor(degraded), to_tensor(clean),
                {"task": TASK_IDS[task], "sigma": sigma})


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


def build_multitask_loader(sources: dict[str, str | Path], *,
                           batch_size: int = 16, patch_size: int = 256,
                           sigmas=(15, 25, 50),
                           sigma_range: tuple[float, float] | None = None,
                           clean_prob: float = 0.0, num_workers: int = 8,
                           seed: int = 0, length: int | None = None,
                           cache_budget_gb: float = 0.75
                           ) -> torch.utils.data.DataLoader:
    """Build the 3-degradation training loader with per-batch task balancing.

    Uses ``batch_sampler``, so ``batch_size``/``shuffle``/``drop_last`` are the
    sampler's job — passing them to the DataLoader as well is an error PyTorch
    raises rather than silently resolving.

    ``cache_budget_gb`` is PER WORKER — total resident cache is roughly
    ``num_workers * cache_budget_gb``. Size it against real RAM, not the dataset.
    """
    dataset = MultiTaskTrainDataset(sources, sigmas=sigmas,
                                    sigma_range=sigma_range,
                                    clean_prob=clean_prob, patch_size=patch_size,
                                    base_seed=seed, length=length,
                                    cache_budget_gb=cache_budget_gb)
    # The trainer consumes MICRO-batches, so the batch budget is derived from
    # the sample budget the caller already sized against total_iters * accum.
    num_batches = max(1, len(dataset) // batch_size)
    sampler = BalancedTaskBatchSampler(dataset.task_ranges(), batch_size,
                                       num_batches=num_batches, base_seed=seed)
    return torch.utils.data.DataLoader(
        dataset, batch_sampler=sampler,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )
