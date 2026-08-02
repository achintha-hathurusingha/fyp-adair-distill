"""Per-batch task balancing for the 3-degradation training mix.

**The deviation from AdaIR, stated precisely.** AdaIR does no per-batch
balancing: ``sample_ids`` is a flat concatenation of per-task streams with
hand-tuned repeat multipliers (derain x120, denoise x3 per sigma, dehaze x1;
``dataset_utils.py:155-168``), shuffled once. A task's share of any given batch
is therefore whatever the shuffle produced — a batch can legitimately be all
dehaze. We instead hold the mix fixed *within* every batch. Both are defensible;
ours makes the gradient each optimizer step sees an all-in-one gradient rather
than a task-specialist one, and it makes "what was this model trained on" an
assertion rather than an expectation over a long run.

That distinction is not academic here. B0-denoise trained on one task for its
entire life and nothing noticed (finding F11). A sampler whose batch composition
can be asserted turns that class of failure into a test.
"""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from torch.utils.data import Sampler


class BalancedTaskBatchSampler(Sampler[list[int]]):
    """Yields index batches holding an equal share of every task.

    Works purely from the dataset's task -> index-range map, so it never has to
    ask the dataset what a given index will turn out to be.

    **Uneven batch sizes.** If ``batch_size`` is not divisible by the number of
    tasks, the leftover slots cannot be shared within a single batch. They are
    handed out on a rotation across batches instead, so every task receives the
    same number of extras over an epoch — batch 0 gives its extra to denoise,
    batch 1 to derain, and so on. Rounding a quota up or down for a fixed task
    would give that task a permanent few-percent advantage.

    **Coverage, not i.i.d. draws.** Within a task the sampler walks a shuffled
    permutation of that task's range and reshuffles only once exhausted, so
    every index is used equally often. Independent draws would leave some images
    unseen and others repeated several times in the same epoch.

    Deterministic given ``(base_seed, epoch)``: the same seed yields the same
    batches, on any machine, at any worker count.
    """

    def __init__(self, task_ranges: dict[str, range], batch_size: int, *,
                 num_batches: int, base_seed: int = 0) -> None:
        """Args:
            task_ranges: task name -> the contiguous index range it owns, as
                returned by ``MultiTaskTrainDataset.task_ranges()``.
            batch_size: samples per batch (the MICRO-batch under accumulation).
            num_batches: batches to yield per epoch. The trainer is
                iteration-driven, so this is a budget rather than a dataset pass.
            base_seed: run seed. Folded together with the epoch and the task.
        """
        if not task_ranges:
            raise ValueError("task_ranges is empty; nothing to sample")
        n_tasks = len(task_ranges)
        if batch_size < n_tasks:
            raise ValueError(
                f"batch_size={batch_size} cannot hold all {n_tasks} tasks; "
                "a balanced batch needs at least one slot per task")
        if num_batches < 1:
            raise ValueError(f"num_batches must be >= 1, got {num_batches}")
        for task, rng in task_ranges.items():
            if len(rng) == 0:
                raise ValueError(f"task {task!r} owns no indices")

        self.tasks = list(task_ranges)
        self.ranges = task_ranges
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.base_seed = base_seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Reshuffle for a new pass. Explicit, never auto-incremented — an
        implicit bump would make a resumed run diverge from a fresh one."""
        self.epoch = epoch

    def quotas(self, batch_index: int) -> dict[str, int]:
        """Per-task slot counts for one batch. Public so tests can assert it."""
        n = len(self.tasks)
        base, extra = divmod(self.batch_size, n)
        counts = {t: base for t in self.tasks}
        for k in range(extra):                  # rotate the leftovers
            counts[self.tasks[(batch_index * extra + k) % n]] += 1
        return counts

    def _permutation(self, task: str, cycle: int) -> np.ndarray:
        # Seeded on the task's POSITION, not hash(task): str hashing is salted
        # per process, so a name-derived seed would silently give a different
        # batch order on every launch.
        rng = np.random.default_rng(
            (self.base_seed, self.epoch, self.tasks.index(task), cycle))
        return rng.permutation(np.arange(self.ranges[task].start,
                                         self.ranges[task].stop))

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[list[int]]:
        perms = {t: self._permutation(t, 0) for t in self.tasks}
        pos = dict.fromkeys(self.tasks, 0)
        cycle = dict.fromkeys(self.tasks, 0)

        for b in range(self.num_batches):
            batch: list[int] = []
            for task, quota in self.quotas(b).items():
                for _ in range(quota):
                    if pos[task] >= len(perms[task]):
                        cycle[task] += 1
                        perms[task] = self._permutation(task, cycle[task])
                        pos[task] = 0
                    batch.append(int(perms[task][pos[task]]))
                    pos[task] += 1
            # Not shuffled within the batch: nothing in NAFNet mixes samples
            # across the batch dimension (no BatchNorm; LayerNorm2d and SCA are
            # both per-sample), so within-batch order is unobservable. Leaving
            # it grouped by task keeps a printed batch readable.
            yield batch
