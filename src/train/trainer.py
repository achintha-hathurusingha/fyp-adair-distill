"""Training loop for the normalization quality ablation (Task 1.5b).

Built to the standing rules: config-driven, seeded, fully resumable, and every
run writes a directory with the resolved config, git hash and `pip freeze`.

Diagnostics are the point of this task, not an afterthought. Alongside PSNR/SSIM
we log **gradient norm** and **mean activation magnitude per pyramid level**,
because those are what separate a *trainability* failure (activations or
gradients blowing up / collapsing) from a *capacity* failure (stable training
that simply plateaus lower). Neither can be recovered after the fact.

Validation always goes through the locked harness ``src/eval/evaluate.py`` — no
inline metrics anywhere.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.data.datasets import build_dataset
from src.eval.evaluate import evaluate
from src.eval.metrics import ADAIR_DEFAULT
from src.losses.reconstruction import build_loss
from src.utils.logging import get_logger
from src.utils.seeding import capture_rng_state, restore_rng_state, seed_everything


@dataclass
class TrainState:
    """Everything needed to resume exactly."""

    iteration: int = 0
    best_psnr: float = -1.0
    history: list[dict[str, Any]] = field(default_factory=list)


class EMA:
    """Exponential moving average of model weights."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(),
                                                     alpha=1 - self.decay)

    def copy_to(self, model: nn.Module) -> dict:
        """Swap EMA weights in, returning the originals for restoration."""
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()
                  if k in self.shadow}
        model.load_state_dict({**model.state_dict(),
                               **{k: v.to(next(model.parameters()).dtype)
                                  for k, v in self.shadow.items()}},
                              strict=False)
        return backup

    def restore(self, model: nn.Module, backup: dict) -> None:
        model.load_state_dict({**model.state_dict(), **backup}, strict=False)

    def state_dict(self) -> dict:
        return {k: v.cpu() for k, v in self.shadow.items()}

    def load_state_dict(self, sd: dict) -> None:
        self.shadow = {k: v.clone() for k, v in sd.items()}


@torch.no_grad()
def activation_stats(model: nn.Module, sample: torch.Tensor) -> dict[str, float]:
    """Mean |activation| at each encoder stage — the trainability diagnostic.

    A healthy network keeps these bounded and comparable across depth. Runaway
    growth (or collapse) through the pyramid is the signature of a normalisation
    removal that has destabilised training, as distinct from one that merely
    costs accuracy.
    """
    stats: dict[str, float] = {}
    handles = []

    def hook(name):
        def fn(_module, _inp, out):
            if isinstance(out, torch.Tensor):
                stats[name] = float(out.detach().abs().mean())
        return fn

    was_training = model.training
    model.eval()
    for i, enc in enumerate(model.encoders):
        handles.append(enc.register_forward_hook(hook(f"enc{i}")))
    handles.append(model.middle_blks.register_forward_hook(hook("middle")))
    try:
        model(sample)
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)
    return stats


def grad_norm(model: nn.Module) -> float:
    """Global L2 gradient norm across all parameters."""
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().norm(2)) ** 2
    return total ** 0.5


class Trainer:
    """Fixed-iteration trainer with resumable state and diagnostic logging."""

    def __init__(self, model: nn.Module, loader, cfg: dict, run_dir: Path, *,
                 device: str = "cuda", val_root: Path | None = None) -> None:
        self.model = model.to(device)
        self.loader = loader
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.device = device
        self.val_root = val_root
        self.log = get_logger("train", run_dir=run_dir)

        opt_cfg = cfg.get("optim", {})
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=opt_cfg.get("lr", 1e-3),
            weight_decay=opt_cfg.get("weight_decay", 1e-4),
            betas=tuple(opt_cfg.get("betas", (0.9, 0.9))))
        self.grad_clip = opt_cfg.get("grad_clip")

        sch = cfg.get("schedule", {})
        self.total_iters = sch.get("total_iters", 50_000)
        self.warmup_iters = sch.get("warmup_iters", 2_000)
        self.min_lr = sch.get("min_lr", 1e-6)
        self.base_lr = opt_cfg.get("lr", 1e-3)

        tr = cfg.get("train", {})
        # Micro-batches accumulated per optimizer step. Effective batch is
        # data.batch_size * accum_steps; see findings F8.
        self.accum_steps = max(1, int(tr.get("accum_steps", 1)))
        self.amp = tr.get("amp", True)
        self.val_every = tr.get("val_every", 2_000)
        self.ckpt_every = tr.get("ckpt_every", 2_000)
        self.ema = EMA(self.model, tr.get("ema_decay", 0.999))
        self.criterion = build_loss(cfg.get("loss", {"name": "charbonnier"}))
        self.state = TrainState()

    def _lr_at(self, it: int) -> float:
        """Linear warmup then cosine decay."""
        import math
        if it < self.warmup_iters:
            return self.base_lr * (it + 1) / max(1, self.warmup_iters)
        progress = (it - self.warmup_iters) / max(1, self.total_iters - self.warmup_iters)
        cos = 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
        return self.min_lr + (self.base_lr - self.min_lr) * cos

    def save_checkpoint(self, path: Path) -> None:
        """Save everything needed for an exact resume."""
        rng = capture_rng_state()
        torch.save({
            "iteration": self.state.iteration,
            "best_psnr": self.state.best_psnr,
            "history": self.state.history,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "ema": self.ema.state_dict(),
            "rng_python": rng.python,
            "rng_numpy": rng.numpy,
            "rng_torch": rng.torch,
            "rng_cuda": rng.torch_cuda,
            "config": self.cfg,
        }, path)

    def load_checkpoint(self, path: Path) -> None:
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ck["model"])
        self.optimizer.load_state_dict(ck["optimizer"])
        self.ema.load_state_dict(ck["ema"])
        self.state.iteration = ck["iteration"]
        self.state.best_psnr = ck["best_psnr"]
        self.state.history = ck.get("history", [])
        from src.utils.seeding import RNGState
        restore_rng_state(RNGState(ck["rng_python"], ck["rng_numpy"],
                                   ck["rng_torch"], ck["rng_cuda"]))
        self.log.info(f"resumed from {path} at iteration {self.state.iteration}")

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """Evaluate EMA weights on BSD68 through the LOCKED harness.

        Returns an empty dict when no ``val_root`` was supplied. Note that
        validation always runs on the final iteration regardless of
        ``val_every``, so this path is reachable in any run configured without
        a validation set — it is warned about rather than allowed to raise deep
        inside the training loop.
        """
        if self.val_root is None:
            self.log.warning("no val_root configured — skipping validation")
            return {}
        backup = self.ema.copy_to(self.model)
        self.model.eval()
        try:
            results = {}
            for sigma in (15, 25, 50):
                ds = build_dataset("denoise", self.val_root, sigma=sigma,
                                   seed_mode="filename")
                res = evaluate(self.model, iter(ds), name=f"s{sigma}",
                               config=ADAIR_DEFAULT, device=self.device,
                               keep_per_image=False)
                results[f"psnr_s{sigma}"] = res.psnr
                results[f"ssim_s{sigma}"] = res.ssim
            results["psnr"] = sum(results[f"psnr_s{s}"] for s in (15, 25, 50)) / 3
            results["ssim"] = sum(results[f"ssim_s{s}"] for s in (15, 25, 50)) / 3
            return results
        finally:
            self.ema.restore(self.model, backup)
            self.model.train()

    def train(self) -> TrainState:
        """Run to ``total_iters``, validating and checkpointing periodically."""
        torch.cuda.reset_peak_memory_stats() if self.device == "cuda" else None
        self.model.train()
        scaler = torch.amp.GradScaler("cuda", enabled=False)  # bf16 needs no scaler
        it = self.state.iteration
        t0 = time.time()
        loss_accum, n_accum = 0.0, 0
        # Running max gradient norm since the last validation. Sampling gnorm
        # only at validation points cannot distinguish a single spike from
        # gradual drift — the exact question the Task 1.5b divergences turned
        # on. Tracking the max between logs removes that blind spot.
        max_gnorm = 0.0
        clip_hits = 0
        # Anchors the clip-hit rate to the interval actually covered, which is
        # shorter than val_every on the first log after a resume.
        last_log_it = it

        # Gradient accumulation. `it` counts OPTIMIZER STEPS, not micro-batches,
        # so `total_iters` and the LR schedule mean the same thing whether or not
        # accumulation is active — which is what keeps a run at micro-batch 16 x
        # 2 comparable with one at native batch 32 (findings F8).
        micro = 0
        self.optimizer.zero_grad(set_to_none=True)

        while it < self.total_iters:
            for degraded, clean, _sigma in self.loader:
                if it >= self.total_iters:
                    break
                lr = self._lr_at(it)
                for g in self.optimizer.param_groups:
                    g["lr"] = lr

                degraded = degraded.to(self.device, non_blocking=True)
                clean = clean.to(self.device, non_blocking=True)

                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
                    pred = self.model(degraded)
                    loss = self.criterion(pred.float(), clean)

                # Divide so accumulated gradients AVERAGE over the effective
                # batch rather than summing — otherwise the effective learning
                # rate scales with accum_steps.
                (loss / self.accum_steps).backward()
                loss_accum += float(loss.detach())
                n_accum += 1
                micro += 1

                if not torch.isfinite(loss):
                    self.log.error(
                        f"non-finite loss at optimizer step {it} "
                        f"(micro-batch {micro}) — diverged")
                    self.state.history.append(
                        {"iteration": it, "diverged": True,
                         "loss": float("nan"), "max_grad_norm": max_gnorm,
                         "clip_hits": clip_hits})
                    self._dump_history()
                    return self.state

                if micro < self.accum_steps:
                    continue                      # keep accumulating

                # --- one optimizer step, on the full effective batch ---
                gn = grad_norm(self.model)
                max_gnorm = max(max_gnorm, gn)
                if self.grad_clip:
                    if gn > self.grad_clip:
                        clip_hits += 1
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   self.grad_clip)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.ema.update(self.model)
                micro = 0

                it += 1
                self.state.iteration = it

                if it % self.val_every == 0 or it == self.total_iters:
                    acts = activation_stats(self.model, degraded[:1])
                    metrics = self.validate()
                    if not metrics:
                        metrics = {"psnr": float("nan"), "ssim": float("nan")}
                    peak = (torch.cuda.max_memory_allocated() / 2**30
                            if self.device == "cuda" else 0.0)
                    # Both counters cover only the interval since the last log
                    # (they are reset below), so the rate is per-interval.
                    steps_since = it - last_log_it
                    row = {
                        "iteration": it,
                        "lr": lr,
                        "loss": loss_accum / max(1, n_accum),
                        "grad_norm": gn,
                        "max_grad_norm": max_gnorm,
                        "clip_hits": clip_hits,
                        "clip_rate": clip_hits / max(1, steps_since),
                        "peak_vram_gb": peak,
                        "elapsed_s": time.time() - t0,
                        **metrics,
                        **{f"act_{k}": v for k, v in acts.items()},
                    }
                    self.state.history.append(row)
                    self.log.info(
                        f"it {it:6d}  loss {row['loss']:.5f}  psnr {metrics['psnr']:.3f}  "
                        f"ssim {metrics['ssim']:.4f}  gnorm {gn:.3f}  "
                        f"maxgn {max_gnorm:.3f}  "
                        f"clip {clip_hits}/{steps_since} ({row['clip_rate']:.1%})  "
                        f"vram {peak:.2f}GB")
                    loss_accum, n_accum = 0.0, 0
                    max_gnorm, clip_hits = 0.0, 0
                    last_log_it = it
                    self._dump_history()

                    if metrics["psnr"] > self.state.best_psnr:
                        self.state.best_psnr = metrics["psnr"]
                        self.save_checkpoint(self.run_dir / "best.pth")

                if it % self.ckpt_every == 0:
                    self.save_checkpoint(self.run_dir / "last.pth")

        self.save_checkpoint(self.run_dir / "last.pth")
        self._dump_history()
        return self.state

    def _dump_history(self) -> None:
        (self.run_dir / "history.json").write_text(
            json.dumps(self.state.history, indent=2), encoding="utf-8")
