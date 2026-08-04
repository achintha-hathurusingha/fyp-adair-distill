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
from src.models.norms import (AffineClampNorm2d, LayerNorm2dClamp,
                              reset_clamp_engagement)
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
        # Which task validation runs on. Defaults to denoise so B0-denoise
        # and B0-v2 are unchanged; a single-task run overrides it, because
        # validating a dehaze model on BSD68 measures nothing it trains for.
        self.val_task = (cfg.get("eval") or {}).get("val_task", "denoise")

        # Optional frozen teacher for response distillation. Absent by default,
        # and every baseline in this project asserts it stays absent -- a
        # baseline that quietly gained a teacher term would invalidate every
        # delta measured against it.
        dcfg = cfg.get("distill") or {}
        self.teacher = None
        self.kd_weight = float(dcfg.get("weight", 0.0))
        self._kd_last = 0.0
        if dcfg.get("teacher"):
            if self.kd_weight <= 0:
                raise ValueError(
                    "distill.teacher is set but distill.weight is not positive; "
                    "that would load a teacher and ignore it.")
            from src.models.teacher_wrapper import load_teacher
            self.teacher = load_teacher(dcfg["teacher"], device=device)
            self.log.info(f"teacher: {Path(dcfg['teacher']).name} "
                          f"(frozen, eval) | kd weight {self.kd_weight}")
        self.log = get_logger("train", run_dir=run_dir)

        opt_cfg = cfg.get("optim", {})
        wd = opt_cfg.get("weight_decay", 1e-4)
        # The residual scales (NAFBlock.beta/gamma) already get `wd` like every
        # other parameter. `residual_weight_decay` splits them into their own
        # group so they can be decayed HARDER — they are the parameters that
        # gate how much each block adds to the residual stream, so they are the
        # direct lever on F6's growth mechanism. Opt-in: absent, behaviour is
        # byte-identical to a single group.
        res_wd = opt_cfg.get("residual_weight_decay")
        if res_wd is None:
            params = self.model.parameters()
        else:
            residual, rest = [], []
            for name, p in self.model.named_parameters():
                (residual if name.endswith((".beta", ".gamma")) else rest).append(p)
            if not residual:
                raise ValueError(
                    "residual_weight_decay set but no .beta/.gamma parameters "
                    "found — the model does not use residual scaling.")
            params = [{"params": rest, "weight_decay": wd},
                      {"params": residual, "weight_decay": res_wd}]
            self.log.info(
                f"residual scales in own group: {len(residual)} params, "
                f"weight_decay {res_wd} (rest: {wd})")
        self.optimizer = torch.optim.AdamW(
            params,
            lr=opt_cfg.get("lr", 1e-3),
            weight_decay=wd,
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
        # Per-step diagnostic trace. val_every (5000) is far too coarse to see a
        # divergence onset: B0 died between two validation points having logged
        # only "clip 1/5000", which says a single step spiked to 6.5e7 but not
        # when, nor what the loss was doing around it. 0 disables.
        self.trace_every = int(tr.get("trace_every", 0))
        self._trace_fh = None
        # When the gradient norm exceeds this, dump the exact micro-batches that
        # produced it. Reconstructing them from the seed offline is fragile —
        # model init and RNG restore both consume the global stream — so the
        # batch is caught in the act instead. 0 disables.
        self.spike_dump = float(tr.get("spike_dump_threshold", 0.0))
        self._recent: list = []
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

    def _clamp_stats(self) -> dict[str, float]:
        """Fraction of forwards in which a clamp actually engaged.

        Rare engagement means the bound is insurance. Frequent engagement means
        it is silently reshaping normal training, and the bound is too tight —
        which would trade a divergence for a quality regression.
        """
        # Reported PER SITE. The two clamps guard different failure modes at
        # opposite ends of the network -- affine_clamp at full resolution (F9,
        # dec3) and layernorm_clamp deep in the encoder (F10, enc3) -- so
        # summing them would let a rising deep-stage rate hide inside a
        # full-resolution rate that is already percent-scale (F12).
        #
        # The unprefixed keys stay on the full-resolution clamp so history.json
        # from B0-denoise and from B0-v2 remain directly comparable, and
        # scripts/trend_test.py keeps reading both without a special case.
        groups = (("clamp", AffineClampNorm2d), ("deep_clamp", LayerNorm2dClamp))
        out: dict[str, float] = {}
        for prefix, cls in groups:
            mods = [m for m in self.model.modules() if isinstance(m, cls)]
            fwd = sum(getattr(m, "forwards", 0) for m in mods)
            if not fwd:
                continue
            eng = sum(getattr(m, "engaged", 0) for m in mods)
            els = sum(getattr(m, "elements_clamped", 0) for m in mods)
            mag = max((getattr(m, "max_preclamp", 0.0) for m in mods), default=0.0)
            for m in mods:          # reset each interval
                reset_clamp_engagement(m)
            out[f"{prefix}_engage_rate"] = eng / fwd
            out[f"{prefix}_elements"] = els
            out[f"{prefix}_max_preclamp"] = mag
        return out

    def _trace(self, it: int, lr: float, loss: float, gn: float) -> None:
        """Append one CSV row of per-step diagnostics.

        Flushed every row: this exists to survive the crash it is diagnosing, so
        buffered output would defeat the purpose.
        """
        if self._trace_fh is None:
            path = self.run_dir / "trace.csv"
            new = not path.exists()
            self._trace_fh = path.open("a", encoding="utf-8")
            if new:
                self._trace_fh.write("iteration,lr,loss,grad_norm\n")
        self._trace_fh.write(f"{it},{lr:.8g},{loss:.8g},{gn:.8g}\n")
        self._trace_fh.flush()

    def _fp32_recheck(self, gn_amp: float) -> float:
        """Recompute the current step's gradient in fp32 and return its norm.

        The decisive precision test: same model state, same batch, same loss —
        only the arithmetic precision differs. Running it inline rather than as
        a separate job matters, because changing AMP at the start of a resumed
        run forks the trajectory and the model never reaches the same state.

        Parameters are fp32 already (AMP only autocasts ops), so disabling
        autocast gives a genuine fp32 backward. The live gradients are saved and
        restored, so this observes without perturbing training.
        """
        saved = {n: (p.grad.detach().clone() if p.grad is not None else None)
                 for n, p in self.model.named_parameters()}
        self.optimizer.zero_grad(set_to_none=True)
        try:
            for degraded, clean in self._recent:
                d = degraded.to(self.device)
                c = clean.to(self.device)
                with torch.autocast("cuda", enabled=False):
                    loss = self.criterion(self.model(d), c)
                (loss / self.accum_steps).backward()
            gn_fp32 = grad_norm(self.model)
        finally:
            for n, p in self.model.named_parameters():
                p.grad = saved[n]
        self.log.warning(
            f"    PRECISION RECHECK: bf16 grad norm {gn_amp:.6e}  vs  "
            f"fp32 grad norm {gn_fp32:.6e}  (ratio {gn_amp / max(gn_fp32, 1e-30):.3e})")
        return gn_fp32

    def _dump_spike(self, it: int, gn: float) -> None:
        """Save the micro-batches responsible for an anomalous gradient."""
        out = self.run_dir / "spikes"
        out.mkdir(exist_ok=True)
        path = out / f"step_{it}_gn_{gn:.3e}.pt"
        torch.save({"iteration": it, "grad_norm": gn,
                    "micro_batches": self._recent,
                    "model": {k: v.detach().cpu()
                              for k, v in self.model.state_dict().items()}},
                   path)
        stats = []
        for i, (d, c) in enumerate(self._recent):
            stats.append(
                f"    micro{i}: degraded[min {float(d.min()):.4f} "
                f"max {float(d.max()):.4f} mean {float(d.mean()):.4f}] "
                f"clean[min {float(c.min()):.4f} max {float(c.max()):.4f} "
                f"mean {float(c.mean()):.4f}] "
                f"nonfinite={int((~torch.isfinite(d)).sum()) + int((~torch.isfinite(c)).sum())}")
        self.log.warning(
            f"GRADIENT SPIKE at step {it}: norm {gn:.6e}\n"
            + "\n".join(stats)
            + f"\n    batch saved to {path}")
        if self.amp and self.device == "cuda":
            self._fp32_recheck(gn)

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
            if self.val_task != "denoise":
                # Paired tasks have no sigma axis: one pass over input/target.
                # Same locked harness and the same ADAIR_DEFAULT conventions --
                # only the dataset differs.
                ds = build_dataset(self.val_task, self.val_root)
                res = evaluate(self.model, iter(ds), name=self.val_task,
                               config=ADAIR_DEFAULT, device=self.device,
                               keep_per_image=False)
                return {"psnr": res.psnr, "ssim": res.ssim,
                        f"psnr_{self.val_task}": res.psnr,
                        f"ssim_{self.val_task}": res.ssim}
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
        # Optimizer steps dropped because the gradient was Inf/NaN. Counted and
        # logged rather than silently swallowed: a run that only survives by
        # skipping steps is not healthy, it is hiding.
        nonfinite_skips = 0
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
            # Third element is per-sample provenance -- a sigma from the
            # denoise loader, a {"task", "sigma"} dict from the multi-task one.
            # The loss uses neither; it is carried for diagnostics and for the
            # batch-composition assertions that F11 would have been caught by.
            for degraded, clean, _provenance in self.loader:
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
                    if self.teacher is not None:
                        # Response distillation: one extra term, matching the
                        # teacher's OUTPUT on the same input. No hooks, no
                        # adapters, no feature matching -- the student never
                        # learns to compute frequencies, only to reproduce what
                        # doing so produced (F7).
                        with torch.no_grad():
                            soft = self.teacher(degraded)
                        kd = self.criterion(pred.float(), soft.float())
                        loss = loss + self.kd_weight * kd
                        self._kd_last = float(kd)

                # Divide so accumulated gradients AVERAGE over the effective
                # batch rather than summing — otherwise the effective learning
                # rate scales with accum_steps.
                if self.spike_dump:
                    self._recent.append((degraded.detach().cpu(),
                                         clean.detach().cpu()))
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
                if self.trace_every and it % self.trace_every == 0:
                    self._trace(it, lr, float(loss.detach()), gn)
                if self.spike_dump and gn > self.spike_dump:
                    self._dump_spike(it, gn)
                self._recent = []

                # Non-finite gradients must never reach the weights. Clipping is
                # NOT a guard against them: clip_grad_norm_ computes
                # clip_coef = max_norm / (total_norm + eps), which for an Inf
                # norm is ~0, and inf * 0 = nan — so clipping actively CONVERTS
                # an Inf gradient into NaN weights. Skip the step instead; the
                # gradients are zeroed below, so the batch is simply dropped.
                # Falls through to the normal iteration accounting below rather
                # than `continue`-ing: a skipped step is still a step, and must
                # still validate, log and checkpoint. Otherwise a run in which
                # every step is skipped would produce no diagnostics at all —
                # exactly when they are most needed.
                if not torch.isfinite(torch.tensor(gn)):
                    nonfinite_skips += 1
                    self.log.warning(
                        f"non-finite gradient norm ({gn}) at optimizer step "
                        f"{it}; skipping step (total skipped: {nonfinite_skips})")
                    self.optimizer.zero_grad(set_to_none=True)
                else:
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
                    # Read (and reset) clamp counters BEFORE any diagnostic or
                    # validation forward pass. activation_stats runs one forward
                    # and validate() runs one per BSD68 image, at full
                    # resolution rather than 128px crops — counting those would
                    # measure engagement on a different input distribution than
                    # the one training actually sees, which is the whole point
                    # of the metric.
                    clamp_stats = self._clamp_stats()
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
                        "nonfinite_skips": nonfinite_skips,
                        **clamp_stats,
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
                        f"skip {nonfinite_skips}  "
                        + (f"clampeng {row['clamp_engage_rate']:.2%} "
                           f"premax {row['clamp_max_preclamp']:.4g}  "
                           if "clamp_engage_rate" in row else "")
                        # Deep-stage clamp printed only when it EXISTS, and
                        # its engagement is the F12 watch item: it is
                        # expected to stay at 0.00%, so a non-zero value
                        # here is the signal, not noise.
                        + (f"deepeng {row['deep_clamp_engage_rate']:.2%} "
                           f"deeppremax {row['deep_clamp_max_preclamp']:.4g}  "
                           if "deep_clamp_engage_rate" in row else "")
                        + f"vram {peak:.2f}GB")
                    loss_accum, n_accum = 0.0, 0
                    max_gnorm, clip_hits = 0.0, 0
                    nonfinite_skips = 0
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
