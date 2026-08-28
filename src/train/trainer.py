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
import torch.nn.functional as F
from torch import nn

from src.data.datasets import build_dataset
from src.eval.evaluate import evaluate
from src.eval.metrics import ADAIR_DEFAULT
from src.losses.frequency import spectrum_loss
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

    def _capture_student_middle(self, _module, _inp, out: torch.Tensor) -> None:
        """Forward hook on ``model.middle_blks``, capturing its output for the
        feature-KD loss. Registered only when ``feat_weight > 0`` (see
        ``__init__``), so this never runs — and never costs anything — for
        any existing baseline or kd_freq config."""
        self._student_middle_capture = out

    def __init__(self, model: nn.Module, loader, cfg: dict, run_dir: Path, *,
                 device: str = "cuda", val_root: Path | None = None,
                 val_tasks: dict[str, Path] | None = None) -> None:
        self.model = model.to(device)
        self.loader = loader
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.device = device
        self.val_root = val_root
        # Multi-task validation (kd_feature_multitask B0V2 eval-gap fix — see
        # reports/kd_feature_multitask/plan.md, section 4): a {task: root}
        # mapping, one held-out set PER task. Set only for a mixed_task run
        # whose config actually asks for it (train.py); every existing
        # single-task arm leaves this None and validate() takes the original
        # single-val_root/val_task path below, unchanged.
        self.val_tasks = val_tasks
        # Which task validation runs on. Defaults to denoise so B0-denoise
        # and B0-v2 are unchanged; a single-task run overrides it, because
        # validating a dehaze model on BSD68 measures nothing it trains for.
        # Irrelevant when val_tasks is set (each task carries its own name).
        self.val_task = (cfg.get("eval") or {}).get("val_task", "denoise")
        self.log = get_logger("train", run_dir=run_dir)

        # Optional frozen teacher for response distillation. Absent by default,
        # and every baseline in this project asserts it stays absent -- a
        # baseline that quietly gained a teacher term would invalidate every
        # delta measured against it.
        dcfg = cfg.get("distill") or {}
        self.teacher = None
        self.kd_weight = float(dcfg.get("weight", 0.0))
        self._kd_last = 0.0
        # Frequency-domain term (F7): asks the student to match the teacher's
        # SPECTRUM, not just its pixels. Zero weight = absent, and the baselines
        # assert it stays zero.
        self.freq_weight = float(dcfg.get("freq_weight", 0.0))
        self.freq_mode = dcfg.get("freq_mode", "magnitude")
        self._freq_last = 0.0
        # `teacher_task` (portable) or `teacher` (explicit path, for a one-off).
        # The task form is preferred: an absolute path in a tracked config is
        # one machine's, which is the rule configs/paths.yaml opens with.
        teacher_path = None
        if dcfg.get("teacher_task"):
            from src.utils.config import teacher_checkpoint
            teacher_path = teacher_checkpoint(dcfg["teacher_task"])
        elif dcfg.get("teacher"):
            teacher_path = Path(dcfg["teacher"])
        if self.freq_weight > 0 and teacher_path is None:
            raise ValueError(
                "distill.freq_weight is set without a teacher; the "
                "frequency term compares against the TEACHER's spectrum and "
                "has nothing to compare to without one.")
        if teacher_path is not None:
            if self.kd_weight <= 0:
                raise ValueError(
                    "a distillation teacher is configured but distill.weight is "
                    "not positive; that would load a teacher and ignore it.")
            from src.models.teacher_wrapper import load_teacher
            self.teacher = load_teacher(teacher_path, device=device)
            self.log.info(f"teacher: {teacher_path.name} "
                          f"(frozen, eval) | kd weight {self.kd_weight}")

        # Feature-level distillation on the teacher's internal `latent_pre`
        # bottleneck, not final-output pixels (kd_feature experiment — see
        # reports/kd_feature/plan.md). Zero weight = absent, same discipline
        # as freq_weight; every existing baseline/kd_freq config leaves this
        # unset and is therefore byte-identical to before this was added.
        self.feat_weight = float(dcfg.get("feat_weight", 0.0))
        self._feat_last = 0.0
        self.adapter = None
        self._student_middle_capture: torch.Tensor | None = None
        if self.feat_weight > 0:
            if teacher_path is None:
                raise ValueError(
                    "distill.feat_weight is set without a teacher; the "
                    "feature term compares against the TEACHER's latent_pre "
                    "and has nothing to compare to without one.")
            from src.models.feature_adapter import FeatureAdapter
            # Architecture lives under cfg["model"] in the resolved config
            # (build_config() in train.py merges geometry + norm there),
            # NOT cfg["arch"] — that key only exists in the raw per-arm YAML
            # before _apply_yaml_overrides folds it in.
            model_cfg = cfg.get("model", {})
            width = model_cfg.get("width")
            enc_blk_nums = model_cfg.get("enc_blk_nums")
            if not width or not enc_blk_nums:
                raise ValueError(
                    "distill.feat_weight requires model.width and "
                    "model.enc_blk_nums to compute the student's middle_blks "
                    "channel count and downsample depth.")
            student_channels = width * (2 ** len(enc_blk_nums))
            student_downsamples = len(enc_blk_nums)
            # AdaIR's fixed, released architecture: dim=48, 3 downsamples to
            # `self.latent` — not derived from config, because the teacher's
            # architecture is frozen and never varies across arms.
            teacher_channels, teacher_downsamples = 384, 3
            scale_factor = (2 ** student_downsamples) / (2 ** teacher_downsamples)
            self.adapter = FeatureAdapter(
                in_channels=student_channels, out_channels=teacher_channels,
                scale_factor=scale_factor).to(device)
            self._middle_hook_handle = self.model.middle_blks.register_forward_hook(
                self._capture_student_middle)
            self.log.info(
                f"feature KD: student middle_blks {student_channels}ch "
                f"@ 1/{2**student_downsamples} -> teacher latent_pre "
                f"{teacher_channels}ch @ 1/{2**teacher_downsamples} "
                f"(scale_factor={scale_factor}) | feat weight {self.feat_weight}")

        # Degradation-conditioning auxiliary loss (kd_feature_multitask — see
        # reports/kd_feature_multitask/plan.md): cross-entropy between the
        # student's OWN DegradationHead prediction and ground-truth task id.
        # `_provenance["task"]` already flows through the multi-task loader
        # unused (see the loop below) — no new data-pipeline work. Additive
        # to the existing losses, never a replacement; zero weight = absent,
        # same discipline as feat_weight/freq_weight.
        self.aux_weight = float(dcfg.get("aux_weight", 0.0))
        self._aux_last = 0.0
        if self.aux_weight > 0 and getattr(self.model, "degradation_head", None) is None:
            raise ValueError(
                "distill.aux_weight is set but the model was not built with "
                "use_degradation_head=True; there is no DegradationHead to "
                "train against it.")

        # Optional MLflow logging -- best-effort, see src/utils/tracking.py for
        # why every call there is wrapped and can never fail a training run.
        # git_commit.txt / seed.txt already exist: create_run_dir() writes them
        # before the Trainer is constructed.
        from src.utils.tracking import RunTracker
        self.tracker = RunTracker(cfg, self.run_dir, self.log)
        commit_f = self.run_dir / "git_commit.txt"
        seed_f = self.run_dir / "seed.txt"
        self.tracker.log_start(
            cfg,
            commit_f.read_text(encoding="utf-8").strip() if commit_f.exists() else None,
            int(seed_f.read_text(encoding="utf-8").strip()) if seed_f.exists() else None)

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
        if self.adapter is not None:
            # The adapter is trained jointly with the student via the feature
            # KD loss and is training-time only — never part of the exported
            # student graph (see src/models/feature_adapter.py), so it needs
            # to be in the optimizer but nowhere else downstream of this.
            adapter_group = {"params": self.adapter.parameters(), "weight_decay": wd}
            params = (params + [adapter_group]) if isinstance(params, list) \
                else [{"params": params, "weight_decay": wd}, adapter_group]
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
        """Evaluate EMA weights through the LOCKED harness.

        Returns an empty dict when neither ``val_tasks`` nor ``val_root`` was
        supplied. Note that validation always runs on the final iteration
        regardless of ``val_every``, so this path is reachable in any run
        configured without a validation set — it is warned about rather than
        allowed to raise deep inside the training loop.
        """
        if self.val_tasks is not None:
            return self._validate_multitask()
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

    @torch.no_grad()
    def _validate_multitask(self) -> dict[str, float]:
        """Evaluate EMA weights on ALL configured tasks, one held-out set per
        task (kd_feature_multitask — see reports/kd_feature_multitask/plan.md,
        section 4: the B0V2 eval gap). Before this, a mixed_task run's
        ``eval`` block never set ``val_task``, so ``validate()`` silently ran
        the single-task denoise branch above against BSD68 only — the
        completed B0V2 baseline (300k iters) has real denoise numbers and NO
        dehaze/derain numbers at all.

        ``psnr``/``ssim`` are the mean across tasks, so history.json keeps the
        same shape (one scalar `best_psnr` to track) as every single-task run;
        ``psnr_<task>``/``ssim_<task>`` carry the per-task breakdown that was
        previously missing.
        """
        backup = self.ema.copy_to(self.model)
        self.model.eval()
        try:
            results: dict[str, float] = {}
            per_task_psnr, per_task_ssim = [], []
            for task, root in self.val_tasks.items():
                if task == "denoise":
                    # Same 3-sigma sweep as the single-task branch, just
                    # against this task's own root instead of self.val_root.
                    sigma_psnr, sigma_ssim = [], []
                    for sigma in (15, 25, 50):
                        ds = build_dataset("denoise", root, sigma=sigma,
                                           seed_mode="filename")
                        res = evaluate(self.model, iter(ds), name=f"denoise_s{sigma}",
                                       config=ADAIR_DEFAULT, device=self.device,
                                       keep_per_image=False)
                        results[f"psnr_denoise_s{sigma}"] = res.psnr
                        results[f"ssim_denoise_s{sigma}"] = res.ssim
                        sigma_psnr.append(res.psnr)
                        sigma_ssim.append(res.ssim)
                    task_psnr = sum(sigma_psnr) / len(sigma_psnr)
                    task_ssim = sum(sigma_ssim) / len(sigma_ssim)
                else:
                    ds = build_dataset(task, root)
                    res = evaluate(self.model, iter(ds), name=task,
                                   config=ADAIR_DEFAULT, device=self.device,
                                   keep_per_image=False)
                    task_psnr, task_ssim = res.psnr, res.ssim
                results[f"psnr_{task}"] = task_psnr
                results[f"ssim_{task}"] = task_ssim
                per_task_psnr.append(task_psnr)
                per_task_ssim.append(task_ssim)
            results["psnr"] = sum(per_task_psnr) / len(per_task_psnr)
            results["ssim"] = sum(per_task_ssim) / len(per_task_ssim)
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
                    if self.aux_weight > 0:
                        # Purely a function of the STUDENT's own forward pass
                        # (`last_degradation_logits`, set as a side effect of
                        # `self.model(degraded)` above) — no teacher involved,
                        # so this runs regardless of whether a teacher is
                        # configured. Ground truth is `_provenance["task"]`,
                        # a per-sample LongTensor from the multi-task loader's
                        # default collation (see the loop comment below).
                        if self.model.last_degradation_logits is None:
                            raise RuntimeError(
                                "distill.aux_weight > 0 but "
                                "last_degradation_logits is None — the "
                                "model's DegradationHead never fired this "
                                "step")
                        task_ids = _provenance["task"].to(
                            self.device, non_blocking=True)
                        aux = F.cross_entropy(
                            self.model.last_degradation_logits.float(),
                            task_ids)
                        loss = loss + self.aux_weight * aux
                        self._aux_last = float(aux)
                    if self.teacher is not None:
                        # Response distillation: one extra term, matching the
                        # teacher's OUTPUT on the same input. No hooks, no
                        # adapters, no feature matching -- the student never
                        # learns to compute frequencies, only to reproduce what
                        # doing so produced (F7).
                        # OUTSIDE autocast, in fp32. AdaIR's FreModule takes an
                        # FFT, and aten::fft_fft2 has no bfloat16 kernel -- the
                        # same frequency-domain machinery that makes the teacher
                        # undeployable (F7) also makes it unable to run in the
                        # student's training precision. The target is a fixed
                        # quantity anyway, so computing it at full precision
                        # costs only throughput.
                        with torch.no_grad(), torch.autocast("cuda", enabled=False):
                            if self.adapter is not None:
                                # Single teacher pass produces BOTH the
                                # response target and latent_pre — no second
                                # forward needed for the feature term.
                                soft, teacher_latent = self.teacher.forward_with_latent(
                                    degraded.float())
                            else:
                                soft = self.teacher(degraded.float())
                        kd = self.criterion(pred.float(), soft.float())
                        loss = loss + self.kd_weight * kd
                        self._kd_last = float(kd)
                        if self.freq_weight > 0:
                            # Also outside autocast: torch.fft has no bfloat16
                            # kernel, the same limitation as the teacher itself.
                            with torch.autocast("cuda", enabled=False):
                                fq = spectrum_loss(pred.float(), soft.float(),
                                                   mode=self.freq_mode)
                            loss = loss + self.freq_weight * fq
                            self._freq_last = float(fq)
                        if self.adapter is not None:
                            # Feature-level KD on the teacher's `latent_pre`
                            # bottleneck (kd_feature experiment, see
                            # reports/kd_feature/plan.md), not final-output
                            # pixels — TEST05.5 (teacher-experiments) found
                            # this representation, not the frequency pathway,
                            # is the well-supported distillation signal.
                            # `_student_middle_capture` was written by the
                            # forward hook during `self.model(degraded)`
                            # above, inside the SAME bfloat16 autocast region
                            # as `pred` — cast to float32 here to compare
                            # against the teacher's fp32 latent_pre, the same
                            # precision-matching pattern as the response and
                            # frequency terms.
                            if self._student_middle_capture is None:
                                raise RuntimeError(
                                    "feat_weight > 0 but the middle_blks hook "
                                    "never fired this step — the student "
                                    "forward pass did not call middle_blks")
                            with torch.autocast("cuda", enabled=False):
                                adapted = self.adapter.match_target(
                                    self._student_middle_capture.float(),
                                    teacher_latent.float())
                                feat = torch.abs(
                                    adapted - teacher_latent.float()).mean()
                            loss = loss + self.feat_weight * feat
                            self._feat_last = float(feat)
                            self._student_middle_capture = None

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
                    # tracker.finish() is NOT called here: metrics.json does not
                    # exist yet (train.py writes it after train() returns), so
                    # the artifact upload would miss it. train.py finishes the
                    # tracker on both the normal and diverged path, once, after
                    # write_metrics.
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
                        # Raw (unweighted) feature-KD term — logged so its
                        # scale relative to the pixel/response terms can
                        # actually be checked, rather than only seeing the
                        # combined `loss` and guessing which term dominates.
                        **({"feat_last": self._feat_last} if self.feat_weight > 0 else {}),
                        # Raw (unweighted) auxiliary degradation-classification
                        # CE — logged so it can be checked to actually decrease
                        # (a stuck value would mean the head/FiLM aren't
                        # learning despite the smoke test showing gradient
                        # flow in isolation).
                        **({"aux_last": self._aux_last} if self.aux_weight > 0 else {}),
                    }
                    self.state.history.append(row)
                    self.tracker.log_metrics(row, step=it)
                    self.log.info(
                        f"it {it:6d}  loss {row['loss']:.5f}  psnr {metrics['psnr']:.3f}  "
                        f"ssim {metrics['ssim']:.4f}  gnorm {gn:.3f}  "
                        f"maxgn {max_gnorm:.3f}  "
                        + (f"feat {row['feat_last']:.5f}  " if "feat_last" in row else "")
                        + (f"aux {row['aux_last']:.5f}  " if "aux_last" in row else "")
                        + f"clip {clip_hits}/{steps_since} ({row['clip_rate']:.1%})  "
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
