"""Training entry point for the Task 1.5b normalization ablation.

    python -m src.train.train --arm Q-A --iters 50000
    python -m src.train.train --arm Q-E --resume runs/1p5b/Q-E/last.pth

Every arm is identical except for the normalization (and, for the escalation
ladder, the optimiser settings). Data, augmentation, seed and schedule are fixed
so the only free variable is the thing under test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.data.build import build_train_loader
from src.models.nafnet import NAFNet
from src.train.trainer import Trainer
from src.utils.config import REPO_ROOT, load_paths, load_yaml
from src.utils.run_dir import create_run_dir, record_resume, write_metrics
from src.utils.seeding import seed_everything

#: The arms. Normalization is the variable; everything else is held fixed.
ARMS: dict[str, dict] = {
    "Q-A": {"norm": {"norm_type": "layernorm2d"},
            "desc": "LayerNorm2d everywhere (reference, 2.510 ms)"},
    "Q-F": {"norm": {"norm_type": "layernorm2d", "full_res_norm_type": "affine"},
            "desc": "affine at full resolution, LayerNorm deeper (1.60x)"},
    "Q-E": {"norm": {"norm_type": "affine"},
            "desc": "affine everywhere (floor, 2.35x)"},
    # Escalation ladder — identical architecture to Q-E, different optimisation.
    "Q-E1": {"norm": {"norm_type": "affine"}, "lr_scale": 0.5, "warmup_scale": 2.0,
             "desc": "Q-E + half LR + extended warmup"},
    "Q-E2": {"norm": {"norm_type": "affine"}, "lr_scale": 0.5, "warmup_scale": 2.0,
             "grad_clip": 1.0, "desc": "Q-E' + gradient clipping (norm 1.0)"},
    "Q-E3": {"norm": {"norm_type": "affine"}, "lr_scale": 0.5, "warmup_scale": 2.0,
             "grad_clip": 1.0, "residual_init": 0.1,
             "desc": "Q-E'' + residual scaling init 0.1"},
    # M spot-check — the norm ablation ran on S (w16_b8), but M (w16_sidd) is
    # the config the Phase 02 grid runs on AND the one carrying the most
    # full-resolution normalization, so a quality cost from N-F surfaces here
    # first. Short run: trend, not convergence.
    "M-A": {"norm": {"norm_type": "layernorm2d"},
            "desc": "M spot-check: LayerNorm2d everywhere on w16_sidd"},
    "M-F": {"norm": {"norm_type": "layernorm2d", "full_res_norm_type": "affine"},
            "desc": "M spot-check: affine at full resolution on w16_sidd"},
    # B0 — the reference baseline. LOCKED architecture (N-F on the M arm),
    # ground truth only, no teacher. Effective batch 32 via accumulation, and a
    # loose grad clip as tail insurance. See configs/train/b0_baseline.yaml.
    "B0": {"norm": {"norm_type": "layernorm2d", "full_res_norm_type": "affine"},
           "config": "configs/train/b0_baseline.yaml",
           "desc": "B0 baseline: locked N-F on w16_sidd, GT only, no teacher"},
}

#: w16_b8 — the config on which every norm variant is already profiled (arm S).
W16_B8 = dict(width=16, enc_blk_nums=[1, 1, 1, 8], middle_blk_num=2,
              dec_blk_nums=[1, 1, 1, 1])

#: w16_sidd — the M arm after family re-selection on N-F latency. It carries
#: the most full-resolution normalization of any config, which is precisely why
#: it gained most from N-F — and precisely why a quality cost from N-F would
#: show up here first. Hence the M spot-check.
W16_SIDD = dict(width=16, enc_blk_nums=[2, 2, 4, 8], middle_blk_num=12,
                dec_blk_nums=[2, 2, 2, 2])

#: Arms that override the default geometry.
ARM_GEOMETRY = {"M-A": W16_SIDD, "M-F": W16_SIDD, "B0": W16_SIDD}


def _apply_yaml_overrides(cfg: dict, spec: dict, arm: str) -> dict:
    """Fold an arm's reviewed YAML into the resolved config.

    The ablation arms are defined entirely by the ARMS table, but B0 is a
    reviewed, multi-day run whose settings live in a YAML that a supervisor
    reads. Duplicating those constants here is how the two silently drift: the
    run directory would record what ARMS says while the reviewed file says
    something else. So the YAML is the authority for B0's schedule/optim/train
    settings, and any architectural disagreement is an error, not a merge.
    """
    yml = load_yaml(spec["config"])

    arch = yml.get("arch", {})
    for key, want in arch.items():
        got = cfg["model"].get(key)
        if got != want:
            raise ValueError(
                f"arm {arm}: architecture drift between {spec['config']} and "
                f"src/train/train.py — '{key}' is {want!r} in the YAML but "
                f"{got!r} in the resolved config. Fix one; do not merge.")

    for section in ("data", "optim", "schedule", "train", "loss"):
        cfg[section].update(yml.get(section, {}))

    # CLI --iters / --batch-size are ablation conveniences. For a reviewed run
    # the YAML wins, so an accidental flag cannot quietly shorten B0.
    cfg["config_source"] = spec["config"]
    return cfg


def build_config(arm: str, iters: int, batch_size: int, lr: float,
                 patch_size: int = 128) -> dict:
    spec = ARMS[arm]
    geometry = ARM_GEOMETRY.get(arm, W16_B8)
    cfg = {
        "arm": arm,
        "description": spec["desc"],
        "model": {**geometry, **spec["norm"]},
        # cache_budget_gb is PER WORKER; unbounded caching converges on
        # num_workers x the decoded training set and exhausts RAM (see
        # src/data/build.py). Arms with a YAML override it from there.
        "data": {"patch_size": patch_size, "batch_size": batch_size,
                 "sigmas": [15, 25, 50], "cache_budget_gb": 0.75},
        "optim": {"name": "adamw", "lr": lr * spec.get("lr_scale", 1.0),
                  "weight_decay": 1e-4, "betas": [0.9, 0.9],
                  "grad_clip": spec.get("grad_clip")},
        "schedule": {"total_iters": iters,
                     "warmup_iters": int(2000 * spec.get("warmup_scale", 1.0)),
                     "min_lr": 1e-6},
        "train": {"ema_decay": 0.999, "amp": True,
                  "accum_steps": spec.get("accum_steps", 1),
                  "val_every": 2000, "ckpt_every": 2000},
        "loss": {"name": "charbonnier", "eps": 1e-3},
        "residual_init": spec.get("residual_init", 0.0),
    }
    if "config" in spec:
        cfg = _apply_yaml_overrides(cfg, spec, arm)
    return cfg


def build_model(cfg: dict) -> NAFNet:
    model = NAFNet(**{k: v for k, v in cfg["model"].items()})
    init = cfg.get("residual_init", 0.0)
    if init:
        # Escalation rung Q-E''': non-zero residual scaling. The reference
        # initialises beta/gamma to zero (identity block); a small positive
        # value gives the residual branches signal from the first step.
        with torch.no_grad():
            for m in model.modules():
                if hasattr(m, "beta") and hasattr(m, "gamma"):
                    m.beta.fill_(init)
                    m.gamma.fill_(init)
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description="Task 1.5b norm ablation training.")
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--iters", type=int, default=50_000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--patch-size", type=int, default=128,
                    help="AdaIR trains at 128 (options.py:15); 256 exceeds 6GB at batch 32")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-root", default="runs/1p5b")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--resume-reason", default="",
                    help="why this run was resumed; recorded in resumes.jsonl")
    args = ap.parse_args()

    seed_everything(args.seed)
    cfg = build_config(args.arm, args.iters, args.batch_size, args.lr,
                       args.patch_size)

    paths = load_paths()
    data_root = Path(paths["data_root"])
    if not data_root.is_absolute():
        data_root = REPO_ROOT / data_root

    run_root = REPO_ROOT / args.out_root / args.arm
    if args.resume:
        run_dir = Path(args.resume).parent
    else:
        run_root.mkdir(parents=True, exist_ok=True)
        run_dir = create_run_dir(run_root, args.arm, config=cfg, seed=args.seed)

    # Read from the RESOLVED config, not from args — for a YAML-backed arm the
    # file is authoritative and the CLI defaults no longer describe the run.
    micro_bs = cfg["data"]["batch_size"]
    accum = cfg["train"]["accum_steps"]
    # The loader is consumed one MICRO-batch at a time, and an optimizer step
    # eats `accum` of them, so the sample budget scales with both.
    length = cfg["schedule"]["total_iters"] * accum * micro_bs
    loader = build_train_loader(
        [data_root / "Train" / "Denoise"],
        batch_size=micro_bs, patch_size=cfg["data"]["patch_size"],
        sigmas=tuple(cfg["data"]["sigmas"]),
        num_workers=cfg["data"].get("num_workers", args.num_workers),
        seed=args.seed, length=length,
        cache_budget_gb=cfg["data"]["cache_budget_gb"])

    model = build_model(cfg)
    trainer = Trainer(model, loader, cfg, run_dir, device=args.device,
                      val_root=data_root / "test" / "denoise" / "bsd68")
    if args.resume:
        trainer.load_checkpoint(Path(args.resume))
        # A resume reuses the run directory, so config.yaml and git_commit.txt
        # still describe the ORIGINAL launch. Record what is in force from here.
        record_resume(run_dir, cfg, iteration=trainer.state.iteration,
                      reason=args.resume_reason)

    trainer.log.info(f"arm {args.arm}: {cfg['description']}")
    trainer.log.info(f"params: {sum(p.numel() for p in model.parameters()):,}")
    state = trainer.train()

    final = state.history[-1] if state.history else {}
    write_metrics(run_dir, {
        "arm": args.arm,
        "description": cfg["description"],
        "iterations": state.iteration,
        "best_psnr": state.best_psnr,
        "final": final,
        "peak_vram_gb": final.get("peak_vram_gb", 0.0),
        "diverged": bool(final.get("diverged")),
    })
    print(json.dumps({"arm": args.arm, "best_psnr": state.best_psnr,
                      "iterations": state.iteration,
                      "peak_vram_gb": final.get("peak_vram_gb", 0.0)}, indent=2))


if __name__ == "__main__":
    main()
