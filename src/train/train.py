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
from src.utils.config import REPO_ROOT, load_paths
from src.utils.run_dir import create_run_dir, write_metrics
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
ARM_GEOMETRY = {"M-A": W16_SIDD, "M-F": W16_SIDD}


def build_config(arm: str, iters: int, batch_size: int, lr: float,
                 patch_size: int = 128) -> dict:
    spec = ARMS[arm]
    geometry = ARM_GEOMETRY.get(arm, W16_B8)
    return {
        "arm": arm,
        "description": spec["desc"],
        "model": {**geometry, **spec["norm"]},
        "data": {"patch_size": patch_size, "batch_size": batch_size,
                 "sigmas": [15, 25, 50]},
        "optim": {"name": "adamw", "lr": lr * spec.get("lr_scale", 1.0),
                  "weight_decay": 1e-4, "betas": [0.9, 0.9],
                  "grad_clip": spec.get("grad_clip")},
        "schedule": {"total_iters": iters,
                     "warmup_iters": int(2000 * spec.get("warmup_scale", 1.0)),
                     "min_lr": 1e-6},
        "train": {"ema_decay": 0.999, "amp": True,
                  "val_every": 2000, "ckpt_every": 2000},
        "loss": {"name": "charbonnier", "eps": 1e-3},
        "residual_init": spec.get("residual_init", 0.0),
    }


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

    loader = build_train_loader(
        [data_root / "Train" / "Denoise"],
        batch_size=args.batch_size, patch_size=args.patch_size,
        sigmas=tuple(cfg["data"]["sigmas"]),
        num_workers=args.num_workers, seed=args.seed,
        length=args.iters * args.batch_size)

    model = build_model(cfg)
    trainer = Trainer(model, loader, cfg, run_dir, device=args.device,
                      val_root=data_root / "test" / "denoise" / "bsd68")
    if args.resume:
        trainer.load_checkpoint(Path(args.resume))

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
