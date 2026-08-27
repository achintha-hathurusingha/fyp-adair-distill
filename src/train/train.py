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

from src.data.build import (build_multitask_loader, build_train_loader,
                            resolve_task_sources)
import src.models.norms as norms
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
    "B0": {"norm": {"norm_type": "layernorm2d",
                    "full_res_norm_type": "affine_clamp", "clamp_bound": 8.0},
           "config": "configs/train/b0_baseline.yaml",
           "desc": "B0 baseline: locked N-F on w16_sidd, GT only, no teacher"},
    # Q-A control for the B0 divergence: identical to B0 in every respect except
    # LayerNorm2d at EVERY stage. Distinguishes "N-F caused the growth" from
    # "NAFNet at this depth grows regardless of per-block normalization".
    # Fix-C integration test: B0 with a magnitude clamp at the full-resolution
    # stages. See findings F9.
    "B0-FIXC": {"norm": {"norm_type": "layernorm2d",
                         "full_res_norm_type": "affine_clamp",
                         "clamp_bound": 8.0},
                "config": "configs/train/b0_fixc.yaml",
                "desc": "B0 + Fix-C: affine_clamp(8.0) at full resolution"},
    "B0-QA": {"norm": {"norm_type": "layernorm2d"},
              "config": "configs/train/b0_qa_control.yaml",
              "desc": "Q-A control: full LayerNorm2d on w16_sidd, B0 schedule"},
    # B0-v2 — the ALL-IN-ONE baseline, which is the one the protocol actually
    # asks for. Same locked architecture as B0-denoise so the two are directly
    # comparable, plus the enc3 clamp insurance and continuous sigma (F10).
    # B0-denoise is retained as the single-task control, not as this (F11).
    "B0V2": {"norm": {"norm_type": "layernorm2d",
                      "full_res_norm_type": "affine_clamp", "clamp_bound": 8.0,
                      # enc3 insurance (F10). Listed here as well as in the YAML
                      # because the drift guard compares the two and refuses to
                      # merge them — which is what caught this being absent.
                      "enc_clamp_stages": [3], "deep_clamp_bound": 32.0},
             "config": "configs/train/b0v2_multitask.yaml",
             "desc": "B0-v2: 3-degradation all-in-one baseline, GT only"},
    # Demo arms: single-task dehazing, GT-only and its distilled counterpart.
    # Same locked architecture as B0-v2 so the only difference is the loss.
    "M-DEHAZE": {"norm": {"norm_type": "layernorm2d",
                          "full_res_norm_type": "affine_clamp", "clamp_bound": 8.0,
                          "enc_clamp_stages": [3], "deep_clamp_bound": 32.0},
                 "config": "configs/train/m_dehaze_baseline.yaml",
                 "desc": "M on dehaze, ground truth only (gap demo)"},
    "M-DEHAZE-KD": {"norm": {"norm_type": "layernorm2d",
                             "full_res_norm_type": "affine_clamp", "clamp_bound": 8.0,
                             "enc_clamp_stages": [3], "deep_clamp_bound": 32.0},
                    "config": "configs/train/m_dehaze_kd.yaml",
                    "desc": "M on dehaze, GT + response KD from AdaIR (gap demo)"},
    # student_arch experiment (reports/student_arch/findings.md): architecture
    # variants against M-DEHAZE's own already-measured GT-only baseline
    # (32.8898dB, 3-seed mean) -- no teacher, no distill block, one arch key
    # changed each. "norm" here must match the YAML's `arch:` section
    # key-for-key or _apply_yaml_overrides's drift guard raises.
    "M-DEHAZE-ECA": {"norm": {"norm_type": "layernorm2d",
                              "full_res_norm_type": "affine_clamp", "clamp_bound": 8.0,
                              "enc_clamp_stages": [3], "deep_clamp_bound": 32.0,
                              "attn_type": "eca"},
                     "config": "configs/train/m_dehaze_eca.yaml",
                     "desc": "M on dehaze, GT only, SCA -> ECA channel attention"},
    "M-DEHAZE-GROUPNORM": {"norm": {"norm_type": "groupnorm",
                                    "full_res_norm_type": "affine_clamp", "clamp_bound": 8.0,
                                    "enc_clamp_stages": [3], "deep_clamp_bound": 32.0},
                           "config": "configs/train/m_dehaze_groupnorm.yaml",
                           "desc": "M on dehaze, GT only, LayerNorm2d -> GroupNorm"},
    # Third arm of the KD ablation: response term PLUS a frequency-domain term.
    # Differs from M-DEHAZE-KD by exactly two config keys.
    "M-DEHAZE-KD-FREQ": {"norm": {"norm_type": "layernorm2d",
                                  "full_res_norm_type": "affine_clamp",
                                  "clamp_bound": 8.0,
                                  "enc_clamp_stages": [3],
                                  "deep_clamp_bound": 32.0},
                         "config": "configs/train/m_dehaze_kd_freq.yaml",
                         "desc": "M on dehaze, GT + response KD + spectrum KD"},
    # kd_feature experiment (reports/kd_feature/plan.md): response term PLUS
    # a feature-level term on the teacher's internal `latent_pre` bottleneck,
    # instead of kd_freq's final-output spectrum term. Motivated by kd_freq's
    # own early result (near-zero, oscillating delta) and TEST05.5
    # (teacher-experiments), which found latent_pre — not the frequency
    # pathway — is the well-supported distillation signal. Phase A: plain L1
    # feature match, isolated from kd_freq exactly like kd_freq isolated the
    # frequency term from plain response KD.
    "M-DEHAZE-KD-FEAT": {"norm": {"norm_type": "layernorm2d",
                                  "full_res_norm_type": "affine_clamp",
                                  "clamp_bound": 8.0,
                                  "enc_clamp_stages": [3],
                                  "deep_clamp_bound": 32.0},
                         "config": "configs/train/m_dehaze_kd_feat.yaml",
                         "desc": "M on dehaze, GT + response KD + latent_pre feature KD"},
    # Same methodology on derain, so the two tasks are directly comparable.
    # Much less gap available: the specialist comparison measured +0.25 dB on
    # derain against +1.6 dB on dehaze.
    "M-DERAIN": {"norm": {"norm_type": "layernorm2d",
                          "full_res_norm_type": "affine_clamp", "clamp_bound": 8.0,
                          "enc_clamp_stages": [3], "deep_clamp_bound": 32.0},
                 "config": "configs/train/m_derain_baseline.yaml",
                 "desc": "M on derain, ground truth only (gap demo)"},
    "M-DERAIN-KD": {"norm": {"norm_type": "layernorm2d",
                             "full_res_norm_type": "affine_clamp", "clamp_bound": 8.0,
                             "enc_clamp_stages": [3], "deep_clamp_bound": 32.0},
                    "config": "configs/train/m_derain_kd.yaml",
                    "desc": "M on derain, GT + response KD from AdaIR (gap demo)"},
    # Item 3: KD-weight robustness check. Same config as M-DEHAZE-KD in every
    # respect except distill.weight -- confirms whether the 3-seed result
    # (weight=1.0, untuned) is sensitive to that choice across a 4x range.
    "M-DEHAZE-KD-W05": {"norm": {"norm_type": "layernorm2d",
                                 "full_res_norm_type": "affine_clamp",
                                 "clamp_bound": 8.0,
                                 "enc_clamp_stages": [3], "deep_clamp_bound": 32.0},
                        "config": "configs/train/m_dehaze_kd_w05.yaml",
                        "desc": "M on dehaze, GT + response KD, weight=0.5"},
    "M-DEHAZE-KD-W20": {"norm": {"norm_type": "layernorm2d",
                                 "full_res_norm_type": "affine_clamp",
                                 "clamp_bound": 8.0,
                                 "enc_clamp_stages": [3], "deep_clamp_bound": 32.0},
                        "config": "configs/train/m_dehaze_kd_w20.yaml",
                        "desc": "M on dehaze, GT + response KD, weight=2.0"},
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
ARM_GEOMETRY = {"M-A": W16_SIDD, "M-F": W16_SIDD, "B0": W16_SIDD,
                "B0-QA": W16_SIDD, "B0-FIXC": W16_SIDD, "B0V2": W16_SIDD,
                "M-DEHAZE": W16_SIDD, "M-DEHAZE-KD": W16_SIDD,
                "M-DERAIN": W16_SIDD, "M-DERAIN-KD": W16_SIDD,
                "M-DEHAZE-KD-FREQ": W16_SIDD, "M-DEHAZE-KD-FEAT": W16_SIDD,
                "M-DEHAZE-KD-W05": W16_SIDD, "M-DEHAZE-KD-W20": W16_SIDD,
                "M-DEHAZE-ECA": W16_SIDD, "M-DEHAZE-GROUPNORM": W16_SIDD}


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

    # Every section the YAML may carry. `eval` and `distill` were missing here
    # while both were read downstream, so the values existed in the reviewed
    # file, were referenced in the code, and still never met: the dehaze run
    # validated on BSD68 denoising, and the KD run would have loaded no teacher
    # at all and silently trained a duplicate baseline. A section absent from
    # the base config is created rather than skipped -- `eval` and `distill`
    # have no defaults, which is exactly why they were dropped.
    for section in ("data", "optim", "schedule", "train", "loss", "eval", "distill"):
        if section in yml:
            cfg.setdefault(section, {}).update(yml[section])

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
    # DELIBERATE exception to YAML authority. num_workers is a machine-specific
    # throughput knob, not a scientific parameter: the dataset seeds its RNG
    # from (base_seed, index) rather than worker id, so the sample stream is
    # worker-count-independent. Verified, not assumed — determinism_check gives
    # a bit-identical fingerprint at 6 and 12 workers. Everything that changes
    # RESULTS still comes from the reviewed YAML and cannot be overridden.
    ap.add_argument("--num-workers", type=int, default=None,
                    help="override dataloader workers (throughput only; proven "
                         "not to change results)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-root", default="runs/1p5b")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--smoke", type=int, default=0, metavar="N",
                    help="pre-launch check: run N iterations on the REAL "
                         "resolved config, validating and checkpointing, then "
                         "stop. Everything except the run length is untouched.")
    ap.add_argument("--resume-reason", default="",
                    help="why this run was resumed; recorded in resumes.jsonl")
    args = ap.parse_args()

    seed_everything(args.seed)
    cfg = build_config(args.arm, args.iters, args.batch_size, args.lr,
                       args.patch_size)
    if args.num_workers is not None:
        # Must happen BEFORE create_run_dir: the run directory has to record the
        # value actually used, not the one the YAML suggested.
        cfg["data"]["num_workers"] = args.num_workers
    if args.smoke:
        # Shrink ONLY the length. A smoke test that quietly altered batch size,
        # normalization or the optimiser would validate a config nobody is about
        # to run. Validation and checkpointing stay on, because "does it train"
        # is a weaker question than "does the whole loop work end to end".
        cfg["schedule"]["total_iters"] = args.smoke
        cfg["schedule"]["warmup_iters"] = max(1, args.smoke // 8)
        cfg["train"]["val_every"] = args.smoke
        cfg["train"]["ckpt_every"] = args.smoke
        cfg["smoke_test"] = args.smoke

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

    # Clamp telemetry, from config rather than a source edit. This was a local
    # uncommitted change to norms.py on the training host for all of B0-denoise,
    # which means a fresh checkout would have recorded nothing — and the
    # engagement drift it captures is the whole of finding F12. A run's own
    # config.yaml now states whether its clamp diagnostics are real or absent.
    #
    # Costs one .abs().max() and one comparison per clamped norm per forward, so
    # it is off by default and on for the runs that are being watched.
    norms.TRACK_CLAMP_ENGAGEMENT = bool(
        cfg["train"].get("track_clamp_engagement", False))

    # Read from the RESOLVED config, not from args — for a YAML-backed arm the
    # file is authoritative and the CLI defaults no longer describe the run.
    micro_bs = cfg["data"]["batch_size"]
    accum = cfg["train"]["accum_steps"]
    # The loader is consumed one MICRO-batch at a time, and an optimizer step
    # eats `accum` of them, so the sample budget scales with both.
    length = cfg["schedule"]["total_iters"] * accum * micro_bs
    workers = (args.num_workers if args.num_workers is not None
               else cfg["data"]["num_workers"])
    common = dict(batch_size=micro_bs, patch_size=cfg["data"]["patch_size"],
                  sigmas=tuple(cfg["data"]["sigmas"]), num_workers=workers,
                  seed=args.seed, length=length,
                  cache_budget_gb=cfg["data"]["cache_budget_gb"])

    # `mixed_task` decides the training SCOPE, and until F11 no code read it:
    # the loader was hardcoded to denoise while the key sat in the config saying
    # otherwise. It is now load-bearing, and `tasks` is mandatory when it is on —
    # falling back to denoise is exactly the failure that produced B0-denoise.
    if cfg["data"]["mixed_task"]:
        tasks = cfg["data"].get("tasks")
        if not tasks:
            raise ValueError(
                "data.mixed_task is true but data.tasks is missing or empty. "
                "List the task roots explicitly (relative to data_root); there "
                "is no default -- see findings F11.")
        # Continuous noise sampling is the F10 fix and applies only to the
        # multi-task path — B0-denoise's discrete {15,25,50} stays exactly as
        # trained so the two remain comparable on the protocol's own sigmas.
        sigma_range = cfg["data"].get("sigma_range")
        loader = build_multitask_loader(
            resolve_task_sources(tasks, data_root),
            sigma_range=tuple(sigma_range) if sigma_range else None,
            clean_prob=cfg["data"].get("clean_prob", 0.0), **common)
    else:
        loader = build_train_loader([data_root / "Train" / "Denoise"], **common)

    # Validation set from config, defaulting to BSD68 so every existing arm
    # is unchanged. A single-task arm points this at its own held-out set.
    val_rel = (cfg.get("eval") or {}).get("val_root")
    val_root = (data_root / val_rel) if val_rel else (
        data_root / "test" / "denoise" / "bsd68")
    if not val_root.exists():
        raise FileNotFoundError(
            f"validation set not found: {val_root}. Set eval.val_root in the "
            "config, or create it — a run with no validation produces no "
            "convergence evidence.")

    model = build_model(cfg)
    trainer = Trainer(model, loader, cfg, run_dir, device=args.device,
                      val_root=val_root)
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

    # After write_metrics, not inside Trainer.train(): metrics.json does not
    # exist until here, and finish() attaches it as an artifact. Runs both the
    # normal-completion and diverged path, since both return through here.
    trainer.tracker.finish({"diverged": bool(final.get("diverged"))})


if __name__ == "__main__":
    main()
