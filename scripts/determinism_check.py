"""Train a few steps from a fixed seed and fingerprint the result.

Run this TWICE with identical arguments on the same machine. Same seed, same
code, same hardware must give the same fingerprint. If it does not, the machine
is computing different answers from identical inputs — which is what silent
memory corruption looks like, and is far more dangerous than a crash.

    python scripts/determinism_check.py --iters 60 --seed 0 --tag run1
    python scripts/determinism_check.py --iters 60 --seed 0 --tag run2

ALWAYS establish a control on known-good hardware first. `seed_everything` sets
cudnn.deterministic but does NOT call torch.use_deterministic_algorithms, so
bit-exactness is not guaranteed by construction — a mismatch is only evidence
about the HARDWARE once a trusted machine has been shown to match itself.

On devon this must be run under `taskset -c 16-31` (E-cores); its P-cores are
degraded. See memory: devon-hardware-instability.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from src.data.build import build_train_loader
from src.train.train import build_config, build_model
from src.train.trainer import Trainer
from src.utils.config import REPO_ROOT, load_paths
from src.utils.seeding import seed_everything


def fingerprint(tensors: dict[str, torch.Tensor]) -> str:
    """SHA-256 over every parameter, in a fixed order.

    Hashing raw bytes rather than comparing floats: the question is whether the
    machine produced *identical* results, not *close* ones.
    """
    h = hashlib.sha256()
    for name in sorted(tensors):
        t = tensors[name].detach().to("cpu").contiguous()
        h.update(name.encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", required=True, help="label for this run's output")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-root", default="runs/determinism")
    ap.add_argument("--num-workers", type=int, default=None,
                    help="override dataloader workers; results MUST NOT depend "
                         "on this (the dataset seeds per-index, not per-worker)")
    args = ap.parse_args()

    seed_everything(args.seed)
    cfg = build_config("B0", 0, 0, 0.0, 0)

    micro_bs = cfg["data"]["batch_size"]
    accum = cfg["train"]["accum_steps"]
    # Shrink only the length; the rest is the real B0 configuration.
    cfg["schedule"]["total_iters"] = args.iters
    cfg["schedule"]["warmup_iters"] = max(1, args.iters // 8)
    cfg["train"]["val_every"] = 10 ** 9        # no validation: pure training path
    cfg["train"]["ckpt_every"] = 10 ** 9

    paths = load_paths()
    data_root = Path(paths["data_root"])
    if not data_root.is_absolute():
        data_root = REPO_ROOT / data_root

    loader = build_train_loader(
        [data_root / "Train" / "Denoise"],
        batch_size=micro_bs, patch_size=cfg["data"]["patch_size"],
        sigmas=tuple(cfg["data"]["sigmas"]),
        num_workers=(args.num_workers if args.num_workers is not None
                     else cfg["data"]["num_workers"]),
        seed=args.seed,
        length=args.iters * accum * micro_bs,
        cache_budget_gb=cfg["data"]["cache_budget_gb"])

    model = build_model(cfg)
    out_dir = REPO_ROOT / args.out_root / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(model, loader, cfg, out_dir, device=args.device)
    state = trainer.train()

    model_fp = fingerprint(dict(model.state_dict()))
    ema_fp = fingerprint(dict(trainer.ema.shadow)) if hasattr(trainer.ema, "shadow") else "n/a"
    losses = [round(float(r["loss"]), 12) for r in state.history]

    result = {
        "tag": args.tag,
        "seed": args.seed,
        "iters": args.iters,
        "device": args.device,
        "num_workers": (args.num_workers if args.num_workers is not None
                        else cfg["data"]["num_workers"]),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu",
        "model_sha256": model_fp,
        "ema_sha256": ema_fp,
        "losses": losses,
    }
    (out_dir / "fingerprint.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
