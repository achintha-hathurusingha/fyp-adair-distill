"""Which task drives clamp engagement?

B0-v2's training log reports one engagement rate over a batch that is one third
denoise, one third derain, one third dehaze, so a task-specific effect is
invisible in it -- a 12% rate on one task and 0% on the others reads exactly
like 4% everywhere. F12's watch item is whether the multi-task mixture changes
the clamp's behaviour, and that question cannot be answered from the aggregate.

This feeds single-task batches through a trained checkpoint with the counters
on, resetting between tasks.

    python scripts/clamp_by_task.py <checkpoint.pth> [--batches 20]

Inference only, so it measures the forward-pass behaviour the deployed model
has, not the training-time dynamics. Those differ -- training sees augmented
128px crops and inference sees whole images -- which is why the numbers here are
compared against each other rather than against the training log.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

import src.models.norms as norms
from src.data.build import TASK_IDS, MultiTaskTrainDataset, resolve_task_sources
from src.models.norms import AffineClampNorm2d, LayerNorm2dClamp, reset_clamp_engagement
from src.train.train import build_model
from src.utils.config import REPO_ROOT, load_paths, load_yaml


def _stats(model) -> dict[str, float]:
    out = {}
    for prefix, cls in (("dec3/full-res", AffineClampNorm2d),
                        ("enc3/deep", LayerNorm2dClamp)):
        mods = [m for m in model.modules() if isinstance(m, cls)]
        fwd = sum(getattr(m, "forwards", 0) for m in mods)
        if not fwd:
            continue
        out[f"{prefix} engage"] = sum(getattr(m, "engaged", 0) for m in mods) / fwd
        out[f"{prefix} premax"] = max(
            (getattr(m, "max_preclamp", 0.0) for m in mods), default=0.0)
    return out


def _reset(model) -> None:
    for m in model.modules():
        if isinstance(m, (AffineClampNorm2d, LayerNorm2dClamp)):
            reset_clamp_engagement(m)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--batches", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg_path = args.config or args.checkpoint.parent / "config.yaml"
    cfg = load_yaml(cfg_path)
    data_root = Path(load_paths()["data_root"])
    if not data_root.is_absolute():
        data_root = REPO_ROOT / data_root

    model = build_model(cfg)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    for key in ("ema", "model", "state_dict"):
        if isinstance(state, dict) and key in state and isinstance(state[key], dict):
            state = state[key]
            break
    model.load_state_dict(state)
    model = model.to(args.device).eval()

    # Counters must be enabled BEFORE any forward, and this script is the only
    # thing running, so flipping the module flag here is safe.
    norms.TRACK_CLAMP_ENGAGEMENT = True

    sources = resolve_task_sources(cfg["data"]["tasks"], data_root)
    d = cfg["data"]
    print(f"checkpoint : {args.checkpoint}")
    print(f"{args.batches} batches x {args.batch_size} at {d['patch_size']}px, per task\n")
    print(f"{'task':<9} {'dec3 engage':>12} {'dec3 premax':>12} "
          f"{'enc3 engage':>12} {'enc3 premax':>12}")

    rows = {}
    for task in sources:
        # One task at a time: a single-task dataset makes every sample that task.
        ds = MultiTaskTrainDataset(
            {task: sources[task]}, sigmas=tuple(d["sigmas"]),
            sigma_range=tuple(d["sigma_range"]) if d.get("sigma_range") else None,
            clean_prob=d.get("clean_prob", 0.0), patch_size=d["patch_size"],
            base_seed=0, length=args.batches * args.batch_size,
            cache_budget_gb=0.5)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        _reset(model)
        with torch.no_grad():
            for deg, _clean, _meta in loader:
                model(deg.to(args.device))
        s = _stats(model)
        rows[task] = s
        print(f"{task:<9} {100 * s.get('dec3/full-res engage', 0):>11.3f}% "
              f"{s.get('dec3/full-res premax', 0):>12.4g} "
              f"{100 * s.get('enc3/deep engage', 0):>11.3f}% "
              f"{s.get('enc3/deep premax', 0):>12.4g}")

    dec = {t: r.get("dec3/full-res engage", 0) for t, r in rows.items()}
    hi, lo = max(dec, key=dec.get), min(dec, key=dec.get)
    print(f"\nhighest dec3 engagement: {hi} ({100 * dec[hi]:.3f}%), "
          f"lowest: {lo} ({100 * dec[lo]:.3f}%)")
    if all(r.get("enc3/deep engage", 0) == 0 for r in rows.values()):
        print("enc3 clamp did not engage on ANY task — inert, as F9/F12 expected")
    else:
        print("enc3 clamp ENGAGED — this is the F12 stop condition, investigate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
