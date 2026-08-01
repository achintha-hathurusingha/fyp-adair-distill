"""Per-stage gradient concentration — where does gradient mass sit?

The AGC diagnostic without adopting AGC: per-stage gradient-norm / parameter-norm
ratio, the quantity Adaptive Gradient Clipping would clip on. Run once on the
spike state (findings F9) it showed mass concentrating at `intro` and the
encoders, with `middle_blks` LOWER than every encoder stage — i.e. a symptom of
the `dec3` forward explosion propagating backwards, not an independent
middle-stage pathology.

    python scripts/agc_snapshot.py --weights <ckpt or spike dump> \\
                                   --batch <spike dump> [--arm B0-FIXC]

PRE-STAGED for a possible re-run on a LATER model state, to check whether the
concentration pattern has moved. The earlier ad-hoc version read both the
weights and the batch from a spike dump; a training checkpoint carries weights
but no batch, so the two sources are now separate arguments. `--batch` supplies
the pathological and healthy samples from any captured spike dump.

Running this does not endorse AGC. It locates where to look next.
"""
from __future__ import annotations

import argparse
import glob
from collections import defaultdict
from pathlib import Path

import torch

from src.losses.reconstruction import build_loss
from src.train.train import build_config, build_model

STAGES = ["intro", "encoders.0", "encoders.1", "encoders.2", "encoders.3",
          "middle_blks", "decoders.0", "decoders.1", "decoders.2",
          "decoders.3", "ending"]


def stage_of(name: str) -> str:
    for key in STAGES:
        if name.startswith(key):
            return key
    return "other"


def load_state(path_glob: str) -> tuple[dict, str]:
    """Accept either a spike dump or a training checkpoint."""
    path = sorted(glob.glob(path_glob))[-1]
    blob = torch.load(path, map_location="cpu", weights_only=False)
    state = blob.get("model", blob)
    it = blob.get("iteration", "?")
    return state, f"{Path(path).name} (iteration {it})"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True,
                    help="training checkpoint OR spike dump supplying weights")
    ap.add_argument("--batch", required=True,
                    help="spike dump supplying the samples to probe")
    ap.add_argument("--arm", default="B0",
                    help="arm whose architecture the weights belong to")
    ap.add_argument("--bad-index", type=int, default=12)
    ap.add_argument("--good-index", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    state, wlabel = load_state(args.weights)
    bpath = sorted(glob.glob(args.batch))[-1]
    bblob = torch.load(bpath, map_location="cpu", weights_only=False)
    if "micro_batches" not in bblob:
        raise SystemExit(f"{bpath} carries no micro_batches — --batch needs a "
                         "spike dump, not a plain checkpoint")
    d0, c0 = bblob["micro_batches"][0]

    print(f"weights : {wlabel}")
    print(f"batch   : {Path(bpath).name}")
    print(f"arm     : {args.arm}\n")

    cfg = build_config(args.arm, 0, 0, 0.0, 0)
    crit = build_loss(cfg["loss"])

    def probe(idx: int, label: str) -> None:
        model = build_model(cfg).to(args.device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected:
            raise SystemExit(f"{len(unexpected)} unexpected keys — weights do "
                             f"not match arm {args.arm}")
        model.train()
        model.zero_grad(set_to_none=True)
        d = d0[idx:idx + 1].to(args.device)
        c = c0[idx:idx + 1].to(args.device)
        loss = crit(model(d), c)
        loss.backward()

        gn, pn = defaultdict(float), defaultdict(float)
        for name, p in model.named_parameters():
            st = stage_of(name)
            pn[st] += float(p.detach().norm(2)) ** 2
            if p.grad is not None:
                gn[st] += float(p.grad.detach().norm(2)) ** 2

        print(f"=== {label} (sample {idx}) — loss {float(loss):.6g} ===")
        print(f"{'stage':<14}{'grad norm':>16}{'param norm':>14}{'g/p ratio':>16}")
        for st in STAGES:
            if st not in pn:
                continue
            g, q = gn[st] ** 0.5, pn[st] ** 0.5
            print(f"{st:<14}{g:>16.4g}{q:>14.4g}{g / max(q, 1e-12):>16.4g}")
        print(f"{'TOTAL':<14}{sum(gn.values()) ** 0.5:>16.4g}\n")

    probe(args.good_index, "healthy sample")
    probe(args.bad_index, "pathological sample")


if __name__ == "__main__":
    main()
