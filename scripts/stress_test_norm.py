"""Adversarial forward-pass stress test for normalization variants.

WHY THIS EXISTS. B0 died at iteration 24356 because one rare low-variance crop
drove the full-resolution decoder stage (`dec3`) to max|a| 5.6e6 and a gradient
norm of 6.5e7. Under N-F that stage has no normalisation to bound its input.
Finding it cost ~25,000 training iterations across several runs; this script
finds it in seconds. It should have existed before B0 ever launched.

Run it against every norm variant and every candidate fix, on every geometry,
BEFORE trusting an architecture decision.

    python scripts/stress_test_norm.py --weights <spike_dump.pt>
    python scripts/stress_test_norm.py            # untrained (weak, see below)

IMPORTANT — untrained weights prove almost nothing here. NAFNet initialises
`beta`/`gamma` to zero, so every block is the identity at init and nothing can
explode. A meaningful test needs weights from a model that has actually drifted
into the fragile region. `--weights` accepts a spike dump (which carries the
model state) or any checkpoint; without it the run is a smoke test only, and
says so in its output.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import torch

from src.data.datasets import load_rgb_uint8
from src.losses.reconstruction import build_loss
from src.models.nafnet import NAFNet
from src.utils.config import REPO_ROOT, load_paths

#: Geometries under test (family lock: reports/family_reselection.md).
GEOMETRIES = {
    "S(w16_b8)": dict(width=16, enc_blk_nums=[1, 1, 1, 8], middle_blk_num=2,
                      dec_blk_nums=[1, 1, 1, 1]),
    "M(w16_sidd)": dict(width=16, enc_blk_nums=[2, 2, 4, 8], middle_blk_num=12,
                        dec_blk_nums=[2, 2, 2, 2]),
}

#: Norm variants and candidate fixes. `full` applies everywhere; `full_res`
#: overrides only the full-resolution stages (encoder level 0 AND decoder
#: level 0 — the model applies it symmetrically at both ends).
VARIANTS = {
    "N-A  (LayerNorm everywhere)": dict(norm_type="layernorm2d",
                                        full_res_norm_type=None),
    "N-F  (affine at full-res)": dict(norm_type="layernorm2d",
                                      full_res_norm_type="affine"),
    "Fix-C (clamp at full-res)": dict(norm_type="layernorm2d",
                                      full_res_norm_type="affine_clamp"),
}

#: Output magnitude above which a variant is considered to have failed. Healthy
#: outputs live in roughly [0, 1]; the B0 failure produced 7e5.
FAIL_ABOVE = 100.0


def adversarial_inputs(patch: int, device: str) -> dict[str, torch.Tensor]:
    """Deliberately worst-case crops, plus the shapes that actually broke B0."""
    z = torch.zeros(1, 3, patch, patch, device=device)
    cases = {
        "all-black": z.clone(),
        "all-white": torch.ones_like(z),
        "near-zero-var": torch.full_like(z, 0.05) + torch.randn_like(z) * 0.002,
        "dark-lowvar": torch.full_like(z, 0.055) + torch.randn_like(z) * 0.084,
        "extreme-noise": torch.clamp(torch.rand_like(z) * 4 - 1.5, 0, 1),
        "half-saturated": torch.cat(
            [torch.zeros(1, 3, patch // 2, patch, device=device),
             torch.ones(1, 3, patch - patch // 2, patch, device=device)], dim=2),
        "typical": torch.rand_like(z) * 0.6 + 0.2,
    }
    return cases


def real_dark_crops(patch: int, device: str, n: int = 3) -> dict[str, torch.Tensor]:
    """The lowest-variance crops in the real training set.

    Synthetic worst cases can be unrepresentative in both directions; these are
    the actual images most like the one that killed B0.
    """
    paths = load_paths()
    root = Path(paths["data_root"])
    if not root.is_absolute():
        root = REPO_ROOT / root
    root = root / "Train" / "Denoise"
    files = sorted(p for p in root.rglob("*")
                   if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})
    if not files:
        return {}
    scored = []
    for f in files[::37]:                      # sample the set, not all of it
        try:
            img = load_rgb_uint8(f, base=1)
        except Exception:
            continue
        a = img[:patch, :patch]
        if a.shape[0] < patch or a.shape[1] < patch:
            continue
        scored.append((float(a.std()) / 255.0, f, a))
    scored.sort(key=lambda t: t[0])
    out = {}
    for std, f, a in scored[:n]:
        t = torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1)))
        out[f"real:{f.name}(std{std:.3f})"] = (t.float() / 255.0)[None].to(device)
    return out


def spike_samples(blob: dict, device: str) -> dict[str, torch.Tensor]:
    """The ACTUAL crops from a captured spike.

    Synthetic worst cases turned out not to reproduce the B0 failure at all —
    all-black, all-white, near-zero-variance and even the lowest-variance real
    crops in the training set all passed cleanly under N-F. Only the genuine
    sample does. A stress suite that omits the known failing input is testing
    the tester's imagination, so the captured batch is always included.
    """
    out = {}
    for mi, (d, _c) in enumerate(blob.get("micro_batches", [])):
        for i in range(d.shape[0]):
            out[f"spike:m{mi}s{i}"] = d[i:i + 1].to(device)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default=None,
                    help="spike dump or checkpoint supplying TRAINED weights")
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--geometry", default=None, choices=sorted(GEOMETRIES))
    args = ap.parse_args()

    state = None
    if args.weights:
        path = sorted(glob.glob(args.weights))[-1]
        blob = torch.load(path, map_location="cpu", weights_only=False)
        state = blob.get("model", blob)
        print(f"weights : {Path(path).name}"
              + (f"  (spike at iteration {blob['iteration']})"
                 if "iteration" in blob else ""))
    else:
        print("weights : UNTRAINED — smoke test only.")
        print("          beta/gamma start at zero, so every block is the")
        print("          identity and nothing can explode. Not evidence.")

    cases = adversarial_inputs(args.patch, args.device)
    cases.update(real_dark_crops(args.patch, args.device))
    if args.weights and "micro_batches" in blob:
        spikes = spike_samples(blob, args.device)
        cases.update(spikes)
        print(f"          + {len(spikes)} real crops from the captured spike")
    print(f"cases   : {len(cases)}  |  fail threshold: max|out| > {FAIL_ABOVE}\n")

    geoms = [args.geometry] if args.geometry else list(GEOMETRIES)
    crit = build_loss({"name": "charbonnier", "eps": 1e-3})

    for gname in geoms:
        geo = GEOMETRIES[gname]
        print(f"=== {gname} ===")
        header = f"{'case':<28}" + "".join(f"{v:>30}" for v in VARIANTS)
        print(header)
        print("-" * len(header))

        models = {}
        for vname, norm in VARIANTS.items():
            m = NAFNet(**geo, **norm).to(args.device).eval()
            if state is not None:
                missing, unexpected = m.load_state_dict(state, strict=False)
                if unexpected:
                    raise ValueError(
                        f"{gname}/{vname}: checkpoint has {len(unexpected)} "
                        "unexpected keys — geometry does not match the weights")
            models[vname] = m

        failures = {v: 0 for v in VARIANTS}
        hidden = 0
        interesting = {}
        for cname, x in cases.items():
            if cname.startswith("spike:"):
                with torch.no_grad():
                    ref = models["N-F  (affine at full-res)"](x)
                if float(ref.abs().max()) <= FAIL_ABOVE:
                    continue          # healthy spike-batch sample; not shown
            interesting[cname] = x

        for cname, x in interesting.items():
            row = f"{cname:<28}"
            for vname, m in models.items():
                with torch.no_grad():
                    out = m(x)
                mx = float(out.abs().max())
                bad = mx > FAIL_ABOVE or not np.isfinite(mx)
                failures[vname] += bool(bad)
                row += f"{('FAIL ' if bad else '     ') + f'{mx:>10.4g}':>30}"
            print(row)
        hidden = len(cases) - len(interesting)
        if hidden:
            print(f"{'(' + str(hidden) + ' healthy spike-batch crops hidden)':<28}")
        print("-" * len(header))
        summary = f"{'FAILURES':<28}"
        for vname in VARIANTS:
            n = failures[vname]
            summary += f"{(f'{n}/{len(interesting)}'):>30}"
        print(summary + "\n")


if __name__ == "__main__":
    main()
