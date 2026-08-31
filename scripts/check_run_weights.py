"""Is the delta-init fix actually in effect in the RUNNING experiment?

Provenance (git_commit.txt) says the right code was loaded. This checks the
trained weights themselves, which is the only thing that proves it behaved.

Signature of the DEFECT (the killed run): conv.weight pinned at its Kaiming init
(|w|max ~= 0.09) and never moving, while fuse grows -- fuse amplifying a frozen
random kernel.
Signature of the FIX: conv started at delta (|w|max = 1.0, centre tap) and is now
adapting away from it, with fuse growing from zero.
"""
import glob
import sys

import torch


def probe(tag, pattern):
    hits = sorted(glob.glob(pattern))
    if not hits:
        print(f"  {tag}: no checkpoint")
        return
    ck = torch.load(hits[-1], map_location="cpu", weights_only=False)
    sd = ck.get("model", ck)
    it = ck.get("iteration")
    conv = [(k, v) for k, v in sd.items() if "reparam" in k and k.endswith("conv.weight")]
    fuse = [(k, v) for k, v in sd.items() if "reparam" in k and k.endswith("fuse.weight")]
    if not conv:
        print(f"  {tag}: no reparam block in this checkpoint")
        return
    print(f"\n  {tag}  (iteration {it})")
    for k, w in conv:
        kh, kw = w.shape[-2:]
        centre = float(w[:, 0, kh // 2, kw // 2].abs().mean())
        off = w.clone()
        off[:, 0, kh // 2, kw // 2] = 0
        offmax = float(off.abs().max())
        print(f"    {k}")
        print(f"      centre tap mean = {centre:.4f}   max off-centre = {offmax:.4f}"
              f"   |w|max = {float(w.abs().max()):.4f}")
        if centre > 0.5:
            verdict = "DELTA-derived, adapting  <- fix in effect"
        elif 0.05 < float(w.abs().max()) < 0.15 and centre < 0.15:
            verdict = "KAIMING-like, frozen  <- THE DEFECT"
        else:
            verdict = "indeterminate"
        print(f"      => {verdict}")
    for k, w in fuse:
        print(f"    {k:<44} |w|max = {float(w.abs().max()):.5f}"
              f"  {'(moved off zero)' if float(w.abs().max()) > 0 else '(still zero)'}")


print("=" * 76)
print("Delta-init: in effect in the trained weights?")
print("=" * 76)
probe("FAILED run (pre-fix, killed at 15k)",
      "runs/s33_b0v3_kd_k11/B0V3-KD-K11/*/last.pth")
probe("LIVE run (post-fix)",
      "runs/s33_b0v3_kd_k11_v2/B0V3-KD-K11/*/last.pth")
print()
