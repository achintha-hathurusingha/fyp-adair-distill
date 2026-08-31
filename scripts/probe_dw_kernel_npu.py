"""Does LARGE-KERNEL DEPTHWISE conv actually run on the NPU, or fall back to CPU?

S0.2 passed the reparameterizable oriented block through op_coverage and found
the merged graph is one `Conv` with zero UNKNOWN ops. That gate explicitly could
NOT see two things, and said so: op_coverage reads only the op TYPE, so it
cannot distinguish depthwise (groups=dim) from dense, and it cannot see kernel
size. ONNX-exportable is not the same as NPU-executable.

Our own 36 AI Hub profiles (reports/aihub_jobs.json) all report
compute_units {"NPU": N} with no CPU entry -- but every one of those models is
NAFNet-family, i.e. depthwise 3x3. They are evidence that depthwise PER SE maps
to the NPU on this target. They say nothing about k=11, which is what S3.1
actually builds, and NPU depthwise units commonly cap out at 3x3/5x5/7x7.

So this sweeps the merged deployment block across kernel size on real hardware
and reads the compute-unit breakdown. What matters is not latency alone but
`npu_fallback_layers` -- any op landing on CPU/GPU instead of NPU/HTP/DSP.

If k=11 falls back, the mitigation is already measured: S0.1 found the
oriented-over-isotropic gain is +0.365 dB at k=7 against +0.384 at k=11, so
dropping to the largest kernel that stays on the NPU costs ~0.02 dB of oracle
headroom. That is a resize, not an abandonment -- but we should know BEFORE
S3.3's remaining GPU hours go into a k=11 design.

Positive control: a DENSE 3x3 conv of the same channel count. If depthwise and
dense both report zero fallback, the metric is not discriminating and the
result means nothing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

import torch
from torch import nn

from src.export.aihub import submit_and_profile
from src.models.reparam_oriented import ReparamOrientedBlock

DEVICE = "Samsung Galaxy S24"     # Snapdragon 8 Gen 3 - the target used previously
DIM = 32                           # decoder stage width at w16
RES = 256                          # the resolution prior on-device work used
KS = [3, 5, 7, 9, 11]
OUT = Path("runs/npu_dw")


class DenseControl(nn.Module):
    """Positive control: dense 3x3 + 1x1, same shape of graph, groups=1."""

    def __init__(self, dim, k=3):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, k, padding=k // 2, bias=False)
        self.fuse = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x):
        return x + self.fuse(self.conv(x))


def export(mod, path):
    mod.eval()
    torch.onnx.export(mod, torch.randn(1, DIM, RES, RES), str(path),
                      opset_version=17, input_names=["x"], output_names=["y"])
    return path


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cases = []

    for k in KS:
        torch.manual_seed(0)
        blk = ReparamOrientedBlock(DIM, k=k)
        with torch.no_grad():
            for p in blk.core.parameters():
                p.normal_(0, 0.2)
            blk.core._apply_mask()
            blk.fuse.weight.normal_(0, 0.1)
        merged = blk.merge()          # depthwise k x k + 1x1, what actually ships
        p = export(merged, OUT / f"dw{k}.onnx")
        cases.append((f"depthwise_{k}x{k}", p))

    torch.manual_seed(0)
    p = export(DenseControl(DIM, 3), OUT / "dense3.onnx")
    cases.append(("dense_3x3_CONTROL", p))

    print(f"device: {DEVICE}   input: (1,{DIM},{RES},{RES})\n")
    print(f"{'case':<22}{'lat ms':>9}{'NPU':>7}{'fallback':>10}  breakdown / error")
    print("-" * 86)
    results = {}
    for name, path in cases:
        r = submit_and_profile(str(path), name, DEVICE,
                               input_shape=(1, DIM, RES, RES), calib_samples=8)
        bd = r.compute_unit_breakdown or {}
        npu = sum(n for u, n in bd.items() if u.upper() in ("NPU", "HTP", "DSP"))
        fb = r.npu_fallback_layers
        lat = f"{r.inference_latency_ms:.3f}" if r.inference_latency_ms else "—"
        detail = json.dumps(bd) if bd else (r.error or "")[:44]
        flag = "  <-- FALLBACK" if fb else ""
        print(f"{name:<22}{lat:>9}{npu:>7}{fb:>10}  {detail}{flag}", flush=True)
        results[name] = {"latency_ms": r.inference_latency_ms,
                         "compute_units": bd, "npu_fallback_layers": fb,
                         "error": r.error, "stage_failed": r.stage_failed,
                         "job_urls": r.job_urls}

    Path("reports").mkdir(exist_ok=True)
    Path("reports/npu_depthwise_kernel_sweep.json").write_text(
        json.dumps(results, indent=2))

    print("\n" + "=" * 86)
    ok = [k for k in KS
          if results.get(f"depthwise_{k}x{k}", {}).get("npu_fallback_layers") == 0
          and results.get(f"depthwise_{k}x{k}", {}).get("compute_units")]
    ctrl = results.get("dense_3x3_CONTROL", {})
    print(f"  control (dense 3x3): fallback={ctrl.get('npu_fallback_layers')} "
          f"units={ctrl.get('compute_units')}")
    if ok:
        print(f"  depthwise kernel sizes running FULLY on NPU: {ok}")
        print(f"  -> largest usable k = {max(ok)}")
        if 11 not in ok:
            print("  -> S3.1 must drop from k=11. S0.1 measured k=7 at +0.365 dB "
                  "vs k=11 at +0.384, so the cost is ~0.02 dB.")
    else:
        print("  NO depthwise kernel mapped cleanly — check the errors above "
              "before concluding; a compile failure is not the same as a fallback.")
    print("\nwrote reports/npu_depthwise_kernel_sweep.json")


if __name__ == "__main__":
    main()
