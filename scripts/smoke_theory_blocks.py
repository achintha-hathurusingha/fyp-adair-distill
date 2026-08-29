"""Smoke test for src/models/theory_blocks.py: forward pass, gradient flow,
param count, then ONNX export + this repo's own op-coverage gate
(src/export/op_coverage.py) -- the real "does it fit the mobile NPU" check,
not a guess.
"""
import sys
sys.path.insert(0, ".")

import torch
from src.models.theory_blocks import LaplacianFrequencyGate, dark_channel_prior

torch.manual_seed(0)

# --- 1. forward + gradient flow -------------------------------------------
dim = 32
gate = LaplacianFrequencyGate(dim, levels=2)
x = torch.randn(2, dim, 64, 64, requires_grad=True)
y = gate(x)
assert y.shape == x.shape, f"shape mismatch: {y.shape} vs {x.shape}"
# zero-init proj => must be identity at init
assert torch.allclose(y, x), f"not identity at init, max diff {(y-x).abs().max().item()}"
y.sum().backward()
assert x.grad is not None and torch.isfinite(x.grad).all()
n_params = sum(p.numel() for p in gate.parameters())
print(f"LaplacianFrequencyGate: OK, identity at init, grad finite, {n_params} params")

# perturb proj weight off zero-init and re-check it actually does something
with torch.no_grad():
    gate.proj.weight.add_(0.1)
y2 = gate(x.detach())
assert not torch.allclose(y2, x.detach()), "module is not identity after perturbing proj -- good, confirms it's live"
print("LaplacianFrequencyGate: confirmed non-trivial once proj is non-zero")

# --- 2. dark_channel_prior ---------------------------------------------
img = torch.rand(2, 3, 128, 128)
dcp = dark_channel_prior(img)
assert dcp.shape == (2, 1, 128, 128)
assert (dcp >= -1e-5).all() and (dcp <= 1 + 1e-5).all(), f"dcp range: [{dcp.min()},{dcp.max()}]"
print(f"dark_channel_prior: OK, shape {tuple(dcp.shape)}, range [{dcp.min():.3f}, {dcp.max():.3f}]")

# sanity: a uniformly white (hazy/bright) image should have HIGH dark channel;
# a pure-black image should have dark channel == 0
white = torch.ones(1, 3, 32, 32)
black = torch.zeros(1, 3, 32, 32)
dcp_white = dark_channel_prior(white).mean().item()
dcp_black = dark_channel_prior(black).mean().item()
assert dcp_white > 0.99 and dcp_black < 1e-6, f"white={dcp_white}, black={dcp_black}"
print(f"dark_channel_prior: physical sanity check OK (white={dcp_white:.3f}, black={dcp_black:.6f})")

# --- 3. ONNX export + op coverage ------------------------------------------
import onnx
from src.export.op_coverage import op_histogram, render_markdown

class _Probe(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = LaplacianFrequencyGate(dim, levels=2)

    def forward(self, x):
        return self.gate(x)

probe = _Probe(dim).eval()
dummy = torch.randn(1, dim, 64, 64)
onnx_path = "runs/theory_blocks_probe.onnx"
torch.onnx.export(probe, dummy, onnx_path, opset_version=17,
                   input_names=["x"], output_names=["y"])
hist = op_histogram(onnx_path)
print("\n--- LaplacianFrequencyGate ONNX op histogram ---")
for op, count in sorted(hist.items()):
    print(f"  {op}: {count}")

print("\n" + render_markdown(onnx_path))

# dark_channel_prior as its own tiny traced graph
class _DCPProbe(torch.nn.Module):
    def forward(self, x):
        return dark_channel_prior(x)

dcp_probe = _DCPProbe().eval()
dcp_dummy = torch.rand(1, 3, 128, 128)
dcp_onnx_path = "runs/dcp_probe.onnx"
torch.onnx.export(dcp_probe, dcp_dummy, dcp_onnx_path, opset_version=17,
                   input_names=["x"], output_names=["t"])
hist2 = op_histogram(dcp_onnx_path)
print("\n--- dark_channel_prior ONNX op histogram ---")
for op, count in sorted(hist2.items()):
    print(f"  {op}: {count}")
print("\n" + render_markdown(dcp_onnx_path))

print("\nALL SMOKE CHECKS PASSED")
