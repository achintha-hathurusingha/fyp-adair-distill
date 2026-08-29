"""Decisive op-coverage probe for MDTA (channel-attention transformer block)
-- left UNVERIFIED (not disproven) in the previous theory review. Given the
new evidence that global/large-range context is specifically what NAFNet's
convolution-only design lacks for derain/dehaze (arXiv:2310.11881), this is
now worth actually checking rather than leaving as a guess either way.
"""
import sys
sys.path.insert(0, ".")

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from src.export.op_coverage import op_histogram, render_markdown


class MDTA(nn.Module):
    """Restormer/AdaIR's own channel-attention block, unmodified."""

    def __init__(self, dim, num_heads, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, 3, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = rearrange(out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w)
        return self.project_out(out)


m = MDTA(dim=32, num_heads=4).eval()
dummy = torch.randn(1, 32, 32, 32)
onnx_path = "runs/mdta_probe.onnx"
torch.onnx.export(m, dummy, onnx_path, opset_version=17, input_names=["x"], output_names=["y"])
hist = op_histogram(onnx_path)
print("--- MDTA ONNX op histogram ---")
for op, count in sorted(hist.items()):
    print(f"  {op}: {count}")
print("\n" + render_markdown(onnx_path))
