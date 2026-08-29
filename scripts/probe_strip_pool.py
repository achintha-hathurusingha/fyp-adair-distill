import sys
sys.path.insert(0, ".")
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.export.op_coverage import op_histogram, render_markdown


class StripPoolingGate(nn.Module):
    """Hu, Zhang, Xie & Yang, 'Strip Pooling: Rethinking Spatial Pooling for
    Scene Parsing,' CVPR 2020. Pools along one FULL spatial axis at a time
    (adaptive_avg_pool2d to (H,1) / (1,W)) -- genuinely global context along
    that axis (not just a larger local window), targeting the "large-range/
    global information" capability arXiv:2310.11881 found convolution-only
    backbones (NAFNet included) specifically lack for deraining/dehazing --
    without attention/Softmax/MatMul (scripts/probe_mdta.py: UNKNOWN on
    qnn/tflite/tensorrt) or FFT (confirmed unsupported on any mobile NPU
    delegate). Zero-init final projection: additive residual, identity at
    init.
    """

    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        hidden = max(1, dim // reduction)
        self.reduce = nn.Conv2d(dim, hidden, 1, bias=False)
        self.conv_h = nn.Conv2d(hidden, hidden, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.conv_w = nn.Conv2d(hidden, hidden, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.fuse = nn.Conv2d(hidden, dim, 1, bias=False)
        nn.init.zeros_(self.fuse.weight)

    def forward(self, x):
        b, c, h, w = x.shape
        y = self.reduce(x)
        yh = self.conv_h(F.adaptive_avg_pool2d(y, (h, 1)))  # (B,hidden,H,1)
        yw = self.conv_w(F.adaptive_avg_pool2d(y, (1, w)))  # (B,hidden,1,W)
        # rely on ordinary tensor-addition broadcasting (ONNX Add is
        # natively broadcasting) instead of an explicit .expand() call --
        # the earlier version's runtime-shape-derived .expand(-1,-1,h,w)
        # traced to Equal/Where/ConstantOfShape/Expand, all UNKNOWN on
        # every backend (verified, not assumed); this version needs none
        # of that machinery.
        return x + self.fuse(yh + yw)


torch.manual_seed(0)
m = StripPoolingGate(32).eval()
x = torch.randn(1, 32, 1, requires_grad=False)
x = torch.randn(2, 32, 48, 48, requires_grad=True)
y = m(x)
assert torch.allclose(y, x), "not identity at init"
y.sum().backward()
assert x.grad is not None and torch.isfinite(x.grad).all()
print(f"StripPoolingGate: identity at init OK, grad finite, params={sum(p.numel() for p in m.parameters())}")

m_probe = StripPoolingGate(32).eval()
dummy = torch.randn(1, 32, 64, 64)
onnx_path = "runs/strippool_probe.onnx"
torch.onnx.export(m_probe, dummy, onnx_path, opset_version=17, input_names=["x"], output_names=["y"])
hist = op_histogram(onnx_path)
print("\n--- StripPoolingGate ONNX op histogram ---")
for op, count in sorted(hist.items()):
    print(f"  {op}: {count}")
print("\n" + render_markdown(onnx_path))
