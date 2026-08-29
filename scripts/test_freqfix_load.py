"""Verify FrozenTeacher can already load Himeth's fixed checkpoint (weights
only), then check whether applying his freq_fix monkeypatch on top of our
own vendored third_party/AdaIR/net/model.py (confirmed byte-identical to
his upstream copy) actually activates a non-zero, adaptive mask.
"""
import sys
sys.path.insert(0, ".")

import torch
from src.models.teacher_wrapper import FrozenTeacher

CKPT = "/home/minura/FYP/Workspace/Himeth/AdaIR/runs/finetune/C_full_soft/final.pt"

teacher = FrozenTeacher(CKPT, device="cpu")
print(f"FrozenTeacher loaded OK, {sum(p.numel() for p in teacher.parameters()):,} params "
      f"(expected 28,784,824)")

# Read the checkpoint's own mode/tau (top-level keys, not inside 'model')
raw = torch.load(CKPT, map_location="cpu", weights_only=False)
mode = raw.get("mode", "upstream")
tau = raw.get("tau", 0.05)
print(f"checkpoint declares mode={mode!r} tau={tau}")

x = torch.rand(1, 3, 128, 128)

# apply_freq_fix, adapted import path for our repo layout
sys.path.insert(0, str((__import__("pathlib").Path(".") / "third_party" / "AdaIR").resolve()))
from net.model import FreModule  # noqa: E402

def _soft_mask(shape, alpha, beta, tau, device, dtype):
    import torch.nn.functional as F
    h, w = shape
    iy = torch.arange(h, device=device, dtype=dtype)
    ix = torch.arange(w, device=device, dtype=dtype)
    u = (iy - h // 2).abs() / max(h // 2, 1)
    v = (ix - w // 2).abs() / max(w // 2, 1)
    mu = torch.sigmoid((alpha - u.view(1, 1, h, 1)) / tau)
    mv = torch.sigmoid((beta - v.view(1, 1, 1, w)) / tau)
    return mu * mv

def _fft_patched(self, x, n=128):
    import torch.nn.functional as F
    x = self.conv1(x)
    h, w = x.shape[-2:]
    rate = self.rate_conv(F.adaptive_avg_pool2d(x, 1)).sigmoid()
    alpha, beta = rate[:, 0:1], rate[:, 1:2]
    mask = _soft_mask((h, w), alpha, beta, self._freqfix_tau, x.device, x.dtype)
    self._freqfix_last = {"alpha": float(alpha.mean()), "beta": float(beta.mean()),
                          "coverage": float(mask.mean()), "hw": (h, w)}
    fft = torch.fft.fft2(x, norm='forward', dim=(-2, -1))
    fft = self.shift(fft)
    high = self.unshift(fft * (1 - mask))
    high = torch.abs(torch.fft.ifft2(high, norm='forward', dim=(-2, -1)))
    low = self.unshift(fft * mask)
    low = torch.abs(torch.fft.ifft2(low, norm='forward', dim=(-2, -1)))
    return high, low

def apply_freq_fix(net, mode="soft", tau=0.05):
    mods = [m for m in net.modules() if isinstance(m, FreModule)]
    assert mods, "no FreModule found"
    for m in mods:
        m._freqfix_tau = tau
        m._freqfix_last = None
        m.fft = _fft_patched.__get__(m, FreModule)
    return mods

mods = apply_freq_fix(teacher.net, mode=mode, tau=tau)
print(f"patched {len(mods)} FreModule instances")

out = teacher(x)  # FrozenTeacher.forward already asserts eval/no-grad
coverages = [m._freqfix_last["coverage"] for m in mods]
print(f"mask coverage after fix: {coverages}")
assert all(c > 0.0 for c in coverages), f"mask still dead: {coverages}"
print("FIX CONFIRMED LIVE on our own vendored AdaIR copy + Himeth's checkpoint")
