"""Repair AdaIR's frequency-mining mask -- ported from Himeth's diagnosis and
fix (~/FYP/Workspace/Himeth/{mask_fix.md,AdaIR/finetune/freq_fix.py} on
devon), adapted only for this repo's import layout. Logic unchanged.

Two defects in the released `FreModule.fft` (third_party/AdaIR/net/model.py):

  1. RESOLUTION. The mask half-width is `h_ = (h // 128 * rate).int()`.
     `h // 128` truncates to 0 for every feature map smaller than 256px, and
     this project (like upstream AdaIR) trains/evaluates on 128px patches --
     the largest map any AFLB ever sees is 64px. The mask is empty, so
     `low = IFFT(FFT(F)*0) = 0` and `high = IFFT(FFT(F)*1) = F` exactly: the
     FFT/IFFT pair cancels and the whole "frequency mining" module degenerates
     to `torch.abs()`. This was true for the entire released `adair3d.ckpt`
     we've used as the teacher throughout kd_feature_multitask.

  2. GRADIENT. The mask is built by writing hard 1s at indices derived from
     `rate`; index selection is not differentiable, so `rate_conv` (the
     paper's MGB) gets exactly zero gradient and never leaves its random
     init -- outputs ~0.5 regardless of the actual degradation.

The fix replaces mask construction with a soft, resolution-independent
sigmoid product (a fraction of the half-spectrum, not a bin count),
differentiable in the same alpha/beta rate_conv already outputs. As tau->0
this converges to the hard rectangle the paper describes. Everything
downstream (FFT, shift, the complementary 1-M split, the three
cross-attentions, FMoM, the para1/para2 gate) is untouched, so any measured
change is attributable to the mask alone.

Verified per Himeth's own controlled experiment (mask_fix.md): the real
contribution of a WORKING mask is small and dehaze-concentrated (+0.056dB
overall, +0.244dB dehaze, ~0 denoise) -- much smaller than AdaIR's own
paper-claimed +1.58dB frequency-mining ablation. Also consistent with this
project's own earlier causal audit (src/models/teacher_wrapper.py's
`forward_with_latent` docstring: "TEST05.5's causal audit found
[latent_pre], not the frequency pathway, is the well-supported distillation
signal"). The main value of Himeth's fine-tuned checkpoint is that it is a
strictly better teacher overall (+1.07dB vs released), not that the
frequency path itself is now a major lever.

Adds NO parameters -- `apply_freq_fix` only rebinds each FreModule
instance's `.fft` method, so it is compatible with the exact state_dict any
released or fine-tuned AdaIR checkpoint already uses.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _soft_mask(shape: tuple[int, int], alpha: torch.Tensor, beta: torch.Tensor,
              tau: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Separable soft low-pass over an fftshift-ed spectrum.

    alpha, beta: (B,1,1,1) cutoffs in (0,1), as a fraction of the
    half-spectrum. Returns (B,1,h,w) in (0,1).
    """
    h, w = shape
    iy = torch.arange(h, device=device, dtype=dtype)
    ix = torch.arange(w, device=device, dtype=dtype)
    u = (iy - h // 2).abs() / max(h // 2, 1)
    v = (ix - w // 2).abs() / max(w // 2, 1)
    mu = torch.sigmoid((alpha - u.view(1, 1, h, 1)) / tau)
    mv = torch.sigmoid((beta - v.view(1, 1, 1, w)) / tau)
    return mu * mv


def _hard_mask_upstream(shape: tuple[int, int], alpha: torch.Tensor,
                        beta: torch.Tensor, n: int = 128) -> torch.Tensor:
    """Bit-for-bit reproduction of the released (broken) mask, for a control arm."""
    h, w = shape
    mask = torch.zeros((alpha.shape[0], 1, h, w), device=alpha.device, dtype=alpha.dtype)
    for i in range(alpha.shape[0]):
        h_ = int((h // n * alpha[i, 0, 0, 0]).int())
        w_ = int((w // n * beta[i, 0, 0, 0]).int())
        if h_ > 0 and w_ > 0:
            mask[i, :, h // 2 - h_:h // 2 + h_, w // 2 - w_:w // 2 + w_] = 1
    return mask


def _fft_patched(self, x: torch.Tensor, n: int = 128):
    """Drop-in replacement for FreModule.fft; signature kept for compatibility."""
    x = self.conv1(x)
    h, w = x.shape[-2:]

    rate = self.rate_conv(F.adaptive_avg_pool2d(x, 1)).sigmoid()  # (B,2,1,1)
    alpha, beta = rate[:, 0:1], rate[:, 1:2]

    if self._freqfix_mode == "soft":
        mask = _soft_mask((h, w), alpha, beta, self._freqfix_tau, x.device, x.dtype)
    elif self._freqfix_mode == "upstream":
        mask = _hard_mask_upstream((h, w), alpha, beta, n)
    else:
        raise ValueError(self._freqfix_mode)

    self._freqfix_last = {  # for probing/logging only
        "alpha": float(alpha.mean()), "beta": float(beta.mean()),
        "coverage": float(mask.mean()), "hw": (h, w),
    }

    fft = torch.fft.fft2(x, norm="forward", dim=(-2, -1))
    fft = self.shift(fft)

    high = self.unshift(fft * (1 - mask))
    high = torch.abs(torch.fft.ifft2(high, norm="forward", dim=(-2, -1)))

    low = self.unshift(fft * mask)
    low = torch.abs(torch.fft.ifft2(low, norm="forward", dim=(-2, -1)))

    return high, low


def _load_fremodule_class():
    """Import FreModule the SAME way src/models/teacher_wrapper.py's
    `_load_adair_class()` imports AdaIR -- sys.path-inserted and under the
    bare module name `net.model`, not `third_party.AdaIR.net.model`. Python
    caches imports by exact dotted name, so importing the identical file
    under a different name would create a SECOND, distinct class object,
    and `isinstance(m, FreModule)` would silently never match any module
    actually built via `_load_adair_class()`. Reusing the cached `net.model`
    import (already present in sys.modules after any FrozenTeacher has been
    constructed) guarantees the same class identity.
    """
    import sys
    from src.utils.config import REPO_ROOT

    if "net.model" in sys.modules:
        return sys.modules["net.model"].FreModule
    repo = REPO_ROOT / "third_party" / "AdaIR"
    sys.path.insert(0, str(repo))
    try:
        from net.model import FreModule
        return FreModule
    finally:
        sys.path.remove(str(repo))


def apply_freq_fix(net: nn.Module, mode: str = "soft", tau: float = 0.05) -> list[nn.Module]:
    """Monkeypatch every FreModule in `net`. Adds no parameters, so the
    state_dict stays compatible with any released or fine-tuned checkpoint.
    """
    FreModule = _load_fremodule_class()

    assert mode in ("soft", "upstream")
    mods = [m for m in net.modules() if isinstance(m, FreModule)]
    if not mods:
        raise RuntimeError("no FreModule found -- was the net built with decoder=True?")
    for m in mods:
        m._freqfix_mode = mode
        m._freqfix_tau = tau
        m._freqfix_last = None
        m.fft = _fft_patched.__get__(m, FreModule)
    return mods
