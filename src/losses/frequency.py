"""Frequency-domain distillation loss.

Matches the student's spectrum to the teacher's. Motivated by F7: AdaIR's
dehazing advantage comes from ``FreModule``'s FFT-based frequency mining, and
the response term (Charbonnier on pixels) does not specifically ask the student
to reproduce that. This term does.

Worth stating up front, because it frames what is being attempted: **AdaIR
itself trains with plain L1** (``third_party/AdaIR/train.py:21``). Its frequency
behaviour is entirely architectural, not the product of a frequency loss. So
this term is not "reusing the teacher's loss" — it is trying to transfer an
architectural property through supervision, to a student that has no such
architecture.

MAGNITUDE OR COMPLEX. The brief called magnitude-only "standard practice"; the
literature is actually split, so the option is explicit rather than assumed:

* **complex** — the widely-cited MIMO-UNet frequency reconstruction loss takes
  L1 over the full ``rfft2`` output, real and imaginary, so phase is included.
  Focal Frequency Loss likewise weights a complex spectrum distance.
* **magnitude** — amplitude-only losses appear in the deblurring literature,
  usually where amplitude and phase are treated as separable factors.

For **dehazing specifically** magnitude is the better-motivated default, and the
reason is the degradation model rather than convention. Haze is
``I = J*t + A*(1-t)``: for locally uniform transmission this is an affine map,
which scales the AC coefficients by ``t`` and shifts DC. It is an
amplitude-domain corruption that leaves phase largely intact. Matching
magnitude therefore targets what haze actually destroys, and asking the student
to match phase it already has would spend capacity on an unbroken quantity.

Both modes are implemented and selected by config, so the assumption is testable
rather than baked in.

NORMALISATION. ``norm="ortho"`` makes the transform unitary, which keeps
coefficient magnitudes independent of transform size. The *loss* is then
scale-free **for broadband signals** — measured at 0.0563 / 0.0560 / 0.0565 for
the same perturbation at 32 / 64 / 128 px, i.e. within 1%. Natural images are
broadband, so this is the case that matters in training and evaluation.

It is **not** scale-free for a sparse spectrum: a single sinusoid gives a loss
proportional to 1/N, because the energy sits in a fixed number of coefficients
while the mean is taken over a count growing as N². Stated because the first
version of this docstring claimed unconditional scale-invariance and the unit
test caught it — the property holds for the signals this term will see, and the
qualifier is the honest form of the claim.

PRECISION. ``torch.fft`` has no bfloat16 kernel, the same limitation that forces
the teacher to run in fp32 (and that blocks AdaIR's ONNX export entirely — F7).
Inputs are cast to float32 here rather than requiring every caller to remember.
"""
from __future__ import annotations

import torch


def spectrum_loss(pred: torch.Tensor, target: torch.Tensor, *,
                  mode: str = "magnitude") -> torch.Tensor:
    """L1 distance between the 2D spectra of ``pred`` and ``target``.

    Args:
        pred: NCHW student output.
        target: NCHW teacher output, same shape.
        mode: ``"magnitude"`` compares ``|F|`` and ignores phase;
            ``"complex"`` compares real and imaginary parts, including phase.

    Returns:
        Scalar loss.

    Raises:
        ValueError: on a shape mismatch or an unknown mode — a silent fallback
            to one convention while the config names the other would make the
            reported ablation meaningless.
    """
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: {pred.shape} vs {target.shape}")
    if mode not in ("magnitude", "complex"):
        raise ValueError(
            f"mode must be 'magnitude' or 'complex', got {mode!r}")

    # Per channel, over the spatial dims. rfft2 exploits conjugate symmetry of a
    # real input, so it holds every unique coefficient without duplication.
    fp = torch.fft.rfft2(pred.float(), norm="ortho")
    ft = torch.fft.rfft2(target.float(), norm="ortho")

    if mode == "magnitude":
        return torch.mean(torch.abs(torch.abs(fp) - torch.abs(ft)))
    return torch.mean(torch.abs(torch.view_as_real(fp) - torch.view_as_real(ft)))
