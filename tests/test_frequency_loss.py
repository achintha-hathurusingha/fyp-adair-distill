"""The frequency distillation term.

Every expected value here is derived from Fourier properties, not from running
the implementation and recording what it printed. That distinction matters: this
project has already shipped one known-answer test whose answer was fabricated to
match the code, and a loss that computes the wrong thing consistently would pass
any test written that way.
"""
from __future__ import annotations

import math

import pytest
import torch

from src.losses.frequency import spectrum_loss


def _sinusoid(n: int = 32, k: int = 3, phase: float = 0.0) -> torch.Tensor:
    """A single spatial frequency, NCHW. Its spectrum is two conjugate spikes."""
    x = torch.arange(n, dtype=torch.float32)
    row = torch.sin(2 * math.pi * k * x / n + phase)
    return row[None, None, None, :].expand(1, 1, n, n).contiguous()


@pytest.mark.parametrize("mode", ["magnitude", "complex"])
def test_identical_inputs_give_zero(mode) -> None:
    x = torch.rand(2, 3, 16, 16)
    assert spectrum_loss(x, x.clone(), mode=mode).item() == pytest.approx(0.0, abs=1e-6)


def test_phase_shift_is_invisible_to_magnitude_and_visible_to_complex() -> None:
    """A translated sinusoid has an IDENTICAL magnitude spectrum.

    This is the Fourier shift theorem: translation multiplies each coefficient
    by a unit-modulus phase factor, leaving |F| unchanged. So the magnitude mode
    must score exactly zero and the complex mode must not — which is the whole
    difference between the two conventions, established analytically rather than
    by observation.
    """
    a = _sinusoid(phase=0.0)
    b = _sinusoid(phase=math.pi / 2)          # a quarter-period translation

    mag = spectrum_loss(a, b, mode="magnitude").item()
    cpx = spectrum_loss(a, b, mode="complex").item()
    assert mag == pytest.approx(0.0, abs=1e-5), f"magnitude mode saw phase: {mag}"
    assert cpx > 1e-3, f"complex mode ignored phase: {cpx}"


def test_dc_offset_moves_only_the_dc_coefficient() -> None:
    """Adding a constant c to an NxN image changes F[0,0] by c*N under 'ortho'.

    With norm='ortho' the forward transform carries 1/N for an NxN image, so a
    constant offset c contributes c*N*N/N = c*N at DC and nothing elsewhere.
    The loss averages |dF| over all C*H*(W//2+1) coefficients, so the expected
    value is c*N divided by that count.
    """
    n, c = 32, 0.25
    x = torch.zeros(1, 1, n, n)
    y = torch.full_like(x, c)
    n_coeff = 1 * n * (n // 2 + 1)
    expected = c * n / n_coeff
    got = spectrum_loss(x, y, mode="magnitude").item()
    assert got == pytest.approx(expected, rel=1e-4), f"{got} vs {expected}"


def test_broadband_loss_is_scale_free() -> None:
    """The same perturbation must score the same at any patch size.

    This is what keeps the term's weight meaningful between 128px training crops
    and full-resolution evaluation. Natural images are broadband, so this is the
    regime that matters.
    """
    torch.manual_seed(0)
    vals = []
    for n in (32, 64, 128):
        a = torch.rand(1, 3, n, n)
        vals.append(spectrum_loss(a, a + 0.1 * torch.randn_like(a),
                                  mode="complex").item())
    assert max(vals) / min(vals) < 1.05, f"scale-dependent: {vals}"


def test_sparse_spectrum_is_NOT_scale_free_and_that_is_expected() -> None:
    """A single sinusoid gives loss proportional to 1/N — pinned deliberately.

    Energy sits in a fixed number of coefficients while the mean is taken over a
    count growing as N^2. The first version of the module docstring claimed
    unconditional scale-invariance; this test is what disproved it. Pinned so the
    qualifier cannot quietly be dropped again.
    """
    prod = [spectrum_loss(_sinusoid(n, n // 10),
                          _sinusoid(n, n // 10, math.pi),
                          mode="complex").item() * n
            for n in (32, 64, 128)]
    assert max(prod) / min(prod) < 1.10, f"loss*N not constant: {prod}"


def test_gradient_flows_to_the_prediction_only() -> None:
    pred = torch.rand(1, 3, 16, 16, requires_grad=True)
    target = torch.rand(1, 3, 16, 16)
    spectrum_loss(pred, target).backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum() > 0


def test_bfloat16_input_is_handled() -> None:
    """torch.fft has no bfloat16 kernel — the same limitation that forces the
    teacher into fp32 and blocks AdaIR's ONNX export entirely (F7)."""
    a = torch.rand(1, 3, 16, 16, dtype=torch.bfloat16)
    b = torch.rand(1, 3, 16, 16, dtype=torch.bfloat16)
    out = spectrum_loss(a, b)
    assert torch.isfinite(out)


def test_shape_mismatch_and_bad_mode_raise() -> None:
    x, y = torch.rand(1, 3, 16, 16), torch.rand(1, 3, 8, 8)
    with pytest.raises(ValueError, match="shape mismatch"):
        spectrum_loss(x, y)
    with pytest.raises(ValueError, match="must be 'magnitude' or 'complex'"):
        spectrum_loss(x, x, mode="amplitude")


def test_a_worse_prediction_scores_higher() -> None:
    """Monotonicity: the loss must order predictions sensibly, or it cannot
    train anything regardless of what the unit values say."""
    target = torch.rand(1, 3, 32, 32)
    close = target + 0.01 * torch.randn_like(target)
    far = target + 0.20 * torch.randn_like(target)
    assert spectrum_loss(close, target) < spectrum_loss(far, target)


# ------------------------------------------------ the arm it will be used by


def test_freq_config_differs_from_response_kd_by_exactly_two_keys() -> None:
    """The three-way comparison is attributable only if each step adds one thing.

    GT-only -> +response KD -> +frequency term. If the frequency config differed
    anywhere else, its delta against M-DEHAZE-KD would not be the frequency
    term's.
    """
    import yaml

    from src.utils.config import REPO_ROOT

    kd = yaml.safe_load((REPO_ROOT / "configs/train/m_dehaze_kd.yaml")
                        .read_text(encoding="utf-8"))
    fq = yaml.safe_load((REPO_ROOT / "configs/train/m_dehaze_kd_freq.yaml")
                        .read_text(encoding="utf-8"))
    for section in ("arch", "data", "optim", "schedule", "train", "loss", "eval"):
        assert kd[section] == fq[section], f"section {section!r} differs"
    assert set(fq) == set(kd), f"top-level keys differ: {set(fq) ^ set(kd)}"

    added = {k: v for k, v in fq["distill"].items() if k not in kd["distill"]}
    assert set(added) == {"freq_weight", "freq_mode"}, added
    assert added["freq_weight"] > 0
    for k, v in kd["distill"].items():
        assert fq["distill"][k] == v, f"distill.{k} changed: {v} -> {fq['distill'][k]}"


def test_no_other_arm_enables_the_frequency_term() -> None:
    """Every other config must leave freq_weight absent or zero, or its result
    would silently include a term it does not claim."""
    import yaml

    from src.utils.config import REPO_ROOT

    for p in sorted((REPO_ROOT / "configs" / "train").glob("*.yaml")):
        if p.name == "m_dehaze_kd_freq.yaml":
            continue
        cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        assert not (cfg.get("distill") or {}).get("freq_weight"), \
            f"{p.name} enables the frequency term"


def test_frequency_term_without_a_teacher_is_rejected(tmp_path) -> None:
    """No silent fallback: the term compares against the TEACHER's spectrum."""
    import torch as _torch

    from src.models.nafnet import NAFNet
    from src.train.trainer import Trainer

    model = NAFNet(width=4, enc_blk_nums=[1], middle_blk_num=1, dec_blk_nums=[1])
    batches = [(_torch.rand(1, 3, 16, 16), _torch.rand(1, 3, 16, 16), 15)]
    cfg = {"optim": {"lr": 1e-3}, "schedule": {"total_iters": 1},
           "train": {"accum_steps": 1, "amp": False},
           "loss": {"name": "charbonnier"},
           "distill": {"freq_weight": 0.2}}
    with pytest.raises(ValueError, match="without distill.teacher"):
        Trainer(model, batches, cfg, tmp_path, device="cpu")
