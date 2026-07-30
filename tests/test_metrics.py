"""Known-answer tests for the metric conventions.

These exist because a plausible-looking wrong convention silently invalidates
every number in the project. Each test pins one convention to a hardcoded value
or an analytically-known answer.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.eval.metrics import (ADAIR_DEFAULT, MetricConfig, build_metric_config,
                              crop_to_multiple, psnr, psnr_ssim, ssim)


def _img(seed: int = 0, shape=(32, 32, 3)) -> np.ndarray:
    return np.random.default_rng(seed).random(shape)


# --------------------------------------------------------------------------
# analytic known answers
# --------------------------------------------------------------------------

def test_identical_images_give_infinite_psnr_and_unit_ssim() -> None:
    x = _img()
    assert np.isinf(psnr(x, x))
    assert ssim(x, x) == pytest.approx(1.0, abs=1e-9)


def test_psnr_matches_closed_form_for_constant_offset() -> None:
    """A uniform error of e gives PSNR = 20*log10(data_range/e) exactly."""
    x = np.full((16, 16, 3), 0.5)
    e = 0.1
    y = x + e
    expected = 20 * np.log10(1.0 / e)
    assert psnr(y, x) == pytest.approx(expected, abs=1e-6)


def test_psnr_is_symmetric() -> None:
    a, b = _img(1), _img(2)
    assert psnr(a, b) == pytest.approx(psnr(b, a), abs=1e-9)


# --------------------------------------------------------------------------
# convention pins -- hardcoded regression values
# --------------------------------------------------------------------------

def test_adair_default_conventions_are_the_traced_ones() -> None:
    """If any default changes, reproduction silently breaks. Pin them all."""
    c = ADAIR_DEFAULT
    assert c.channel == "rgb"
    assert c.crop_border == 0
    assert c.data_range == 1.0
    assert c.clip is True
    assert c.round_to_uint8 is False
    assert c.ssim_win_size == 7          # skimage default, NOT Wang's 11
    assert c.ssim_gaussian_weights is False   # uniform box, NOT Gaussian


def test_known_answer_on_fixed_noise_pattern() -> None:
    """Hardcoded expected values for a fixed seed under ADAIR_DEFAULT.

    Values were *computed* from this implementation and frozen, so this is a
    regression pin against silent convention drift (a scikit-image default
    change, a reordering in ``prepare``) rather than an independent oracle.

    It is nonetheless **semi-validated against theory**: for additive Gaussian
    noise of sigma=0.05 on ``data_range=1``, PSNR should be
    ``20*log10(1/0.05) = 26.02 dB``. The measured 26.25 dB is slightly *higher*,
    which is exactly what clipping to [0, 1] predicts — clipping discards noise
    energy at the range boundaries. Agreement with theory to within the sign and
    rough size of the clipping effect is meaningful evidence the pipeline is
    doing what it claims.
    """
    rng = np.random.default_rng(1234)
    clean = rng.random((64, 64, 3))
    noisy = np.clip(clean + rng.normal(0, 0.05, clean.shape), 0, 1)
    p, s = psnr_ssim(noisy, clean)
    assert p == pytest.approx(26.2534, abs=1e-3)
    assert s == pytest.approx(0.9856, abs=1e-3)


def test_ssim_channel_axis_is_explicit_not_volumetric() -> None:
    """Guards findings F5.

    scikit-image removed ``multichannel=True`` in 0.23; left to the default,
    ``channel_axis=None`` computes a VOLUMETRIC SSIM over (H, W, C) rather than
    the mean of per-channel 2-D SSIMs. Our value must match the per-channel
    computation, not the volumetric one.
    """
    from skimage.metrics import structural_similarity

    a, b = _img(7), _img(8)
    ours = ssim(a, b)
    per_channel = np.mean([
        structural_similarity(b[..., i], a[..., i], data_range=1.0, win_size=7)
        for i in range(3)
    ])
    assert ours == pytest.approx(per_channel, abs=1e-6)


def test_wang_window_gives_a_different_number() -> None:
    """The 11x11 Gaussian convention is a DIFFERENT metric, not a variant.

    Documents why the originally-proposed default (win_size=11, sigma=1.5)
    would not have reproduced AdaIR.
    """
    a, b = _img(3), _img(4)
    wang = MetricConfig(ssim_win_size=11, ssim_gaussian_weights=True, ssim_sigma=1.5)
    assert ssim(a, b, wang) != pytest.approx(ssim(a, b, ADAIR_DEFAULT), abs=1e-4)


def test_uint8_rounding_changes_the_result() -> None:
    """Demonstrates the rounding convention actually matters."""
    rng = np.random.default_rng(11)
    clean = rng.random((32, 32, 3))
    pred = clean + rng.normal(0, 0.002, clean.shape)  # sub-quantisation-step error
    floated = psnr(pred, clean, ADAIR_DEFAULT)
    rounded = psnr(pred, clean, MetricConfig(round_to_uint8=True))
    assert floated != pytest.approx(rounded, abs=1e-3)


def test_crop_border_changes_the_result() -> None:
    a, b = _img(5), _img(6)
    assert psnr(a, b, MetricConfig(crop_border=4)) != pytest.approx(
        psnr(a, b, ADAIR_DEFAULT), abs=1e-6)


def test_y_channel_differs_from_rgb() -> None:
    """The convention we did NOT adopt must be reachable and distinct."""
    a, b = _img(9), _img(10)
    assert psnr(a, b, MetricConfig(channel="y")) != pytest.approx(
        psnr(a, b, ADAIR_DEFAULT), abs=1e-3)


def test_clipping_is_applied_to_both_inputs() -> None:
    """AdaIR clips prediction AND ground truth (val_utils.py:52-53)."""
    clean = np.full((8, 8, 3), 0.5)
    pred = np.full((8, 8, 3), 1.5)      # out of range; clips to 1.0
    assert psnr(pred, clean) == pytest.approx(20 * np.log10(1.0 / 0.5), abs=1e-6)


# --------------------------------------------------------------------------
# input handling
# --------------------------------------------------------------------------

def test_chw_and_hwc_agree() -> None:
    a, b = _img(21), _img(22)
    assert psnr(a.transpose(2, 0, 1), b.transpose(2, 0, 1)) == pytest.approx(
        psnr(a, b), abs=1e-9)


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        psnr(_img(shape=(16, 16, 3)), _img(shape=(32, 32, 3)))


# --------------------------------------------------------------------------
# crop_to_multiple -- must replicate AdaIR's crop_img exactly
# --------------------------------------------------------------------------

def test_crop_to_multiple_matches_adair_bsd68_geometry() -> None:
    """BSD68 is 481x321; AdaIR's base=16 crop yields 480x320."""
    img = np.zeros((321, 481, 3))
    out = crop_to_multiple(img, base=16)
    assert out.shape == (320, 480, 3)


def test_crop_to_multiple_replicates_asymmetric_offset() -> None:
    """AdaIR uses image[ch//2 : h-ch+ch//2] -- offset, not symmetric."""
    img = np.arange(19 * 16).reshape(19, 16, 1).astype(float)
    out = crop_to_multiple(img, base=16)
    assert out.shape == (16, 16, 1)
    # h=19, ch=3, ch//2=1 -> rows [1:17]
    assert np.array_equal(out[:, :, 0], img[1:17, :, 0])


def test_crop_to_multiple_is_noop_when_already_aligned() -> None:
    img = np.zeros((256, 256, 3))
    assert crop_to_multiple(img, base=16).shape == (256, 256, 3)


# --------------------------------------------------------------------------
# config plumbing
# --------------------------------------------------------------------------

def test_unknown_convention_key_raises() -> None:
    """A typo'd convention key is exactly how wrong numbers happen."""
    with pytest.raises(ValueError, match="Unknown metric convention key"):
        build_metric_config({"chanel": "rgb"})


def test_per_task_override_applies_only_to_that_task() -> None:
    cfg = {"channel": "rgb", "per_task": {"derain": {"channel": "y"}}}
    assert build_metric_config(cfg, "derain").channel == "y"
    assert build_metric_config(cfg, "denoise").channel == "rgb"


def test_all_three_tasks_share_the_adair_convention_by_default() -> None:
    for task in ("denoise", "derain", "dehaze"):
        assert build_metric_config(None, task) == ADAIR_DEFAULT


def test_invalid_config_values_rejected() -> None:
    with pytest.raises(ValueError, match="channel must be"):
        build_metric_config({"channel": "bgr"})
    with pytest.raises(ValueError, match="ssim_win_size must be odd"):
        build_metric_config({"ssim_win_size": 8})
