"""PSNR and SSIM with every convention explicit and configurable.

Metric conventions are the single most common source of irreproducible numbers
in image restoration. Every convention below is a config key, and every default
is what **AdaIR's own evaluation code literally does** — traced in
``reports/eval_conventions.md`` against commit ``ccb8b98``, not inferred from the
paper.

Defaults, and their source (all three tasks share ONE convention; AdaIR uses a
single ``compute_psnr_ssim`` for denoise, derain and dehaze):

===========================  ==========================================  ==================================
convention                   default                                     source
===========================  ==========================================  ==================================
channel basis                RGB, all 3 channels (never Y/luma)          ``utils/val_utils.py:61-62``
border crop                  0 px                                        ``utils/val_utils.py:50-64``
data range                   1.0 (inputs in ``[0, 1]``)                  ``utils/val_utils.py:61-62``
clipping                     clip BOTH pred and GT to ``[0, 1]``         ``utils/val_utils.py:52-53``
uint8 rounding               **no** — compared as float                  ``utils/val_utils.py:50-64``
save-then-load               **no** — computed in memory before PNG      ``test.py:64`` then ``:68``
SSIM implementation          ``skimage.metrics.structural_similarity``   ``utils/val_utils.py:4,62``
SSIM window                  7, **uniform box** filter                   skimage default (NOT Wang 11x11)
SSIM gaussian_weights        False                                       skimage default
SSIM channel handling        ``channel_axis=-1``, set EXPLICITLY         see below
===========================  ==========================================  ==================================

**Two traps, both deliberate choices here:**

* ``win_size=7`` with a uniform window is skimage's default, and is what AdaIR
  gets. The classic Wang et al. convention (11x11 Gaussian, sigma=1.5) is a
  *different metric* and does not reproduce AdaIR's published numbers.
* AdaIR passes ``multichannel=True``, which scikit-image **removed in 0.23**.
  On modern versions it is silently swallowed by ``**kwargs``, leaving
  ``channel_axis=None`` and yielding a volumetric SSIM. We therefore pass
  ``channel_axis`` **explicitly** and pin it with a known-answer test. See
  ``reports/findings.md`` F5.

Per-task overrides are supported but unnecessary for AdaIR: the mechanism exists
so we can compare against Y-channel-reporting baselines (e.g. MPRNet) later
without touching this module.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

#: Rec.601 luma coefficients, matching the YCbCr convention used by the
#: MATLAB ``rgb2ycbcr`` that restoration papers inherit. Only used when
#: ``channel="y"``; AdaIR never takes this path.
_Y_COEFFS_601 = np.array([65.481, 128.553, 24.966], dtype=np.float64)
_Y_OFFSET_601 = 16.0


@dataclass(frozen=True)
class MetricConfig:
    """Every metric convention, explicit. Defaults reproduce AdaIR."""

    channel: str = "rgb"                 # "rgb" | "y"
    crop_border: int = 0                 # pixels removed from each edge
    data_range: float = 1.0              # 1.0 for [0,1] float, 255 for uint8
    clip: bool = True                    # clip both inputs to [0, data_range]
    round_to_uint8: bool = False         # quantise before comparison
    ssim_win_size: int = 7               # skimage default; NOT Wang's 11
    ssim_gaussian_weights: bool = False  # skimage default; NOT Wang's Gaussian
    ssim_sigma: float = 1.5              # only used if gaussian_weights=True

    def validate(self) -> None:
        """Reject nonsense configurations loudly (rule 9)."""
        if self.channel not in ("rgb", "y"):
            raise ValueError(f"channel must be 'rgb' or 'y', got {self.channel!r}")
        if self.crop_border < 0:
            raise ValueError(f"crop_border must be >= 0, got {self.crop_border}")
        if self.data_range <= 0:
            raise ValueError(f"data_range must be > 0, got {self.data_range}")
        if self.ssim_win_size % 2 == 0:
            raise ValueError(
                f"ssim_win_size must be odd, got {self.ssim_win_size}")


#: The AdaIR-reproducing configuration. Locked once G3 passes; do not edit.
ADAIR_DEFAULT = MetricConfig()


def build_metric_config(cfg: dict | None, task: str | None = None) -> MetricConfig:
    """Build a :class:`MetricConfig`, applying an optional per-task override.

    Args:
        cfg: mapping with optional top-level convention keys and an optional
            ``per_task`` mapping of task name -> override keys.
        task: task name whose overrides should be applied, if any.

    Raises:
        ValueError: on an unknown convention key — never silently ignored,
            because a typo'd convention key is exactly how wrong numbers happen.
    """
    if not cfg:
        return ADAIR_DEFAULT

    known = set(MetricConfig.__dataclass_fields__)
    base = {k: v for k, v in cfg.items() if k != "per_task"}
    unknown = set(base) - known
    if unknown:
        raise ValueError(
            f"Unknown metric convention key(s): {sorted(unknown)}. "
            f"Known keys: {sorted(known)}")

    config = replace(ADAIR_DEFAULT, **base)
    overrides = (cfg.get("per_task") or {}).get(task) if task else None
    if overrides:
        unknown = set(overrides) - known
        if unknown:
            raise ValueError(
                f"Unknown metric key(s) in per_task[{task!r}]: {sorted(unknown)}")
        config = replace(config, **overrides)
    config.validate()
    return config


def _to_hwc(img: np.ndarray) -> np.ndarray:
    """Accept HWC or CHW and return HWC float64."""
    arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = arr.transpose(1, 2, 0)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    return arr.astype(np.float64)


def _rgb_to_y(img: np.ndarray, data_range: float) -> np.ndarray:
    """Rec.601 luma, returned on the same scale as the input.

    Not used by the AdaIR protocol; provided for comparison against
    Y-channel-reporting baselines.
    """
    scaled = img / data_range                      # -> [0, 1]
    y = scaled @ _Y_COEFFS_601 + _Y_OFFSET_601     # -> [16, 235] on 0-255 scale
    return (y / 255.0) * data_range


def prepare(pred: np.ndarray, target: np.ndarray,
            config: MetricConfig) -> tuple[np.ndarray, np.ndarray]:
    """Apply clipping, rounding, cropping and channel conversion, in AdaIR order.

    Order matters and follows ``compute_psnr_ssim``: clip first, then (optional)
    round, then crop, then (optional) channel conversion.
    """
    a, b = _to_hwc(pred), _to_hwc(target)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: pred {a.shape} vs target {b.shape}")

    if config.clip:
        a = np.clip(a, 0.0, config.data_range)
        b = np.clip(b, 0.0, config.data_range)

    if config.round_to_uint8:
        scale = 255.0 / config.data_range
        a = np.round(a * scale).clip(0, 255) / scale
        b = np.round(b * scale).clip(0, 255) / scale

    c = config.crop_border
    if c:
        if min(a.shape[0], a.shape[1]) <= 2 * c:
            raise ValueError(
                f"crop_border={c} removes the entire image of shape {a.shape}")
        a, b = a[c:-c, c:-c, :], b[c:-c, c:-c, :]

    if config.channel == "y":
        a = _rgb_to_y(a, config.data_range)[:, :, None]
        b = _rgb_to_y(b, config.data_range)[:, :, None]
    return a, b


def psnr(pred: np.ndarray, target: np.ndarray,
         config: MetricConfig = ADAIR_DEFAULT) -> float:
    """Peak signal-to-noise ratio in dB. ``inf`` for identical inputs."""
    from skimage.metrics import peak_signal_noise_ratio

    a, b = prepare(pred, target, config)
    return float(peak_signal_noise_ratio(b, a, data_range=config.data_range))


def ssim(pred: np.ndarray, target: np.ndarray,
         config: MetricConfig = ADAIR_DEFAULT) -> float:
    """Structural similarity.

    ``channel_axis`` is always passed explicitly — never left to a default that
    scikit-image has already changed once (see module docstring and findings F5).
    """
    from skimage.metrics import structural_similarity

    a, b = prepare(pred, target, config)
    kwargs = dict(
        data_range=config.data_range,
        win_size=config.ssim_win_size,
        gaussian_weights=config.ssim_gaussian_weights,
        channel_axis=-1,
    )
    if config.ssim_gaussian_weights:
        kwargs["sigma"] = config.ssim_sigma
    return float(structural_similarity(b, a, **kwargs))


def psnr_ssim(pred: np.ndarray, target: np.ndarray,
              config: MetricConfig = ADAIR_DEFAULT) -> tuple[float, float]:
    """Both metrics under one config. Preferred entry point."""
    return psnr(pred, target, config), ssim(pred, target, config)


def crop_to_multiple(img: np.ndarray, base: int = 16) -> np.ndarray:
    """Centre-crop HW to a multiple of ``base`` — AdaIR's ``crop_img``.

    Replicates ``utils/image_utils.py:59-64`` exactly, including its asymmetric
    offset (``crop//2`` from the top/left). This runs on BOTH the degraded input
    and the ground truth before inference, so metrics are computed on the cropped
    region; evaluating at full resolution will not match AdaIR.
    """
    arr = np.asarray(img)
    h, w = arr.shape[0], arr.shape[1]
    ch, cw = h % base, w % base
    return arr[ch // 2:h - ch + ch // 2, cw // 2:w - cw + cw // 2, ...]
