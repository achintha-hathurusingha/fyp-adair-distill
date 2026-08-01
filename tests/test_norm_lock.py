"""Pin the LOCKED normalization variant (findings F9).

Plain N-F killed B0 at iteration 24356: the full-resolution decoder stage has no
normalisation to bound a large-but-finite input, and one rare low-variance crop
drove it to max|a| 5.6e6 and a gradient norm of 6.5e7. The lock is now
`affine_clamp` at bound 8.0, validated over 50k live iterations through both
original divergence points, with a pre-committed Mann-Kendall test showing no
trend in engagement (p=0.27) or pre-clamp magnitude (p=0.71).

These tests exist so the clamp cannot be silently dropped — the failure it
prevents takes ~24k iterations and a 1-in-thousands sample to reappear, so it
would not be caught by anything short.
"""
from __future__ import annotations

import torch

from src.models.nafnet import NAFNet
from src.models.norms import AffineClampNorm2d, LayerNorm2d
from src.train.train import build_config, build_model
from src.utils.config import load_yaml

LOCKED = load_yaml("configs/model/nafnet_locked.yaml")


def test_locked_config_specifies_the_clamp() -> None:
    assert LOCKED["norm_type"] == "layernorm2d"
    assert LOCKED["full_res_norm_type"] == "affine_clamp", (
        "full-resolution stages must use affine_clamp; plain affine is what "
        "killed B0 (findings F9)")
    assert LOCKED["clamp_bound"] == 8.0


def test_b0_builds_with_the_clamp_active() -> None:
    """The run that diverged must not be reproducible from the B0 arm."""
    cfg = build_config("B0", 0, 0, 0.0, 0)
    assert cfg["model"]["full_res_norm_type"] == "affine_clamp"
    assert cfg["model"]["clamp_bound"] == 8.0


def test_full_resolution_stages_are_clamped_and_deep_stages_are_not() -> None:
    """The clamp belongs ONLY at full resolution — that is the whole point.

    Clamping deeper stages would be an unmeasured change; N-F's speedup comes
    from full-resolution norms specifically, because normalisation cost is
    per-element (findings F1).
    """
    model = NAFNet(width=16, enc_blk_nums=[2, 2, 4, 8], middle_blk_num=12,
                   dec_blk_nums=[2, 2, 2, 2],
                   norm_type=LOCKED["norm_type"],
                   full_res_norm_type=LOCKED["full_res_norm_type"],
                   clamp_bound=LOCKED["clamp_bound"])

    clamps = [m for m in model.modules() if isinstance(m, AffineClampNorm2d)]
    layernorms = [m for m in model.modules() if isinstance(m, LayerNorm2d)]
    assert clamps, "no clamp modules built"
    assert layernorms, "deeper stages lost their LayerNorm"
    assert all(m.bound == 8.0 for m in clamps)

    # enc0 has 2 blocks, dec3 has 2 blocks, 2 norms each => 8 clamped norms.
    assert len(clamps) == 8, f"expected 8 clamped norms, got {len(clamps)}"


def test_clamp_bounds_a_pathological_activation() -> None:
    """The behaviour the lock exists for, asserted directly."""
    n = AffineClampNorm2d(4, bound=8.0)
    out = n(torch.full((1, 4, 4, 4), 7.05e5))     # the B0 failure magnitude
    assert float(out.abs().max()) == 8.0


def test_clamp_is_inert_on_healthy_activations() -> None:
    """It must not act as an undocumented activation function."""
    n = AffineClampNorm2d(4, bound=8.0)
    x = torch.randn(2, 4, 8, 8)                   # ~N(0,1), well inside
    assert torch.allclose(n(x), x, atol=1e-6)


def test_clamp_weights_stay_compatible_with_the_other_variants() -> None:
    """State-dict compatibility is what made the fix comparison possible.

    A buffer or extra parameter here would break loading a plain-N-F checkpoint
    into the clamped model, which is how Fix-C was validated in the first place.
    """
    shapes = lambda m: {k: tuple(v.shape) for k, v in m.state_dict().items()}
    assert shapes(AffineClampNorm2d(8)) == shapes(LayerNorm2d(8))


def test_locked_model_forward_is_finite_on_adversarial_input() -> None:
    cfg = build_config("B0", 0, 0, 0.0, 0)
    model = build_model(cfg).eval()
    for name, x in (("all-black", torch.zeros(1, 3, 64, 64)),
                    ("all-white", torch.ones(1, 3, 64, 64)),
                    ("low-var", torch.full((1, 3, 64, 64), 0.055)
                     + torch.randn(1, 3, 64, 64) * 0.084)):
        with torch.no_grad():
            out = model(x)
        assert torch.isfinite(out).all(), f"non-finite output on {name}"
