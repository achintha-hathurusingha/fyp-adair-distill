"""Tests for the normalization variants (Task 1.5c).

The central test is :func:`test_rsqrt_variant_is_numerically_identical` — if
N-A' computes the same function as N-A, swapping them is a pure graph rewrite
requiring no retraining and no quality ablation.
"""
from __future__ import annotations

import pytest
import torch

from src.models.nafnet import NAFNet
from src.models.norms import (NORM_TYPES, AffineNorm2d, IdentityNorm2d,
                              LayerNorm2d, LayerNorm2dRsqrt, build_norm)


# --------------------------------------------------------------------------
# N-A' equivalence: the reason it goes first
# --------------------------------------------------------------------------

def test_rsqrt_variant_is_numerically_identical() -> None:
    """N-A' must match N-A to floating-point tolerance, for shared weights."""
    torch.manual_seed(0)
    a = LayerNorm2d(32).eval()
    b = LayerNorm2dRsqrt(32).eval()
    b.load_state_dict(a.state_dict())  # identical parameters

    x = torch.randn(4, 32, 16, 16)
    with torch.no_grad():
        assert torch.allclose(a(x), b(x), atol=1e-5)


def test_rsqrt_variant_matches_with_trained_affine_params() -> None:
    """Equivalence must hold for non-default weight/bias, not just ones/zeros."""
    torch.manual_seed(1)
    a = LayerNorm2d(16).eval()
    with torch.no_grad():
        a.weight.copy_(torch.randn(16))
        a.bias.copy_(torch.randn(16))
    b = LayerNorm2dRsqrt(16).eval()
    b.load_state_dict(a.state_dict())

    x = torch.randn(2, 16, 8, 8) * 5.0 + 3.0
    with torch.no_grad():
        assert torch.allclose(a(x), b(x), atol=1e-5)


def test_rsqrt_state_dict_is_interchangeable() -> None:
    """A checkpoint trained with N-A must load into N-A' unchanged."""
    a = LayerNorm2d(8)
    b = LayerNorm2dRsqrt(8)
    missing, unexpected = b.load_state_dict(a.state_dict(), strict=True), None
    assert unexpected is None
    assert set(a.state_dict()) == set(b.state_dict())


def test_full_model_rsqrt_swap_is_equivalent() -> None:
    """Whole-network equivalence, not just the isolated module."""
    torch.manual_seed(2)
    kw = dict(width=8, enc_blk_nums=[1, 1], middle_blk_num=1, dec_blk_nums=[1, 1])
    a = NAFNet(**kw, norm_type="layernorm2d").eval()
    b = NAFNet(**kw, norm_type="layernorm2d_rsqrt").eval()
    b.load_state_dict(a.state_dict())

    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        assert torch.allclose(a(x), b(x), atol=1e-5)


def test_rsqrt_variant_exports_without_div() -> None:
    """The entire point of N-A': the exported graph must contain no `Div`.

    Regression guard for a real trap — ``torch.rsqrt`` lowers to
    ``Sqrt`` + ``Div(1, .)`` (verified at opset 17 and 20), which is strictly
    worse than N-A. Only ``torch.reciprocal(torch.sqrt(.))`` reaches
    ``Reciprocal``. If someone "simplifies" the implementation back to
    ``torch.rsqrt``, this test fails.
    """
    import tempfile
    from pathlib import Path

    from src.export.op_coverage import op_histogram
    from src.export.to_onnx import export_onnx

    with tempfile.TemporaryDirectory() as td:
        path = export_onnx(LayerNorm2dRsqrt(32).eval(),
                           Path(td) / "narsqrt.onnx", (1, 32, 32, 32))
        hist = op_histogram(path)
    assert hist.get("Div", 0) == 0, f"N-A' must emit no Div, got {dict(hist)}"
    assert hist.get("Reciprocal", 0) == 1, \
        f"N-A' must emit exactly one Reciprocal, got {dict(hist)}"


def test_reference_layernorm_does_emit_div() -> None:
    """Contrast case: N-A is the thing N-A' is trying to improve on."""
    import tempfile
    from pathlib import Path

    from src.export.op_coverage import op_histogram
    from src.export.to_onnx import export_onnx

    with tempfile.TemporaryDirectory() as td:
        path = export_onnx(LayerNorm2d(32).eval(),
                           Path(td) / "na.onnx", (1, 32, 32, 32))
        hist = op_histogram(path)
    assert hist.get("Div", 0) == 1


# --------------------------------------------------------------------------
# registry behaviour
# --------------------------------------------------------------------------

@pytest.mark.parametrize("norm_type", NORM_TYPES)
def test_every_norm_type_builds_and_preserves_shape(norm_type: str) -> None:
    norm = build_norm(norm_type, 16).eval()
    x = torch.randn(2, 16, 8, 8)
    with torch.no_grad():
        assert norm(x).shape == x.shape


def test_unknown_norm_type_raises() -> None:
    """No silent fallback to a default normalisation (rule 9)."""
    with pytest.raises(ValueError, match="Unknown norm_type"):
        build_norm("groupnorm", 16)


def test_identity_norm_is_a_passthrough() -> None:
    x = torch.randn(2, 8, 4, 4)
    assert torch.equal(IdentityNorm2d(8)(x), x)


def test_affine_norm_computes_no_statistics() -> None:
    """N-E must be a pure per-channel affine: shifting input shifts output."""
    norm = AffineNorm2d(4).eval()
    x = torch.randn(1, 4, 4, 4)
    with torch.no_grad():
        # A pure affine is equivariant to input scaling; a normaliser is not.
        assert torch.allclose(norm(2 * x), 2 * norm(x) - norm.bias.view(1, -1, 1, 1),
                              atol=1e-5)


# --------------------------------------------------------------------------
# N-F: resolution-selective normalisation
# --------------------------------------------------------------------------

def test_full_res_override_applies_only_to_level_zero() -> None:
    """N-F must swap the full-resolution norms and leave deeper stages alone."""
    model = NAFNet(width=8, enc_blk_nums=[1, 1, 1], middle_blk_num=1,
                   dec_blk_nums=[1, 1, 1],
                   norm_type="layernorm2d", full_res_norm_type="affine")
    # encoder level 0 == full resolution -> overridden
    assert isinstance(model.encoders[0][0].norm1, AffineNorm2d)
    # deeper encoder stages keep the reference norm
    assert isinstance(model.encoders[1][0].norm1, LayerNorm2d)
    assert isinstance(model.encoders[2][0].norm1, LayerNorm2d)
    # last decoder stage is back at full resolution -> overridden
    assert isinstance(model.decoders[-1][0].norm1, AffineNorm2d)
    # earlier decoder stages are not
    assert isinstance(model.decoders[0][0].norm1, LayerNorm2d)
    # bottleneck is deepest, never overridden
    assert isinstance(model.middle_blks[0].norm1, LayerNorm2d)


def test_default_model_uses_layernorm() -> None:
    """Absent configuration, behaviour is unchanged from the G1 baseline."""
    model = NAFNet(width=8, enc_blk_nums=[1, 1], middle_blk_num=1,
                   dec_blk_nums=[1, 1])
    assert isinstance(model.encoders[0][0].norm1, LayerNorm2d)
