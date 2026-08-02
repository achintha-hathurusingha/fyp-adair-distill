"""Deep-stage clamp insurance at enc3 (F10).

Two properties matter and they pull against each other: the clamp must be
placed where it was asked for, and it must be INERT — a clamp that engages in
normal training is not insurance, it is an undocumented activation function
quietly reshaping the network.
"""
from __future__ import annotations

import pytest
import torch

from src.models import norms
from src.models.nafnet import NAFNet

LOCKED = dict(width=16, enc_blk_nums=[2, 2, 4, 8], middle_blk_num=12,
              dec_blk_nums=[2, 2, 2, 2], norm_type="layernorm2d",
              full_res_norm_type="affine_clamp", clamp_bound=8.0)


def _model(**kw):
    return NAFNet(**{**LOCKED, **kw})


# --------------------------------------------------------------- placement


def test_clamp_lands_on_the_named_stage_only() -> None:
    m = _model(enc_clamp_stages=[3])
    kinds = [type(enc[0].norm1).__name__ for enc in m.encoders]
    assert kinds == ["AffineClampNorm2d", "LayerNorm2d", "LayerNorm2d",
                     "LayerNorm2dClamp"], kinds
    assert type(m.middle_blks[0].norm1).__name__ == "LayerNorm2d"


def test_both_norms_in_every_block_of_the_stage_are_clamped() -> None:
    m = _model(enc_clamp_stages=[3])
    for blk in m.encoders[3]:
        assert isinstance(blk.norm1, norms.LayerNorm2dClamp)
        assert isinstance(blk.norm2, norms.LayerNorm2dClamp)


def test_decoder_is_untouched() -> None:
    """dec3's exposure is already covered by full_res_norm_type (F9)."""
    m = _model(enc_clamp_stages=[3])
    assert type(m.decoders[-1][0].norm1).__name__ == "AffineClampNorm2d"
    for dec in m.decoders[:-1]:
        assert type(dec[0].norm1).__name__ == "LayerNorm2d"


def test_no_stages_named_means_no_clamp_anywhere() -> None:
    m = _model()
    assert not any(isinstance(mod, norms.LayerNorm2dClamp) for mod in m.modules())


def test_multiple_stages_can_be_clamped() -> None:
    m = _model(enc_clamp_stages=[2, 3])
    assert isinstance(m.encoders[2][0].norm1, norms.LayerNorm2dClamp)
    assert isinstance(m.encoders[3][0].norm1, norms.LayerNorm2dClamp)


# --------------------------------------------------------------- weights


def test_state_dict_is_identical_to_the_unclamped_model() -> None:
    """A B0-denoise checkpoint must load straight into the clamped model.

    The clamp adds no parameters, so the two are weight-compatible — which is
    what makes a with/without comparison possible on identical weights.
    """
    plain, clamped = _model(), _model(enc_clamp_stages=[3])
    shapes = lambda m: {k: tuple(v.shape) for k, v in m.state_dict().items()}
    assert shapes(plain) == shapes(clamped)
    assert sum(p.numel() for p in plain.parameters()) == \
        sum(p.numel() for p in clamped.parameters()) == 7_371_923
    clamped.load_state_dict(plain.state_dict())     # raises if incompatible


def test_clamped_model_matches_the_plain_one_while_inert() -> None:
    """With the bound above every activation, output must be bit-identical.

    Residual scales perturbed for the same reason as the test below: at init
    they are zero and the stage is an exact identity, which would make this
    pass without proving anything.
    """
    torch.manual_seed(0)
    plain = _model()
    with torch.no_grad():
        for blk in plain.encoders[3]:
            blk.beta.fill_(0.5)
            blk.gamma.fill_(0.5)
    clamped = _model(enc_clamp_stages=[3], deep_clamp_bound=1e9)
    clamped.load_state_dict(plain.state_dict())
    plain.eval(); clamped.eval()
    x = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        assert torch.equal(plain(x), clamped(x)), "an inert clamp changed the output"


def test_a_tight_bound_does_change_the_output() -> None:
    """Guard against the clamp being wired up but never applied.

    NAFBlock's residual scales `beta` and `gamma` are zero-initialised, so a
    freshly built block is an EXACT identity and nothing its norms compute can
    reach the output — this test passes vacuously without the perturbation
    below. The scales are non-zero in any trained model, so setting them is what
    makes this exercise the real path.
    """
    torch.manual_seed(0)
    plain = _model()
    with torch.no_grad():
        for blk in plain.encoders[3]:
            blk.beta.fill_(0.5)
            blk.gamma.fill_(0.5)
    tight = _model(enc_clamp_stages=[3], deep_clamp_bound=1e-3)
    tight.load_state_dict(plain.state_dict())
    plain.eval(); tight.eval()
    x = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        assert not torch.equal(plain(x), tight(x)), "the clamp is not in the path"


# --------------------------------------------------------------- the module


def test_bound_is_applied_and_defaults_to_the_module_constant() -> None:
    n = norms.build_norm("layernorm2d_clamp", 4)
    assert n.bound == norms.DEEP_CLAMP_BOUND == 32.0
    out = n(torch.randn(2, 4, 8, 8) * 1e6)
    assert float(out.abs().max()) <= 32.0


def test_bound_is_read_at_construction_not_import() -> None:
    """Same trap that made an earlier bound sweep return identical results."""
    original = norms.DEEP_CLAMP_BOUND
    try:
        norms.DEEP_CLAMP_BOUND = 3.0
        assert norms.build_norm("layernorm2d_clamp", 4).bound == 3.0
    finally:
        norms.DEEP_CLAMP_BOUND = original
    assert norms.build_norm("layernorm2d_clamp", 4).bound == original


def test_the_two_bounds_are_separate_keys() -> None:
    """affine_clamp and layernorm2d_clamp bound different quantities.

    One shared key would silently retune one of them whenever the other changed.
    """
    n = norms.build_norm("layernorm2d_clamp", 4, clamp_bound=8.0)
    assert n.bound == norms.DEEP_CLAMP_BOUND, "took the affine_clamp bound"
    a = norms.build_norm("affine_clamp", 4, deep_clamp_bound=99.0)
    assert a.bound == norms.AFFINE_CLAMP_BOUND, "took the deep bound"


def test_normalisation_still_happens_before_the_clamp() -> None:
    n = norms.build_norm("layernorm2d_clamp", 8, deep_clamp_bound=1e9)
    out = n(torch.randn(2, 8, 4, 4) * 100 + 50)
    assert abs(float(out.mean())) < 1e-4, "output is not zero-mean"


# --------------------------------------------------------------- failing loudly


def test_stage_index_out_of_range_is_rejected() -> None:
    with pytest.raises(ValueError, match="only 4 encoder stages"):
        _model(enc_clamp_stages=[4])


def test_clamping_stage_0_conflicts_with_full_res_override() -> None:
    """Stage 0 already carries affine_clamp; silently discarding that would
    change the F9 fix without saying so."""
    with pytest.raises(ValueError, match="already uses full_res_norm_type"):
        _model(enc_clamp_stages=[0])


def test_unknown_norm_type_still_raises() -> None:
    with pytest.raises(ValueError, match="Unknown norm_type"):
        norms.build_norm("layernorm2d_clamped", 4)
