"""Student v3 -- an all-in-one restoration student designed from this
project's OWN measurements, not from a teacher's architecture diagram.

Every design decision below traces to a specific measured finding. Nothing
is included because a paper said it works; nothing is included where our
own baseline already ties.

-----------------------------------------------------------------------
WHY NOT v1/v2 (what changed and why)
-----------------------------------------------------------------------
v1/v2 added modules as opt-in flags on NAFNet to A/B them one at a time --
correct for testing, but it left no coherent architecture. v3 composes the
survivors into one model, and DELETES the one that didn't survive scrutiny.

DELETED: LaplacianFrequencyGate. Its sole justification was AdaIR's Table-7
claim that "frequency mining" is worth +1.58dB. We disproved that claim
directly (reports/student_theory_review/, teacher-experiments/test01,
test05_5): AdaIR's frequency mask is mathematically zero at the resolution
it trains at, so the ablation cannot have been measuring frequency
computation; and when we repaired the mask and retrained on real data with
a control arm, the mask fix was worth ~0.00dB. Two independent SOTA lines
(SFNet/FSNet ICLR'23+TPAMI, EvoIR) have since dropped forward-pass FFT
entirely. Keeping a module whose motivation we ourselves refuted would be
incoherent, so it is not in the default path. It remains importable from
theory_blocks for ablation.

-----------------------------------------------------------------------
DESIGN PRINCIPLE: match the operator to the degradation's structure,
and add NOTHING where the baseline already ties.
-----------------------------------------------------------------------
Measured per-task gap of the current student vs its own GT-only baseline
(reports/kd_feature_multitask/cond_regression.md, real 3-way eval):

    denoise   30.69 vs 30.69   TIE      -> add nothing
    derain    36.07 vs 36.83   -0.76dB  -> oriented operator
    dehaze    34.10 vs 34.65   -0.55dB  -> global operator + physical prior

This is corroborated by an independent controlled backbone study
(arXiv:2310.11881), which ran NAFNet / SwinIR / Restormer on identical
tasks and found NAFNet loses 1.7dB (derain) and 2.9dB (dehaze) to
Restormer while WINNING on deblurring -- attributing it to depthwise
convolution's "relatively weak spatial mapping capability" versus
attention's ability to "handle large-range or even global information."
Denoising it calls architecture-flexible. That is exactly our pattern,
independently reproduced, with a mechanism attached.

  1. DEHAZE -> global + physics.
     Koschmieder's atmospheric scattering model I = J*t + A*(1-t) makes
     haze a smooth, spatially-varying, LOW-frequency field. Two operators:
       - dark_channel_prior (He/Sun/Tang CVPR'09) concatenated as a 4th
         input channel: a zero-learned-parameter, per-pixel transmission
         estimate. The network does not have to discover from data
         something physics already gives in closed form.
       - StripPoolingGate (Hu et al. CVPR'20) at the bottleneck and the
         first decoder stage: pools along one FULL spatial axis at a time,
         which is genuinely global along that axis -- the "large-range
         information" capability the backbone study identifies as missing,
         obtained from AveragePool+Conv instead of attention.

  2. DERAIN -> orientation.
     Rain-layer decomposition (Kang et al. TIP'12; Li et al. CVPR'16)
     models rain as a sparse, directionally-ANISOTROPIC high-frequency
     layer. Every conv in NAFBlock is a square, 4-fold-symmetric kernel --
     structurally the wrong shape for a signal whose defining property is
     its orientation. Freeman & Adelson (TPAMI'91) give the theory: a small
     basis of oriented filters, combined with learned angle-dependent
     weights, synthesizes a response at any orientation. Placed at the two
     HIGHEST-resolution decoder stages, because streaks are 3-10px
     structures that do not survive to the bottleneck.

  3. DENOISE -> nothing.
     We tie the baseline. Adding capacity here would only risk the
     regression v1 already demonstrated (cond_regression.md) and confound
     any later ablation. Restraint is a design decision, recorded as one.

-----------------------------------------------------------------------
NPU CONSTRAINT (hard, empirically enforced -- not assumed)
-----------------------------------------------------------------------
Target backends are QNN (Hexagon), TFLite, TensorRT, checked by real ONNX
export through src/export/op_coverage.py -- this repo's own Gate-G1
methodology, which has already caught three real bugs that pure reasoning
missed (ReduceMin/Neg unsupported; a runtime-shape .expand() lowering to
Equal/Where/ConstantOfShape; adaptive_avg_pool2d with a traced dynamic
size failing export only once embedded in the full model).

BANNED, with evidence:
  * torch.fft  -- absent from every backend table; also now pointless (see
                  DELETED above).
  * attention  -- scripts/probe_mdta.py: MDTA lowers to MatMul/Softmax/
                  ReduceL2, none of which appear in ANY of the three
                  curated tables. Unverified, not proven-bad -- but not
                  worth betting the deployment target on.
  * ReduceMin / Neg / dynamic-shape Expand / adaptive-pool-with-traced-size.

Everything in v3 is built only from ops already SUPPORTED on all three
backends: Conv, AveragePool, MaxPool, Add, Sub, Mul, Sigmoid, ReLU,
GlobalAveragePool, Concat, Split, PixelShuffle(DepthToSpace), Clip.

KNOWN REMAINING RISK (documented, deliberately NOT changed here):
LayerNorm2d decomposes into ReduceMean/Pow/Sqrt/Div -- ~394 CAUTION-flagged
ops in the exported graph, by far the dominant NPU risk left, and this
project's own findings F1 record that normalization (not convolution)
dominates INT8 Hexagon latency. The locked config
(configs/model/nafnet_locked.yaml) already mitigates the worst of it with
affine_clamp at full resolution, a choice earned through documented
divergence debugging (F9). Replacing the remaining deep-stage LayerNorm2d
is the single largest NPU lever left, but it is a TRAINING-STABILITY
change (F6: affine-everywhere diverged in all four variants tried), so it
belongs in its own controlled experiment -- not silently bundled into an
architecture change about restoration quality.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.nafnet import NAFBlock
from src.models.norms import build_norm
from src.models.theory_blocks import (
    OrientedStreakGate,
    StripPoolingGate,
    dark_channel_prior,
)

__all__ = ["StudentV3", "build_student_v3"]


from src.models.reparam_oriented import (  # noqa: E402
    PlainLargeKernelBlock, ReparamOrientedBlock,
)


class StudentV3(nn.Module):
    """U-shaped restoration network: NAFNet skeleton + degradation-matched
    operators placed only where the baseline measurably fails.

    Args mirror NAFNet's locked geometry so the two stay directly
    comparable (same width/blocks/norms => any delta is attributable to
    the added operators, not to a different backbone).

    Toggles exist for ablation, all default ON except the deliberately
    excluded frequency gate; turning all three off must reproduce plain
    NAFNet exactly, which the smoke test asserts.
    """

    def __init__(
        self,
        img_channels: int = 3,
        width: int = 16,
        enc_blk_nums: list[int] | None = None,
        middle_blk_num: int = 12,
        dec_blk_nums: list[int] | None = None,
        *,
        norm_type: str = "layernorm2d",
        full_res_norm_type: str | None = "affine_clamp",
        clamp_bound: float | None = 8.0,
        enc_clamp_stages: list[int] | None = None,
        deep_clamp_bound: float | None = 32.0,
        attn_type: str = "sca",
        # --- degradation-matched operators (see module docstring) ---
        use_dcp_prior: bool = True,
        use_strip_pool: bool = True,
        use_oriented_streak: bool = True,
        mid_strip_every: int | None = None,
        # --- S3.1 reparameterizable oriented block (plan Phase 3) ---
        # Default OFF so every pre-existing arm stays byte-identical; S3.3
        # turns it on and ablates placement one variable at a time.
        use_reparam_oriented: bool = False,
        reparam_k: int = 11,          # fixed by S0.1's oracle ceiling
        reparam_stages: tuple[int, ...] | list[int] | None = None,
        reparam_middle: bool = False,
        # "oriented" = the S3.1 block; "plain" = matched large-kernel CONTROL
        # (same size, same fuse, same deployed cost, no oriented structure).
        # S3.3 needs both or a win cannot be attributed to orientation.
        reparam_variant: str = "oriented",
    ) -> None:
        super().__init__()
        enc_blk_nums = enc_blk_nums or [2, 2, 4, 8]
        dec_blk_nums = dec_blk_nums or [2, 2, 2, 2]
        if len(enc_blk_nums) != len(dec_blk_nums):
            raise ValueError("enc and dec must have equal stage counts, "
                             f"got {len(enc_blk_nums)} vs {len(dec_blk_nums)}")
        self.use_dcp_prior = use_dcp_prior
        self.use_strip_pool = use_strip_pool
        self.use_oriented_streak = use_oriented_streak
        self.use_reparam_oriented = use_reparam_oriented
        self.enc_clamp_stages = tuple(enc_clamp_stages or (3,))

        def stage_norm(stage_idx: int, n_stages: int, *, decoder: bool) -> str:
            """Identical policy to NAFNet's, so norms are not a confound."""
            level = (n_stages - 1 - stage_idx) if decoder else stage_idx
            if level == 0 and full_res_norm_type is not None:
                return full_res_norm_type
            if not decoder and level in self.enc_clamp_stages:
                return "layernorm2d_clamp"
            return norm_type

        blk = lambda c, nt: NAFBlock(  # noqa: E731 -- keeps the stage loops readable
            c, norm_type=nt, attn_type=attn_type,
            clamp_bound=clamp_bound, deep_clamp_bound=deep_clamp_bound)

        # -- stem. +1 channel iff the physical haze prior is concatenated.
        self.intro = nn.Conv2d(img_channels + (1 if use_dcp_prior else 0),
                               width, 3, padding=1)
        self.ending = nn.Conv2d(width, img_channels, 3, padding=1)

        # -- encoder: plain NAFBlocks. Nothing added; denoise already ties.
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        chan = width
        n_stages = len(enc_blk_nums)
        for i, n in enumerate(enc_blk_nums):
            nt = stage_norm(i, n_stages, decoder=False)
            self.encoders.append(nn.Sequential(*[blk(chan, nt) for _ in range(n)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, stride=2))
            chan *= 2

        # -- bottleneck: cheapest, most semantic place for a global operator.
        #
        # mid_strip_every controls HOW MANY global operators the bottleneck
        # gets, and it matters more than it looks:
        #   None (default) -- ONE StripPoolingGate after all middle blocks.
        #                     This is what B0V3 ran; kept as the default so
        #                     that run stays exactly reproducible from its
        #                     recorded commit.
        #   int N          -- interleave a gate after every Nth middle block,
        #                     INSIDE the Sequential. A single zero-init
        #                     residual applied once may simply be too weak to
        #                     move a 12-block bottleneck. Multi-level
        #                     injection is also what the literature does and
        #                     ablates in favour of: PromptIR reports 37.04dB
        #                     multi-level vs 36.76dB single-point on their own
        #                     Rain100L ablation, and AdaIR itself places three
        #                     FreModules rather than one.
        #
        # The gates go INSIDE middle_blks (not after it) deliberately: the
        # trainer attaches its feature-KD hook to model.middle_blks, so
        # keeping them inside preserves KD compatibility -- the hook still
        # sees the whole bottleneck's output either way.
        self.mid_strip_every = mid_strip_every
        if use_strip_pool and mid_strip_every:
            mid = []
            for i in range(middle_blk_num):
                mid.append(blk(chan, norm_type))
                if (i + 1) % mid_strip_every == 0:
                    mid.append(StripPoolingGate(chan))
            self.middle_blks = nn.Sequential(*mid)
            self.mid_strip = None          # already interleaved; no trailing gate
        else:
            self.middle_blks = nn.Sequential(*[blk(chan, norm_type) for _ in range(middle_blk_num)])
            self.mid_strip = StripPoolingGate(chan) if use_strip_pool else None

        # -- decoder. Channel widths mirror the encoder in reverse.
        decoder_channels = []
        c = chan
        for _ in dec_blk_nums:
            c //= 2
            decoder_channels.append(c)

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        chan_d = chan
        for i, n in enumerate(dec_blk_nums):
            nt = stage_norm(i, len(dec_blk_nums), decoder=True)
            self.ups.append(nn.Sequential(
                nn.Conv2d(chan_d, chan_d * 2, 1, bias=False),
                nn.PixelShuffle(2),
            ))
            chan_d //= 2
            self.decoders.append(nn.Sequential(*[blk(chan_d, nt) for _ in range(n)]))

        # Global context propagated once into the coarsest decoder stage --
        # haze is a whole-image phenomenon, so the estimate should reach the
        # decoder, not only live at the bottleneck.
        self.dec_strip = (StripPoolingGate(decoder_channels[0])
                          if use_strip_pool else None)

        # Oriented filters at the two HIGHEST-resolution decoder stages
        # (indices -2, -1). Rain streaks are 3-10px structures; at coarser
        # stages they are no longer resolvable, so placing the operator
        # there would cost parameters for nothing.
        self.streak_stages = (len(dec_blk_nums) - 2, len(dec_blk_nums) - 1)
        if use_oriented_streak:
            self.streak_gates = nn.ModuleDict({
                str(i): OrientedStreakGate(decoder_channels[i], reduction=4)
                for i in self.streak_stages
            })
        else:
            self.streak_gates = None

        # S3.1 block. Placed by default at the same two highest-resolution
        # decoder stages as the streak gates: S0.1 measured orientation as
        # worth +0.385 dB on derain and ~0 on denoise/dehaze, and rain streaks
        # are 3-10 px structures that are not resolvable at coarser stages.
        if use_reparam_oriented:
            stages = (tuple(reparam_stages) if reparam_stages is not None
                      else (len(dec_blk_nums) - 2, len(dec_blk_nums) - 1))
            bad = [i for i in stages if not 0 <= i < len(dec_blk_nums)]
            if bad:
                raise ValueError(
                    f"reparam_stages {bad} out of range for "
                    f"{len(dec_blk_nums)} decoder stages")
            if reparam_variant not in ("oriented", "plain"):
                raise ValueError(
                    f"reparam_variant must be 'oriented' or 'plain', "
                    f"got {reparam_variant!r}")
            self.reparam_variant = reparam_variant
            mk = (ReparamOrientedBlock if reparam_variant == "oriented"
                  else PlainLargeKernelBlock)
            self.reparam_stages = tuple(stages)
            self.reparam_blocks = nn.ModuleDict({
                str(i): mk(decoder_channels[i], k=reparam_k)
                for i in self.reparam_stages})
            self.mid_reparam = mk(chan, k=reparam_k) if reparam_middle else None
        else:
            self.reparam_variant = None
            self.reparam_stages = ()
            self.reparam_blocks = None
            self.mid_reparam = None

        self.padder_size = 2 ** len(enc_blk_nums)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        _, _, h, w = inp.shape
        inp = self._pad(inp)

        if self.use_dcp_prior:
            x = self.intro(torch.cat([inp, dark_channel_prior(inp)], dim=1))
        else:
            x = self.intro(inp)

        skips = []
        for enc, down in zip(self.encoders, self.downs):
            x = enc(x)
            skips.append(x)
            x = down(x)

        x = self.middle_blks(x)
        if self.mid_strip is not None:
            x = self.mid_strip(x)
        if self.mid_reparam is not None:
            x = self.mid_reparam(x)

        for i, (dec, up, skip) in enumerate(zip(self.decoders, self.ups, reversed(skips))):
            x = up(x)
            x = x + skip
            x = dec(x)
            if i == 0 and self.dec_strip is not None:
                x = self.dec_strip(x)
            if self.streak_gates is not None and str(i) in self.streak_gates:
                x = self.streak_gates[str(i)](x)
            if self.reparam_blocks is not None and str(i) in self.reparam_blocks:
                x = self.reparam_blocks[str(i)](x)

        x = self.ending(x)
        x = x + inp
        return x[:, :, :h, :w]

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        ph = (self.padder_size - h % self.padder_size) % self.padder_size
        pw = (self.padder_size - w % self.padder_size) % self.padder_size
        return F.pad(x, (0, pw, 0, ph))


def build_student_v3(cfg: dict, **overrides) -> StudentV3:
    """Construct from a model-config dict (see configs/model/), with the
    same key names NAFNet uses so a v3 config is a drop-in edit of a
    NAFNet one."""
    kwargs = dict(
        img_channels=cfg.get("img_channels", 3),
        width=cfg.get("width", 16),
        enc_blk_nums=cfg.get("enc_blk_nums"),
        middle_blk_num=cfg.get("middle_blk_num", 12),
        dec_blk_nums=cfg.get("dec_blk_nums"),
        norm_type=cfg.get("norm_type", "layernorm2d"),
        full_res_norm_type=cfg.get("full_res_norm_type", "affine_clamp"),
        clamp_bound=cfg.get("clamp_bound", 8.0),
        enc_clamp_stages=cfg.get("enc_clamp_stages"),
        deep_clamp_bound=cfg.get("deep_clamp_bound", 32.0),
        attn_type=cfg.get("attn_type", "sca"),
        use_dcp_prior=cfg.get("use_dcp_prior", True),
        use_strip_pool=cfg.get("use_strip_pool", True),
        use_oriented_streak=cfg.get("use_oriented_streak", True),
        mid_strip_every=cfg.get("mid_strip_every"),
    )
    kwargs.update(overrides)
    return StudentV3(**kwargs)
