"""Normalization variants for the INT8 latency sweep (Task 1.5c).

On-device profiling of `w16_b8` (Snapdragon 8 Gen 3, Hexagon v75, INT8) showed
`LayerNorm2d` consuming ~62% of NPU cycles against ~3.4% for `Conv`, with no
CPU fallback at all. The cost is fixed-point `Div`/`Sqrt` on the integer
pipeline, and it is per-element — so normalisations at full resolution dominate.

All variants live behind a single ``norm_type`` key so the model file is never
forked. **Latency does not depend on weights**, so every variant here can be
exported, quantized and profiled untrained; only the survivors need training.

| id   | norm_type              | statistics | inference ops | retrain? |
|------|------------------------|------------|---------------|----------|
| N-A  | ``layernorm2d``        | mean+var   | many          | reference|
| N-A' | ``layernorm2d_rsqrt``  | mean+var   | many (rsqrt)  | **no**   |
| N-E  | ``affine``             | none       | fold-able     | yes      |
| N-C  | ``identity``           | none       | zero          | yes      |
| N-B  | ``batchnorm``          | running    | fold-able     | yes      |
| N-G  | ``groupnorm``          | mean+var   | many          | yes      |

N-G (kd_feature/student_arch experiment, not the INT8 latency sweep): tests
whether grouped-channel statistics change TRAINING QUALITY, not latency —
GroupNorm computes mean+var same as LayerNorm2d, so it is NOT expected to
beat LayerNorm2d on NPU cycles (same Div/Sqrt-on-integer-pipeline cost
class). Literature ablation (NAFNet block variant study) found it performs
comparably to LayerNorm2d, not dramatically better or worse — the point of
testing it here is quality, not the latency story N-A through N-B tell.
"""
from __future__ import annotations

import torch
from torch import nn

#: Every supported ``norm_type``. Unknown values raise (rule 9).
NORM_TYPES = ("layernorm2d", "layernorm2d_rsqrt", "affine", "affine_clamp",
              "layernorm2d_clamp", "identity", "batchnorm", "groupnorm")

#: Magnitude bound for :class:`LayerNorm2dClamp` on the deep encoder stages.
#: From measurement: during the F9 divergence — the single worst activation
#: event on record, where `dec3` reached max|a| 5.6e6 — `enc3` reached only 7.59.
#: A bound of 32 is >4x that and cannot engage in any state resembling one
#: observed so far, which is the point: it is insurance, not an operator.
DEEP_CLAMP_BOUND = 32.0

#: Default magnitude bound for :class:`AffineClampNorm2d` (variant N-F-clamp).
#: Chosen from measurement, not taste: at the B0 divergence the exploding stage
#: (`dec3`) reached max|a| 5.6e6 while every healthy sample stayed under ~30
#: through the same stage. A bound of 64 is >2x the healthy maximum observed and
#: ~5 orders below the pathological one, so it is inert in normal training and
#: only engages on the failure mode.
AFFINE_CLAMP_BOUND = 64.0

#: When True, :class:`AffineClampNorm2d` counts how often it actually engages.
#: Off by default — this costs a comparison and a sync-free reduction per
#: forward, which is fine for a diagnostic run and not for production. A clamp
#: that engages constantly is not insurance, it is an undocumented activation
#: function distorting normal training, so this is the number that decides
#: whether a chosen bound is defensible.
TRACK_CLAMP_ENGAGEMENT = False


class LayerNorm2d(nn.Module):
    """N-A (reference): channel-wise LayerNorm for NCHW tensors.

    Exports to ReduceMean/Sub/Pow/Sqrt/Div. The `Div` and `Sqrt` are the
    expensive fixed-point operations on Hexagon.
    """

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=1, keepdim=True)
        var = (x - mu).pow(2).mean(dim=1, keepdim=True)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class LayerNorm2dRsqrt(nn.Module):
    """N-A': mathematically identical to :class:`LayerNorm2d`, with no `Div`.

    **REJECTED — retained as a documented negative result. Do not adopt.**

    Measured on Snapdragon 8 Gen 3 (Hexagon v75, INT8): **2.87 ms vs 2.51 ms
    for N-A — 14% slower**, despite removing every elementwise `Div`.

    Why: QNN *fuses* the whole LayerNorm subgraph and bills it to the terminal
    `Div` node (its neighbours profile at exactly zero cycles). Rewriting the
    division breaks that fusion match, so `Sub`/`Pow`/`ReduceMean`/`Mul` — free
    when fused — start executing separately and cost more than the division
    saved. The canonical form is the fast path *because* it is the
    fusion-matched form. See ``reports/findings.md`` F3 and F4.

    Because the computed function is unchanged (to floating-point tolerance),
    swapping N-A for N-A' is a **pure graph rewrite**: no retraining, no quality
    ablation, and an N-A checkpoint loads directly.

    Spelling matters, and the obvious spelling is a trap:

    * ``torch.rsqrt(v)``  ->  ONNX ``Sqrt`` then ``Div(1, .)``. The `Div`
      survives and an extra `Mul` is added — strictly **worse** than N-A.
      Verified at opset 17 and 20.
    * ``torch.reciprocal(torch.sqrt(v))``  ->  ONNX ``Sqrt`` then
      ``Reciprocal``. `Div` count drops to zero.

    The second form is used here. The expected win is not merely
    "Reciprocal beats Div" but a change in *how many elements* the expensive
    op touches: in N-A the division runs over the full ``(N, C, H, W)`` tensor
    (broadcasting a ``(N, 1, H, W)`` denominator), whereas here `Reciprocal`
    runs over just ``(N, 1, H, W)`` — a factor of ``C`` fewer elements — and
    the full-size work becomes a cheap `Mul`.
    """

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=1, keepdim=True)
        var = (x - mu).pow(2).mean(dim=1, keepdim=True)
        # NOTE: torch.reciprocal(torch.sqrt(.)), NOT torch.rsqrt(.) -- see docstring.
        x = (x - mu) * torch.reciprocal(torch.sqrt(var + self.eps))
        return x * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class AffineNorm2d(nn.Module):
    """N-E: per-channel learnable scale and bias. No statistics computed.

    Zero reductions, zero division. Folds into an adjacent 1x1 convolution, so
    at inference it can cost nothing at all. Needs retraining (it does not
    normalise, so activation scales must be learned).
    """

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        del eps  # unused; kept for a uniform constructor signature
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class GroupNorm2d(nn.Module):
    """N-G: standard GroupNorm, num_groups fixed at 8.

    8 divides every channel count this project's student architectures
    actually use (width doubles each stage: 16/32/64/128/256 for the
    width=16 family) — chosen for that, not tuned. Raises rather than
    silently falling back if a future config's channel count isn't
    divisible, since a wrong group count changes what's being normalised
    without any other symptom.
    """

    def __init__(self, channels: int, eps: float = 1e-6, num_groups: int = 8) -> None:
        super().__init__()
        if channels % num_groups != 0:
            raise ValueError(
                f"GroupNorm2d: channels={channels} not divisible by "
                f"num_groups={num_groups}")
        self.gn = nn.GroupNorm(num_groups, channels, eps=eps, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gn(x)


def record_clamp_engagement(module: nn.Module, x: torch.Tensor) -> None:
    """Accumulate this interval's clamp diagnostics onto ``module``.

    Shared by every clamped norm so a new clamp site cannot be added without
    telemetry — which is exactly what happened to ``LayerNorm2dClamp``: it
    shipped as F10 insurance with no counters at all, so the finding it was
    meant to let us watch (F12) would have been unobservable at that site.

    Records the **pre-clamp** magnitude, not just fire/no-fire. Engagement count
    alone cannot distinguish "catches a stable large event occasionally" from
    "catches an increasingly large event occasionally" — both give the same
    rate. The magnitude is what says whether the bound still has headroom.

    Plain attributes, deliberately NOT buffers: a buffer would enter
    ``state_dict`` and break weight compatibility between the affine, clamped
    and layernorm variants, which is what makes the F9 fix comparison possible.
    """
    if not TRACK_CLAMP_ENGAGEMENT:
        return
    module.forwards = getattr(module, "forwards", 0) + 1
    mag = float(x.abs().max())
    if mag > getattr(module, "max_preclamp", 0.0):
        module.max_preclamp = mag
    over = int((x.abs() > module.bound).sum())
    if over:
        module.engaged = getattr(module, "engaged", 0) + 1
        module.elements_clamped = getattr(module, "elements_clamped", 0) + over


def reset_clamp_engagement(module: nn.Module) -> None:
    """Zero one module's counters at an interval boundary."""
    module.forwards = module.engaged = module.elements_clamped = 0
    module.max_preclamp = 0.0


class AffineClampNorm2d(nn.Module):
    """N-F-clamp: affine, then a hard magnitude bound. No statistics.

    Motivation (findings F9). Under N-F the full-resolution decoder stage
    `dec3` has no normalisation to bound its input. A rare low-variance sample
    drove it to max|a| 5.6e6 and produced a gradient norm of 6.5e7, killing B0.
    LayerNorm survives that because it renormalises to unit variance — but it
    costs reductions, a `Sqrt` and a `Div`, which is exactly what N-F removed to
    win 1.59x on the NPU.

    A clamp bounds worst-case magnitude directly with one elementwise op, no
    reduction. It is also *stronger* than normalisation against this failure:
    LayerNorm rescales by measured statistics, which a sufficiently pathological
    input can still evade, whereas a clamp bounds unconditionally.

    ``Clip`` is a first-class quantized op on Hexagon and is commonly fused into
    the preceding operation, so the expected latency cost is near zero — but
    that is a claim to MEASURE on AI Hub, not to assume.
    """

    def __init__(self, channels: int, eps: float = 1e-6,
                 bound: float | None = None) -> None:
        super().__init__()
        del eps  # unused; uniform constructor signature
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        # Read the module constant at CONSTRUCTION time, not as a default
        # argument: Python binds defaults once at def time, so `bound=
        # AFFINE_CLAMP_BOUND` would freeze the value at import and silently
        # ignore any later override. That produced a bound sweep in which every
        # setting returned an identical result.
        self.bound = AFFINE_CLAMP_BOUND if bound is None else bound

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.weight[None, :, None, None] + self.bias[None, :, None, None]
        record_clamp_engagement(self, x)
        return torch.clamp(x, -self.bound, self.bound)


class LayerNorm2dClamp(LayerNorm2d):
    """LayerNorm2d followed by a hard magnitude bound. Deep-stage insurance.

    **Read the caveat before adopting this.** LayerNorm already bounds its own
    output structurally — it renormalises to unit variance per pixel, so the
    output magnitude is governed by the learned ``weight``/``bias`` rather than
    by the input. That is precisely why `dec3`, which under N-F has *no*
    normalisation, reached max|a| 5.6e6 during the F9 divergence while `enc3`,
    which has LayerNorm, reached 7.59 on the very same sample. A clamp here
    therefore guards a failure mode that has never been observed at this depth.

    It ships anyway, for the F9 reason: a clamp bounds unconditionally, and
    distribution coverage does not. `enc3` is the stage where a low-noise input
    is amplified across 8 blocks at 128 channels (finding F10), and the cost of
    being wrong about "LayerNorm makes this impossible" is another multi-day run.

    **Latency consequence.** This changes the exported graph, so every INT8
    number measured on the locked M arm — 2.885 ms and everything derived from
    it — describes a model WITHOUT this clamp. `Clip` is a first-class quantized
    op on Hexagon and usually fuses into its predecessor, so the expected cost is
    near zero, but that is a claim to MEASURE on AI Hub before it is quoted.
    """

    def __init__(self, channels: int, eps: float = 1e-6,
                 bound: float | None = None) -> None:
        super().__init__(channels, eps)
        # Read at CONSTRUCTION time, not as a default argument -- see
        # AffineClampNorm2d for the bug that convention prevents.
        self.bound = DEEP_CLAMP_BOUND if bound is None else bound

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super().forward(x)
        # Same telemetry as the full-resolution clamp. F12's watch item is
        # "does enc3 engage at all"; without this it could not be answered.
        record_clamp_engagement(self, x)
        return torch.clamp(x, -self.bound, self.bound)


class IdentityNorm2d(nn.Module):
    """N-C: no normalisation at all. Zero ops.

    Training stability is unproven — expect to need a reduced learning rate.
    """

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        del channels, eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def build_norm(norm_type: str, channels: int, eps: float = 1e-6,
               clamp_bound: float | None = None,
               deep_clamp_bound: float | None = None) -> nn.Module:
    """Construct a normalisation module by name.

    Args:
        norm_type: one of :data:`NORM_TYPES`.
        channels: channel count of the tensor being normalised.
        eps: numerical epsilon (ignored by variants without statistics).
        clamp_bound: magnitude bound for ``affine_clamp``; ``None`` uses
            :data:`AFFINE_CLAMP_BOUND`. Ignored by other variants.
        deep_clamp_bound: magnitude bound for ``layernorm2d_clamp``; ``None``
            uses :data:`DEEP_CLAMP_BOUND`. Ignored by other variants.

    Raises:
        ValueError: on an unknown ``norm_type`` — never silently defaults.
    """
    if norm_type == "layernorm2d":
        return LayerNorm2d(channels, eps)
    if norm_type == "layernorm2d_rsqrt":
        return LayerNorm2dRsqrt(channels, eps)
    if norm_type == "affine":
        return AffineNorm2d(channels)
    if norm_type == "affine_clamp":
        return AffineClampNorm2d(channels, bound=clamp_bound)
    if norm_type == "layernorm2d_clamp":
        # NOTE: deliberately NOT sharing `clamp_bound` with affine_clamp. The two
        # bound different quantities -- an unnormalised affine output at full
        # resolution versus a renormalised deep-stage output -- and were chosen
        # from separate measurements. One key for both would silently retune one
        # of them whenever the other was changed.
        return LayerNorm2dClamp(channels, eps, bound=deep_clamp_bound)
    if norm_type == "identity":
        return IdentityNorm2d(channels)
    if norm_type == "groupnorm":
        return GroupNorm2d(channels, eps)
    if norm_type == "batchnorm":
        # N-B: latency datapoint only. BatchNorm folds to zero inference ops so
        # it will look excellent, but it is NOT expected to win on quality:
        #  1. EDSR and successors removed BN from restoration networks because
        #     it degraded output quality; NAFNet's authors also avoided it.
        #  2. BN statistics are computed over spatial dims as well as batch.
        #     Training on 256x256 patches and evaluating on full-resolution
        #     images gives running statistics that do not match test-time
        #     activation distributions — a concrete train/test mismatch.
        return nn.BatchNorm2d(channels, eps=eps)
    raise ValueError(
        f"Unknown norm_type {norm_type!r}. Supported: {list(NORM_TYPES)}")
