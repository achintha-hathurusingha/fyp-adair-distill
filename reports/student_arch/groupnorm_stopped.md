## M-DEHAZE-GROUPNORM — stopped at iteration 10,000/60,000

Stopped deliberately, not crashed. PSNR itself looked normal (25.7 -> 28.7 ->
29.8 -> 30.1 -> 30.1, matching a typical early-training curve), but the
deep-stage pre-clamp activation magnitude (`premax`, tracked via
`layernorm2d_clamp`'s clamp-engagement telemetry) was exploding:

| iteration | premax | clampeng |
|---|---:|---:|
| 2,000 | 64.87 | 1.60% |
| 4,000 | 42.61 | 2.19% |
| 6,000 | 545.2 | 3.44% |
| 8,000 | 1,807 | 3.76% |
| 10,000 | 5,688 | 4.29% |

The `deep_clamp_bound=32.0` clamp was still catching this, so PSNR hadn't
visibly cratered yet -- but a pre-clamp magnitude growing this fast, with
clamp engagement rising to compensate, is the same instability signature
this project's own F9/F10/F12 findings flag as a precursor to divergence,
not a stable-but-different training dynamic. GroupNorm's per-instance
statistics being computed over a different (smaller) set of channels than
LayerNorm2d's full-channel normalization is the likely mechanism -- worth
a real explanation if this direction is revisited, not just noted as "it
got worse."

**Verdict**: LayerNorm2d -> GroupNorm, at least in this exact locked
architecture with `layernorm2d_clamp`'s existing clamp bound tuned for
LayerNorm2d's own statistics, is not a safe swap. Stopped before the clamp
would have started dominating (or failing to contain) the signal, rather
than waiting for a PSNR collapse to confirm it.
