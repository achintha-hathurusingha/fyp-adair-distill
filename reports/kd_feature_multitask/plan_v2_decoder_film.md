# Plan — decoder-stage FiLM conditioning (kd_feature_multitask v2)

## Where this picks up

`B0V2-KD-FEAT-COND` (`reports/kd_feature_multitask/cond_regression.md`) made
every task worse, gap widening over training (dehaze −1.22dB by iteration
69,000). Diagnosis: FiLM modulated `middle_blks`'s output — the exact tensor
the feature-KD loss also reads (via the adapter, against the teacher's
`latent_pre`). Two objectives were fighting over one representation.

The literature review in that report converged on one fix, common to every
successful design surveyed: **never condition at the point another loss
already reads.** PromptIR's own ablation quantifies it directly — prompting
only the latent bottleneck scores 36.76dB on Rain100L; spreading the same
mechanism across decoder levels 4+3+2 reaches 37.04dB. This plan is that
fix, isolated as the only change from the failed design.

## Design — classify at the bottleneck, condition the decoder

Two changes from `DegradationHead`, everything else (the classifier itself,
`aux_weight=0.1`, the cross-entropy target `_provenance["task"]`) unchanged:

1. **Classification stays put, but becomes read-only.** The auxiliary head
   still pools and classifies `middle_blks`'s output — that tensor carries
   the clearest degradation signal (TEST19: 99.0% leave-scene-out accuracy on
   the teacher's own equivalent representation). The difference: it is only
   *read* now, never modulated. Feature-KD sees the pure, un-touched
   `middle_blks` tensor, exactly as it did in the control arm.
2. **FiLM moves to the decoder, one small head per stage.** Four small
   `Linear(3, 2*C_i)` projections (`C_i` = 128/64/32/16 for W16 SIDD's
   `dec_blk_nums=[2,2,2,2]`), one per decoder stage, all fed from the same
   softmax prediction. Applied right after each decoder stage's NAFBlocks —
   matching PromptIR's "between consecutive decoder levels" placement, and
   giving the conditioning signal four independent injection points instead
   of one, the other half of their ablation finding.

```python
class DecoderDegradationHead(nn.Module):
    """See reports/kd_feature_multitask/plan_v2_decoder_film.md. Classifies
    off middle_blks (read-only -- never modulates it, unlike the v1 design
    that regressed, see cond_regression.md), FiLM-conditions each decoder
    stage instead (PromptIR-style multi-level, decoder-only injection)."""

    def __init__(self, middle_channels: int, decoder_channels: list[int],
                 n_tasks: int = N_TASKS) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(middle_channels, n_tasks)
        self.films = nn.ModuleList(
            nn.Linear(n_tasks, 2 * c) for c in decoder_channels)
        self.decoder_channels = decoder_channels

    def classify(self, middle_feat: torch.Tensor):
        pooled = self.pool(middle_feat).flatten(1)
        logits = self.classifier(pooled)
        return logits, torch.softmax(logits, dim=-1)

    def modulate(self, x: torch.Tensor, probs: torch.Tensor, stage: int):
        c = self.decoder_channels[stage]
        scale, shift = self.films[stage](probs).chunk(2, dim=-1)
        scale = scale.view(-1, c, 1, 1)
        shift = shift.view(-1, c, 1, 1)
        return x * (1 + scale) + shift
```

`NAFNet.forward` change (the only wiring difference from v1 — `middle_blks`'s
output now flows to the decoder loop completely unmodified):

```python
x = self.middle_blks(x)
probs = None
if self.degradation_head is not None:
    self.last_degradation_logits, probs = self.degradation_head.classify(x)

for i, (dec, up, skip) in enumerate(zip(self.decoders, self.ups, reversed(skips))):
    x = up(x)
    x = x + skip
    x = dec(x)
    if self.degradation_head is not None:
        x = self.degradation_head.modulate(x, probs, i)
```

Parameter cost: classifier (771) + 4 FiLM heads (1024+512+256+128=1920) ≈
2,691 params for W16 SIDD — the same order of magnitude as v1's 2,819, not a
meaningfully bigger model.

## This is genuinely a controlled test of the diagnosis, not a new guess

Everything else stays byte-identical to `B0V2-KD-FEAT-COND`: same
`aux_weight=0.1` (already scale-checked, no reason to re-tune it — the
classifier and its loss are unchanged), same feature-KD term, same data mix,
same teacher, same schedule. The *only* variable is where the conditioning
signal gets written. If this arm still regresses, the tap-point diagnosis is
wrong and the next escalation (gated fusion, or R2R-style retrieval) is
warranted. If it recovers to at least match the control, the diagnosis is
confirmed and this becomes the arm to build on.

## Build order

1. `DecoderDegradationHead` module — smoke-test in isolation (shapes,
   gradient flow to every FiLM head, classifier gradient still flowing from
   `middle_blks`) before touching `NAFNet`, matching v1's own discipline.
2. Wire into `NAFNet` per the forward-pass change above, behind the same
   `use_degradation_head` flag (default off). Smoke-test end-to-end on CPU:
   verify `middle_blks`'s output tensor is bit-identical whether or not the
   head is active (the actual regression-causing behaviour from v1 must be
   gone), and that all 4 decoder stages show non-zero gradient on their FiLM
   heads.
3. Register a new arm, `B0V2-KD-FEAT-COND-DECFILM` (or similar — v1's name
   stays retired, its data is the control-comparison record in
   `cond_regression.md`), config identical to `B0V2-KD-FEAT-COND` in every
   field except the architecture change above.
4. `--smoke N` via the real training CLI, same checks as v1's own smoke
   pass: aux loss decreasing, FiLM heads receiving gradient, `middle_blks`
   output confirmed unaffected by the head (spot-check: with the head
   forcibly disabled after computing classification, decoder output should
   differ from the head-enabled run only downstream of the first FiLM
   application, never upstream of it).
5. Launch, staged against the current `B0V2-KD-FEAT` baseline the same way
   v1 was — watch the first couple of checkpoints before trusting it, given
   v1's regression was already visible by iteration 15,000.
