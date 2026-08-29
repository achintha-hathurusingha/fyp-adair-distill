"""One-off smoke test for DecoderDegradationHead, in isolation, before wiring
it into NAFNet/trainer.py. CPU-only, dummy tensors. See
reports/kd_feature_multitask/plan_v2_decoder_film.md.
"""
import sys
sys.path.insert(0, ".")

import torch
import torch.nn.functional as F

from src.models.decoder_degradation_head import DecoderDegradationHead, N_TASKS

DECODER_CH = [128, 64, 32, 16]  # W16 SIDD, dec_blk_nums=[2,2,2,2]
MIDDLE_CH = 256

head = DecoderDegradationHead(middle_channels=MIDDLE_CH, decoder_channels=DECODER_CH)

middle = torch.randn(4, MIDDLE_CH, 8, 8, requires_grad=True)
labels = torch.tensor([0, 1, 2, 0])

logits, probs = head.classify(middle)
assert logits.shape == (4, N_TASKS), f"logits shape wrong: {logits.shape}"
assert probs.shape == (4, N_TASKS)
print(f"PASS  classify() shapes ok: logits={tuple(logits.shape)} probs={tuple(probs.shape)}")

# The critical property this whole redesign exists for: classify() must be
# READ-ONLY. Modulating downstream decoder stages must not change `middle`
# itself -- that's the exact bug being fixed.
middle_before = middle.detach().clone()

# Simulate 4 decoder stages at decreasing spatial resolution (upsampling),
# each modulated in turn.
xs = [torch.randn(4, c, 8 * (2 ** i), 8 * (2 ** i), requires_grad=True)
      for i, c in enumerate(DECODER_CH)]
outs = [head.modulate(x, probs, i) for i, x in enumerate(xs)]

assert torch.equal(middle, middle_before), \
    "classify() mutated its input -- this is exactly the v1 bug, must never happen"
print("PASS  classify() is read-only: middle_blks tensor unchanged after classify()+modulate()")

for i, (x, out) in enumerate(zip(xs, outs)):
    assert out.shape == x.shape, f"stage {i}: shape mismatch {out.shape} vs {x.shape}"
print(f"PASS  all {len(DECODER_CH)} decoder stages: shape preserved")

# Gradient flow check -- every FiLM head AND the classifier must receive
# gradient, and it must flow back to `middle` (via the classifier) and to
# each `x` (via that stage's FiLM), matching v1's own discipline.
ce = F.cross_entropy(logits, labels)
recon = sum((out - x.detach()).pow(2).mean() for out, x in zip(outs, xs))
loss = ce + recon
loss.backward()

assert head.classifier.weight.grad is not None and head.classifier.weight.grad.abs().sum() > 0, \
    "classifier got no gradient"
assert middle.grad is not None and middle.grad.abs().sum() > 0, \
    "middle_blks input got no gradient via the classifier path"
for i, film in enumerate(head.films):
    assert film.weight.grad is not None and film.weight.grad.abs().sum() > 0, \
        f"FiLM head {i} got no gradient -- that stage's conditioning would silently do nothing"
    assert xs[i].grad is not None and xs[i].grad.abs().sum() > 0, \
        f"decoder stage {i} input got no gradient"
print("PASS  gradients flow: classifier, middle_blks input, and all 4 FiLM heads")

# Wrong channel counts must raise, not silently misbehave.
try:
    head.classify(torch.randn(1, 128, 4, 4))
    print("FAIL: wrong middle channel count should have raised")
except ValueError as e:
    print(f"PASS  wrong middle channels raises: {e}")

try:
    head.modulate(torch.randn(1, 999, 4, 4), probs[:1], 0)
    print("FAIL: wrong decoder channel count should have raised")
except ValueError as e:
    print(f"PASS  wrong decoder channels raises: {e}")

n_params = sum(p.numel() for p in head.parameters())
print(f"\nDecoderDegradationHead params: {n_params:,} "
      f"(middle={MIDDLE_CH}, decoder={DECODER_CH})")
print("ALL SMOKE CHECKS PASSED")
