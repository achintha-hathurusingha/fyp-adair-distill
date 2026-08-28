"""One-off smoke test for DegradationHead, in isolation, before wiring it
into NAFNet/trainer.py. CPU-only, dummy tensors, no GPU needed.
"""
import sys
sys.path.insert(0, ".")

import torch
import torch.nn.functional as F

from src.models.degradation_head import DegradationHead, N_TASKS

x = torch.randn(4, 256, 8, 8, requires_grad=True)
labels = torch.tensor([0, 1, 2, 0])  # denoise, derain, dehaze, denoise

head = DegradationHead(channels=256)
out, logits = head(x)

assert out.shape == x.shape, f"shape mismatch: {out.shape} vs {x.shape}"
assert logits.shape == (4, N_TASKS), f"logits shape wrong: {logits.shape}"
print(f"PASS  shapes ok: out={tuple(out.shape)}  logits={tuple(logits.shape)}")

# Gradient flow check -- the real risk this smoke test exists for. An
# auxiliary head with no gradient reaching the FiLM scale/shift would
# silently do nothing at train time.
ce = F.cross_entropy(logits, labels)
recon = (out - x.detach()).pow(2).mean()  # dummy "downstream" loss using out
loss = ce + recon
loss.backward()

assert head.classifier.weight.grad is not None and head.classifier.weight.grad.abs().sum() > 0, \
    "classifier got no gradient"
assert head.film.weight.grad is not None and head.film.weight.grad.abs().sum() > 0, \
    "FiLM layer got no gradient -- conditioning would silently do nothing"
assert x.grad is not None and x.grad.abs().sum() > 0, "input got no gradient"
print(f"PASS  gradients flow: classifier.grad.abs().sum()={head.classifier.weight.grad.abs().sum():.4f}  "
      f"film.grad.abs().sum()={head.film.weight.grad.abs().sum():.4f}")

# Wrong channel count must raise, not silently misbehave.
try:
    head(torch.randn(1, 128, 4, 4))
    print("FAIL: wrong channel count should have raised")
except ValueError as e:
    print(f"PASS  wrong channels raises as expected: {e}")

n_params = sum(p.numel() for p in head.parameters())
print(f"\nDegradationHead params: {n_params:,} (channels=256)")
print("ALL SMOKE CHECKS PASSED")
