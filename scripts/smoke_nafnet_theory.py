"""End-to-end smoke test: NAFNet + all four theory additions together
(use_freq_gate, use_dcp_prior, use_strip_pool, use_oriented_streak), on the
real W16 SIDD (locked) geometry -- forward pass, param delta vs the plain
locked config, gradient flow through every new parameter, then a real ONNX
export + op-coverage check of the WHOLE model (not just the isolated
blocks) so the accounting in the lit review matches what actually gets
exported, not just the pieces in isolation.
"""
import sys
sys.path.insert(0, ".")

import torch
from src.models.nafnet import build_nafnet
from src.export.op_coverage import op_histogram, render_markdown

BASE_CFG = dict(width=16, enc_blk_nums=[2, 2, 4, 8], middle_blk_num=12, dec_blk_nums=[2, 2, 2, 2],
                norm_type="layernorm2d", full_res_norm_type="affine_clamp",
                clamp_bound=8.0, enc_clamp_stages=[3], deep_clamp_bound=32.0)

m_base = build_nafnet(BASE_CFG)
n_base = sum(p.numel() for p in m_base.parameters())

m_theory = build_nafnet(BASE_CFG, use_freq_gate=True, use_dcp_prior=True,
                        use_strip_pool=True, use_oriented_streak=True)
n_theory = sum(p.numel() for p in m_theory.parameters())
print(f"base params:   {n_base:,}")
print(f"+theory params: {n_theory:,}  (+{n_theory - n_base:,}, {(n_theory/n_base - 1) * 100:.2f}%)")

x = torch.rand(2, 3, 128, 128, requires_grad=True)
m_theory.train()
y = m_theory(x)
assert y.shape == x.shape, f"shape mismatch {y.shape} vs {x.shape}"
loss = y.sum()
loss.backward()
assert x.grad is not None and torch.isfinite(x.grad).all()

no_grad = [n for n, p in m_theory.named_parameters() if p.grad is None]
assert not no_grad, f"parameters with no gradient: {no_grad}"
print(f"forward+backward OK, all {sum(1 for _ in m_theory.parameters())} parameter tensors received gradient")

# gate identity-at-init sanity, in the full model context (not just isolated):
# freeze everything except confirm output changes negligibly from a freq_gate
# that starts at identity -- compare theory-model output at init against a
# hand-disabled version (base config) on the SAME input, expect them close
# but not necessarily identical since use_dcp_prior changes the input to
# `intro` (real difference, not a bug) -- so only check freq_gate alone.
m_freqonly = build_nafnet(BASE_CFG, use_freq_gate=True)
m_freqonly.eval(); m_base.eval()
m_freqonly.load_state_dict(m_base.state_dict(), strict=False)
missing = set(m_freqonly.state_dict()) - set(m_base.state_dict())
print(f"freq_gate-only adds {len(missing)} new state_dict tensors, e.g. {list(missing)[:2]}")
with torch.no_grad():
    x_eval = torch.rand(1, 3, 128, 128)
    y_base = m_base(x_eval)
    y_freq = m_freqonly(x_eval)
    max_diff = (y_base - y_freq).abs().max().item()
assert max_diff < 1e-5, f"freq_gate not identity at init when loaded onto base weights: max diff {max_diff}"
print(f"freq_gate confirmed identity-at-init in the FULL model (max diff {max_diff:.2e})")

# --- real ONNX export + op coverage of the WHOLE model -----------------
m_theory.eval()
dummy = torch.rand(1, 3, 128, 128)
onnx_path = "runs/nafnet_theory_probe.onnx"
torch.onnx.export(m_theory, dummy, onnx_path, opset_version=17,
                   input_names=["x"], output_names=["y"])
print("\n" + render_markdown(onnx_path))

print("ALL INTEGRATION SMOKE CHECKS PASSED")
