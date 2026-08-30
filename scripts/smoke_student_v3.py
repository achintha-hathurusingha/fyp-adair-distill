"""Smoke test for StudentV3 -- forward/backward, the no-confound identity
check against plain NAFNet, param accounting, and a real ONNX export +
op-coverage gate on the WHOLE model.

The identity check is the load-bearing one: with all three operators
disabled, v3 must be architecturally identical to NAFNet (same modules,
same param count, same output on the same weights). If that holds, any
later PSNR delta is attributable to the added operators alone and not to
an accidentally different backbone.
"""
import sys
sys.path.insert(0, ".")

import torch
from src.models.nafnet import build_nafnet
from src.models.student_v3 import build_student_v3
from src.export.op_coverage import op_histogram, render_markdown

CFG = dict(width=16, enc_blk_nums=[2, 2, 4, 8], middle_blk_num=12, dec_blk_nums=[2, 2, 2, 2],
           norm_type="layernorm2d", full_res_norm_type="affine_clamp",
           clamp_bound=8.0, enc_clamp_stages=[3], deep_clamp_bound=32.0)

torch.manual_seed(0)

# --- 1. no-confound identity: all operators OFF == plain NAFNet ----------
naf = build_nafnet(CFG).eval()
v3_off = build_student_v3(CFG, use_dcp_prior=False, use_strip_pool=False,
                          use_oriented_streak=False).eval()

n_naf = sum(p.numel() for p in naf.parameters())
n_off = sum(p.numel() for p in v3_off.parameters())
assert n_naf == n_off, f"param mismatch with operators off: NAFNet {n_naf:,} vs v3 {n_off:,}"

missing, unexpected = v3_off.load_state_dict(naf.state_dict(), strict=False)
assert not missing and not unexpected, f"state_dict mismatch: missing={missing[:3]} unexpected={unexpected[:3]}"

x_eval = torch.rand(1, 3, 128, 128)
with torch.no_grad():
    d = (naf(x_eval) - v3_off(x_eval)).abs().max().item()
assert d == 0.0, f"v3 with operators off is NOT identical to NAFNet: max diff {d}"
print(f"no-confound check PASSED: operators-off v3 == NAFNet exactly "
      f"(max diff {d}, {n_naf:,} params both)")

# --- 2. full model: forward + backward, every parameter gets gradient ----
v3 = build_student_v3(CFG)
n_v3 = sum(p.numel() for p in v3.parameters())
print(f"\nStudentV3 params: {n_v3:,}  (+{n_v3 - n_naf:,} over NAFNet, "
      f"{(n_v3 / n_naf - 1) * 100:.2f}%)")

x = torch.rand(2, 3, 128, 128, requires_grad=True)
v3.train()
y = v3(x)
assert y.shape == x.shape, f"shape mismatch {y.shape} vs {x.shape}"
y.sum().backward()
assert x.grad is not None and torch.isfinite(x.grad).all()
no_grad = [n for n, p in v3.named_parameters() if p.grad is None]
assert not no_grad, f"parameters with no gradient: {no_grad[:5]}"
n_tensors = sum(1 for _ in v3.parameters())
print(f"forward+backward OK, all {n_tensors} parameter tensors received finite gradient")

# --- 3. operator placement is where the design says it is ---------------
assert v3.mid_strip is not None, "bottleneck strip pooling missing"
assert v3.dec_strip is not None, "decoder strip pooling missing"
assert set(v3.streak_gates.keys()) == {"2", "3"}, \
    f"oriented gates should sit at the 2 highest-res decoder stages, got {set(v3.streak_gates.keys())}"
assert v3.intro.in_channels == 4, f"DCP prior not concatenated (intro in_channels={v3.intro.in_channels})"
print("operator placement verified: strip@bottleneck+dec0, oriented@dec2,3, DCP->4ch stem")

# --- 4. non-square / odd input (padding path) ---------------------------
with torch.no_grad():
    odd = torch.rand(1, 3, 130, 71)
    out_odd = v3.eval()(odd)
assert out_odd.shape == odd.shape, f"padding path broken: {out_odd.shape} vs {odd.shape}"
print(f"odd-size input OK: {tuple(odd.shape)} -> {tuple(out_odd.shape)}")

# --- 5. real ONNX export + op coverage on the WHOLE model ---------------
v3.eval()
onnx_path = "runs/student_v3_probe.onnx"
torch.onnx.export(v3, torch.rand(1, 3, 128, 128), onnx_path, opset_version=17,
                  input_names=["x"], output_names=["y"])
hist = op_histogram(onnx_path)
banned = {"DFT", "MatMul", "Softmax", "ReduceL2", "ReduceMin", "Neg", "Expand", "Where", "Equal"}
found_banned = banned & set(hist)
assert not found_banned, f"BANNED ops present in exported graph: {found_banned}"
print(f"\nno banned ops in exported graph (checked for {sorted(banned)})")
print("\n" + render_markdown(onnx_path))

print("ALL STUDENT-V3 SMOKE CHECKS PASSED")
