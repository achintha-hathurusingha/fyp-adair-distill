"""End-to-end smoke test of the REAL production path: FrozenTeacher loading
the config-resolved freq-fix checkpoint with NO manual patching -- confirms
the auto-detect-and-apply integration in teacher_wrapper.py actually works,
and that the OLD (all_in_one) checkpoint is completely unaffected (backward
compatibility, the whole point of making this additive-only).
"""
import sys
sys.path.insert(0, ".")

import yaml
import torch
from src.models.teacher_wrapper import FrozenTeacher

paths = yaml.safe_load(open("configs/paths.local.yaml"))
weights_root = paths["adair_weights_root"]
teachers = paths["teachers"]

# --- 1. old checkpoint: must be completely unaffected ---------------------
old_ckpt = f"{weights_root}/{teachers['all_in_one']}"
old_teacher = FrozenTeacher(old_ckpt, device="cpu")
assert old_teacher.freq_fix_mode is None, \
    f"old checkpoint should have freq_fix_mode=None, got {old_teacher.freq_fix_mode!r}"
print(f"OLD checkpoint ({teachers['all_in_one']}): freq_fix_mode={old_teacher.freq_fix_mode!r} -- unaffected, as expected")

# --- 2. new checkpoint: fix must auto-apply, mask must be live -------------
new_ckpt = f"{weights_root}/{teachers['all_in_one_freqfix']}"
new_teacher = FrozenTeacher(new_ckpt, device="cpu")
assert new_teacher.freq_fix_mode == "soft", \
    f"expected freq_fix_mode='soft', got {new_teacher.freq_fix_mode!r}"
print(f"NEW checkpoint ({teachers['all_in_one_freqfix']}): freq_fix_mode={new_teacher.freq_fix_mode!r} -- fix auto-applied")

x = torch.rand(1, 3, 128, 128)
out_old = old_teacher(x)
out_new = new_teacher(x)
assert out_old.shape == x.shape and out_new.shape == x.shape
assert torch.isfinite(out_old).all() and torch.isfinite(out_new).all()
print(f"both teachers produce finite output of the right shape")

# --- 3. confirm the mask is genuinely non-zero on the NEW teacher only ----
def _mods(net):
    import sys
    FreModule = sys.modules["net.model"].FreModule
    return [m for m in net.modules() if isinstance(m, FreModule)]

old_mods = _mods(old_teacher.net)
new_mods = _mods(new_teacher.net)
old_teacher(x)  # populate _freqfix_last if patched (it isn't, for old)
new_teacher(x)
old_coverage = [getattr(m, "_freqfix_last", None) for m in old_mods]
new_coverage = [m._freqfix_last["coverage"] for m in new_mods]
print(f"old teacher's FreModule._freqfix_last (should be all None, unpatched): {old_coverage}")
print(f"new teacher's mask coverage (should be non-zero): {new_coverage}")
assert all(c is None for c in old_coverage)
assert all(c > 0.0 for c in new_coverage)

print("\nALL CHECKS PASSED -- old checkpoint unaffected, new checkpoint's fix auto-applies")
