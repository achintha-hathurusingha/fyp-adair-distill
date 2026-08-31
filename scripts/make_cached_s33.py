"""Cached-teacher variants of the S3.3 pair.

WHY BOTH. The cache trades diversity for speed: measured against a live-teacher
control it is -0.119 dB at 12k and widening (a finite 180k pool, revisited).
Running only the block arm cached and comparing it to the LIVE B0V3-KD-FEAT
control would charge that -0.119 dB to the block. Both arms cached puts the
offset on both sides, where it cancels -- the same reasoning that makes a
baseline-vs-treatment pair valid even when both sit below some other reference.

Cost: live ~2.06 it/s (~12.1h for 90k); cached ~5.58 it/s (~4.5h). Two cached
runs are cheaper than one live run AND give a matched control.

Cached mode is mutually exclusive with teacher_task, and freq_weight is
unsupported (it needs a live fp32 forward), so teacher_task is dropped here.
"""
import re, sys
from pathlib import Path

PAIRS = {
    "S33-K11-CACHED":  ("configs/train/b0v3_kd_k11.yaml",
                        "configs/train/s33_k11_cached.yaml",
                        "S3.3 receptive-field control, 11x11 depthwise, CACHED teacher"),
    "S33-CTRL-CACHED": ("configs/train/b0v3_kd_feat.yaml",
                        "configs/train/s33_ctrl_cached.yaml",
                        "S3.3 no-block control, CACHED teacher (pairs with S33-K11-CACHED)"),
}

for arm, (src, dst, desc) in PAIRS.items():
    t = Path(src).read_text(encoding="utf-8")
    t = re.sub(r"(?m)^  teacher_task:.*\n", "", t)
    t = re.sub(r"(?m)^(distill:\n)",
               r"\1  use_cached_teacher: true\n  cache_dir: cache/teacher_cache_v2\n", t)
    t = (f"# {desc}\n"
         f"# Generated from {src}; identical except the cached-teacher switch.\n"
         f"# MUST be compared only against the other CACHED arm -- the cache costs\n"
         f"# ~0.12 dB vs a live teacher, which cancels only within a cached pair.\n" + t)
    Path(dst).write_text(t, encoding="utf-8")
    print(f"wrote {dst}")

tp = Path("src/train/train.py"); s = tp.read_text(encoding="utf-8")
if "S33-K11-CACHED" in s:
    print("arms already registered"); sys.exit(0)
anchor = '    "B0V3-KD-K11": {"norm": {"arch": "student_v3",'
assert s.count(anchor) == 1, "anchor not unique"

blocks = ""
for arm, (src, dst, desc) in PAIRS.items():
    extra = ('                     "use_reparam_oriented": True,\n'
             '                     "reparam_k": 11,\n'
             '                     "reparam_variant": "plain",\n'
             '                     "reparam_middle": False,\n') if "K11" in arm else ""
    blocks += (f'    "{arm}": {{"norm": {{"arch": "student_v3",\n'
               f'                     "use_dcp_prior": True,\n'
               f'                     "use_strip_pool": True,\n'
               f'                     "use_oriented_streak": True,\n'
               f'{extra}'
               f'                     "norm_type": "layernorm2d",\n'
               f'                     "full_res_norm_type": "affine_clamp",\n'
               f'                     "clamp_bound": 8.0,\n'
               f'                     "enc_clamp_stages": [3], "deep_clamp_bound": 32.0}},\n'
               f'              "config": "{dst}",\n'
               f'              "desc": "{desc}"}},\n')
s = s.replace(anchor, blocks + anchor)

geo = '                "B0V3-KD-K11": W16_SIDD,'
assert s.count(geo) == 1
s = s.replace(geo, "".join(f'                "{a}": W16_SIDD,\n' for a in PAIRS) + geo)
tp.write_text(s, encoding="utf-8")
print(f"registered {len(PAIRS)} cached arms")
