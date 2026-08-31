"""Pre-flight for the cached S3.3 pair, before 9 h of GPU.

Checks the things that would silently produce a wrong result rather than a
crash -- which is how both of the last two defects behaved.
"""
import sys

sys.path.insert(0, ".")

import json
from pathlib import Path

import torch
import yaml

FAIL = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not cond:
        FAIL.append(name)


CACHE = "cache/teacher_cache_v2"
ARMS = {"S33-CTRL-CACHED": "configs/train/s33_ctrl_cached.yaml",
        "S33-K11-CACHED": "configs/train/s33_k11_cached.yaml"}

print("=" * 74)
print("Cached S3.3 pair -- pre-flight")
print("=" * 74)

idx = json.loads(Path(CACHE, "index.json").read_text())
cache_data = {k: v for k, v in (idx.get("data_cfg") or {}).items()
              if k in ("sigmas", "sigma_range", "clean_prob", "patch_size")}

print("\n[1] does the cache match what each arm will train on?")
print(f"    cache built for: {idx.get('built_for_config')}")
print(f"    cache data_cfg : {cache_data}")
cfgs = {}
for arm, p in ARMS.items():
    c = yaml.safe_load(Path(p).read_text())
    cfgs[arm] = c
    d = c.get("data", {}) or {}
    arm_data = {"sigmas": d.get("sigmas"), "sigma_range": d.get("sigma_range"),
                "clean_prob": d.get("clean_prob"),
                "patch_size": d.get("patch_size")}
    check(f"{arm}: data regime matches the cache", arm_data == cache_data,
          str(arm_data))
    check(f"{arm}: use_cached_teacher is on",
          (c.get("distill") or {}).get("use_cached_teacher") is True)
    check(f"{arm}: points at {CACHE}",
          (c.get("distill") or {}).get("cache_dir") == CACHE,
          str((c.get("distill") or {}).get("cache_dir")))

print("\n[2] the pair differs in EXACTLY the block, nothing else")
a, b = cfgs["S33-CTRL-CACHED"], cfgs["S33-K11-CACHED"]
BLOCK_KEYS = {"use_reparam_oriented", "reparam_k", "reparam_variant", "reparam_middle"}
for sec in ("data", "optim", "schedule", "train", "loss", "distill", "eval"):
    check(f"{sec}: identical across the pair", a.get(sec) == b.get(sec))
ka, kb = a.get("arch", {}) or {}, b.get("arch", {}) or {}
diff = {k for k in set(ka) | set(kb) if ka.get(k) != kb.get(k)}
check("arch differs only in the block flags", diff <= BLOCK_KEYS, str(sorted(diff)))

print("\n[3] models build, and only the block differs in parameters")
from src.train.train import ARMS as REG  # noqa: E402
from src.models.student_v3 import StudentV3  # noqa: E402
ns = {}
for arm in ARMS:
    check(f"{arm}: registered", arm in REG)
    # build straight from the config's arch: section -- build_model() wants a
    # RESOLVED config (model: is a path in the raw yaml, not a dict)
    a_cfg = dict(yaml.safe_load(Path(REG[arm]["config"]).read_text())["arch"])
    a_cfg.pop("arch", None)
    torch.manual_seed(0)
    m = StudentV3(**a_cfg)
    ns[arm] = sum(p.numel() for p in m.parameters())
    print(f"      {arm:<18} {ns[arm]:>12,} params")
delta = ns["S33-K11-CACHED"] - ns["S33-CTRL-CACHED"]
check("K11 adds exactly the block's parameters", delta == 7088, f"+{delta:,}")

print("\n[4] the cached dataset actually serves sane data")
from src.data.cached_teacher_dataset import CachedTeacherDataset  # noqa: E402
ds = CachedTeacherDataset(CACHE)
check("dataset length matches index", len(ds) == idx["n"], f"{len(ds):,}")
for i in (0, idx["task_ranges"]["derain"][0], idx["task_ranges"]["dehaze"][0]):
    item = ds[i]
    t = item if isinstance(item, (tuple, list)) else tuple(item.values())
    shapes = [tuple(x.shape) for x in t if torch.is_tensor(x)]
    rng = [(float(x.min()), float(x.max())) for x in t
           if torch.is_tensor(x) and x.dtype.is_floating_point][:3]
    print(f"      row {i:>6}: shapes={shapes}")
    print(f"                 ranges={[(round(a,3), round(b,3)) for a, b in rng]}")
    finite = all(bool(torch.isfinite(x).all()) for x in t if torch.is_tensor(x))
    check(f"row {i}: all tensors finite", finite)

print("\n" + "=" * 74)
print("FAILED: " + str(FAIL) if FAIL else "ALL PRE-FLIGHT CHECKS PASSED")
sys.exit(1 if FAIL else 0)
