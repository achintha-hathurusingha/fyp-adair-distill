"""Compare sustained (post-warmup) batch throughput: kd_feat's single-task
dehaze loader (num_workers=8, 4000-image seeded subset) vs the new B0V2
mixed-task loader (num_workers=6, full denoise+derain+dehaze, balanced
sampler). Enough batches to exhaust any prefetch-queue head start and
actually measure sustained throughput, not just draining a warm buffer.
"""
import sys, time
sys.path.insert(0, ".")

import yaml
from pathlib import Path

from src.data.build import build_train_loader, build_multitask_loader, resolve_task_sources

paths = yaml.safe_load(open("configs/paths.local.yaml"))
data_root = Path(paths["data_root"])

N_WARMUP = 10
N_MEASURE = 80

def time_loader(name, loader):
    it = iter(loader)
    for _ in range(N_WARMUP):
        next(it)
    t0 = time.time()
    for _ in range(N_MEASURE):
        next(it)
    dt = time.time() - t0
    print(f"{name}: {N_MEASURE} batches in {dt:.2f}s = {dt/N_MEASURE*1000:.1f} ms/batch "
          f"({N_MEASURE/dt:.2f} batch/s)")
    return dt / N_MEASURE

# kd_feat's exact single-task loader config (m_dehaze_kd_feat.yaml)
single_sources = {
    "dehaze": {
        "input": data_root / "Train/Dehaze/synthetic",
        "target": data_root / "Train/Dehaze/clear",
        "list": Path("reports/dehaze_train_list.txt"),
    }
}
print("Building single-task (kd_feat-style) loader, num_workers=8...")
single_loader = build_multitask_loader(resolve_task_sources(single_sources, data_root),
                                       batch_size=16, patch_size=128, num_workers=8,
                                       seed=0, length=16 * (N_WARMUP + N_MEASURE + 5))
t_single = time_loader("single-task (dehaze only, 8 workers)", single_loader)

# B0V2-KD-FEAT's exact multi-task loader config (b0v2_kd_feat.yaml)
multi_sources = {
    "denoise": data_root / paths["datasets"]["denoise_train"],
    "derain": data_root / paths["datasets"]["derain_train"],
    "dehaze": {"input": data_root / "Train/Dehaze/synthetic",
              "target": data_root / "Train/Dehaze/clear"},
}
print("\nBuilding multi-task (B0V2) loader, num_workers=6...")
multi_loader = build_multitask_loader(multi_sources, batch_size=16, patch_size=128,
                                      sigma_range=(0.0, 55.0), clean_prob=0.05,
                                      num_workers=6, seed=0,
                                      length=16 * (N_WARMUP + N_MEASURE + 5))
t_multi = time_loader("multi-task (denoise+derain+dehaze, 6 workers)", multi_loader)

# Same multi-task loader but with num_workers=8, to isolate worker-count vs
# task-mix as the cause.
print("\nBuilding multi-task loader again, num_workers=8 (isolate worker count)...")
multi_loader8 = build_multitask_loader(multi_sources, batch_size=16, patch_size=128,
                                       sigma_range=(0.0, 55.0), clean_prob=0.05,
                                       num_workers=8, seed=0,
                                       length=16 * (N_WARMUP + N_MEASURE + 5))
t_multi8 = time_loader("multi-task (denoise+derain+dehaze, 8 workers)", multi_loader8)

print(f"\nmulti-task(6w) / single-task ratio: {t_multi/t_single:.2f}x")
print(f"multi-task(8w) / single-task ratio: {t_multi8/t_single:.2f}x")
print(f"multi-task(6w) / multi-task(8w) ratio: {t_multi/t_multi8:.2f}x")
