"""Comparison figures from this session's real, already-collected data:
  1. Teacher vs GT-only baseline vs KD-FEAT, per task (the student-vs-
     teacher gap table).
  2. Control (B0V2-KD-FEAT) vs Treatment-v1 (B0V2-KD-FEAT-COND), per-task
     PSNR delta over training iterations -- the cond_regression.md finding,
     as a chart instead of just a table.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "reports/kd_feature_multitask/figures"
import os
os.makedirs(OUT, exist_ok=True)

NAVY = "#1f3a5f"
TEAL = "#2a9d8f"
AMBER = "#e9a723"
CORAL = "#e76f51"
GREY = "#6b7280"

# ---------------------------------------------------------------------------
# Figure 1: three-way per-task comparison (real numbers, this session).
# ---------------------------------------------------------------------------
tasks = ["Denoise", "Derain", "Dehaze"]
teacher = [31.253, 39.725, 36.928]
baseline = [30.686, 36.828, 34.645]  # GT-only, 285k iters
kdfeat = [30.693, 36.071, 34.100]    # +response+feature KD, 90k iters
# Student v3 (degradation-matched operators), GT-only, seed1 @ 90k -- read from
# runs/b0v3/B0V3/B0V3_seed1_20260830_180851/history.json
v3 = [30.649, 36.129, 33.781]

x = np.arange(len(tasks))
w = 0.20
fig, ax = plt.subplots(figsize=(9, 5.5))
b1 = ax.bar(x - 1.5 * w, teacher, w, label="AdaIR Teacher (28.78M params)", color=NAVY)
b2 = ax.bar(x - 0.5 * w, baseline, w, label="B0V2 GT-only (285k iters)", color=GREY)
b3 = ax.bar(x + 0.5 * w, kdfeat, w, label="B0V2-KD-FEAT (90k iters)", color=AMBER)
b4 = ax.bar(x + 1.5 * w, v3, w, label="Student v3 GT-only (90k iters)", color=TEAL)

for bars in (b1, b2, b3, b4):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.15,
                f"{b.get_height():.2f}", ha="center", fontsize=7.4, weight="bold")

ax.set_xticks(x)
ax.set_xticklabels(tasks, fontsize=11)
ax.set_ylabel("PSNR (dB)", fontsize=10.5)
ax.set_ylim(28, 42)
ax.set_title("Student vs teacher, per task — real evaluation, same harness\n"
              "(285k vs 90k iters is NOT matched. v3 vs KD-FEAT both 90k, so that pair IS matched.)",
              fontsize=10.5, weight="bold", color=NAVY, pad=12)
ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(f"{OUT}/teacher_student_gap.png", dpi=170, facecolor="white")
plt.close(fig)
print("wrote teacher_student_gap.png")

# ---------------------------------------------------------------------------
# Figure 2: control vs treatment-v1, per-task delta over iterations.
# ---------------------------------------------------------------------------
control_hist = json.load(open(
    "runs/b0v2_kd_feat/B0V2-KD-FEAT/B0V2-KD-FEAT_seed0_20260828_193951/history.json"))
cond_hist = json.load(open(
    "runs/b0v2_kd_feat_cond/B0V2-KD-FEAT-COND/B0V2-KD-FEAT-COND_seed0_20260828_200459/history.json"))

def by_iter(hist):
    return {h["iteration"]: h for h in hist}

c_map, t_map = by_iter(control_hist), by_iter(cond_hist)
common_iters = sorted(set(c_map) & set(t_map))

fig, ax = plt.subplots(figsize=(9.5, 5.5))
metrics = [("psnr", "Combined", NAVY, "-o"),
          ("psnr_denoise", "Denoise", TEAL, "-s"),
          ("psnr_derain", "Derain", AMBER, "-^"),
          ("psnr_dehaze", "Dehaze", CORAL, "-d")]
for key, label, color, style in metrics:
    deltas = [t_map[it][key] - c_map[it][key] for it in common_iters]
    ax.plot(common_iters, deltas, style, color=color, label=label,
           markersize=5, linewidth=1.8)

ax.axhline(0, color="#1a1a1a", linewidth=1, linestyle="--", alpha=0.6)
ax.set_xlabel("Iteration", fontsize=10.5)
ax.set_ylabel("PSNR delta, treatment − control (dB)", fontsize=10.5)
ax.set_title("B0V2-KD-FEAT-COND (v1, FiLM on middle_blks) vs control\n"
              "Every task gets worse, gap widening — stopped at it 69,000",
              fontsize=11, weight="bold", color=NAVY, pad=12)
ax.legend(loc="lower left", fontsize=9.5, framealpha=0.9)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(f"{OUT}/cond_v1_regression.png", dpi=170, facecolor="white")
plt.close(fig)
print("wrote cond_v1_regression.png")

print(f"\nBoth figures written to {OUT}")
