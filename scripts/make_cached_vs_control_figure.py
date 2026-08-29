"""Cached-teacher (B0V2-KD-FEAT-CACHED) vs control (B0V2-KD-FEAT), iteration-matched.

Step 5 validation figure: does training on the finite cached pool converge to
comparable PSNR at the same iteration count as the live-teacher control? Plus
the measured throughput (it/s) for each arm's own solo segment (both arms'
first checkpoint window ran with nothing else on the GPU, so first.iteration
/ first.elapsed_s is the fair, uncontaminated rate for both -- avoids diluting
control's rate with the period it later shared the GPU with COND).
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

OUT = "reports/kd_feature_multitask/figures"
os.makedirs(OUT, exist_ok=True)

NAVY = "#1f3a5f"
TEAL = "#2a9d8f"
AMBER = "#e9a723"
CORAL = "#e76f51"

control = json.load(open(
    "runs/b0v2_kd_feat/B0V2-KD-FEAT/B0V2-KD-FEAT_seed0_20260828_193951/history.json"))
cached = json.load(open(
    "runs/b0v2_kd_feat_cached/B0V2-KD-FEAT-CACHED/B0V2-KD-FEAT-CACHED_seed0_20260829_181115/history.json"))

c_map = {h["iteration"]: h for h in control}
k_map = {h["iteration"]: h for h in cached}
common = sorted(set(c_map) & set(k_map))

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# --- Panel 1: PSNR trajectory, control (live) vs cached, matched iterations ---
ax = axes[0]
metrics = [("psnr", "Combined", NAVY, "-o"),
           ("psnr_denoise", "Denoise", TEAL, "--s"),
           ("psnr_derain", "Derain", AMBER, "--^"),
           ("psnr_dehaze", "Dehaze", CORAL, "--d")]
for key, label, color, style in metrics:
    c_vals = [c_map[it][key] for it in common]
    k_vals = [k_map[it][key] for it in common]
    ax.plot(common, c_vals, style.replace("--", "-"), color=color,
            markersize=6, linewidth=2.0,
            label=f"{label} (live teacher)" if key == "psnr" else None)
    ax.plot(common, k_vals, style, color=color, markersize=6, linewidth=1.6,
             alpha=0.75, label=f"{label} (cached)" if key == "psnr" else None)

# Simplify legend: one entry per metric (solid=live, dashed=cached), plus a style key
handles = [plt.Line2D([0], [0], color=c, marker=m[-1], linestyle="-", label=l)
           for (_, l, c, m) in metrics]
handles += [plt.Line2D([0], [0], color="#444", linestyle="-", label="solid = live teacher"),
            plt.Line2D([0], [0], color="#444", linestyle="--", label="dashed = cached teacher")]
ax.legend(handles=handles, fontsize=8, loc="lower right", framealpha=0.9)
ax.set_xlabel("Iteration", fontsize=10.5)
ax.set_ylabel("PSNR (dB)", fontsize=10.5)
ax.set_title(f"Cached vs live teacher, matched iterations (n={len(common)} checkpoints)\n"
             "Nearly identical -- finite-pool cache is not costing quality so far",
             fontsize=10.5, weight="bold", color=NAVY, pad=10)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(alpha=0.25)

# --- Panel 2: measured throughput, each arm's own solo segment ---
ax2 = axes[1]
c_first = control[0]
k_first = cached[0]
c_rate = c_first["iteration"] / c_first["elapsed_s"]
k_rate = k_first["iteration"] / k_first["elapsed_s"]
speedup = k_rate / c_rate

bars = ax2.bar(["B0V2-KD-FEAT\n(live teacher)", "B0V2-KD-FEAT-CACHED\n(precomputed)"],
                [c_rate, k_rate], color=[NAVY, AMBER], width=0.55)
for b, v in zip(bars, [c_rate, k_rate]):
    ax2.text(b.get_x() + b.get_width() / 2, v + 0.08, f"{v:.2f} it/s",
              ha="center", fontsize=10.5, weight="bold")
ax2.set_ylabel("Iterations / second", fontsize=10.5)
ax2.set_ylim(0, max(c_rate, k_rate) * 1.35)
ax2.set_title(f"Measured speedup: {speedup:.2f}x\n"
              "(both arms' own first checkpoint window, solo on GPU)",
              fontsize=10.5, weight="bold", color=NAVY, pad=10)
ax2.spines[["top", "right"]].set_visible(False)
ax2.grid(axis="y", alpha=0.25)

fig.tight_layout()
fig.savefig(f"{OUT}/cached_vs_control.png", dpi=170, facecolor="white")
plt.close(fig)
print(f"wrote cached_vs_control.png  (speedup={speedup:.2f}x, "
      f"psnr@{common[-1]}: live={c_map[common[-1]]['psnr']:.3f} "
      f"cached={k_map[common[-1]]['psnr']:.3f})")
