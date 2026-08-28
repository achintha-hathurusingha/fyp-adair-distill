"""Diagrams for the student-approach report (reports/student_approach/).

Every number here is real, pulled from this project's own committed
configs/logs, not illustrative placeholders:
  - W16_SIDD geometry: src/train/train.py (width=16, enc_blk_nums=[2,2,4,8],
    middle_blk_num=12, dec_blk_nums=[2,2,2,2])
  - channel progression: NAFNet doubles channels once per encoder stage
    (16 -> 32 -> 64 -> 128 -> 256 middle), confirmed against the actual
    training log line "student middle_blks 256ch @ 1/16" (trainer.py).
  - teacher latent_pre 384ch @ 1/8: AdaIR dim=48, 3 downsamples, confirmed
    against the same log line ("teacher latent_pre 384ch @ 1/8").
  - PSNR figures: established, already-reported results from this project's
    own dehaze-only ablation ladder (GT-only/response-KD/kd_freq/kd_feat) and
    the B0V2 baseline's real denoise number.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = "reports/student_approach/figures"
import os
os.makedirs(OUT, exist_ok=True)

NAVY = "#1f3a5f"
TEAL = "#2a9d8f"
AMBER = "#e9a723"
CORAL = "#e76f51"
GREY = "#6b7280"
LIGHT = "#eef2f6"


def box(ax, x, y, w, h, text, fc=LIGHT, ec=NAVY, fontsize=8.5, weight="normal", tc="#1a1a1a"):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                        linewidth=1.3, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, weight=weight, color=tc, zorder=3)
    return b


def arrow(ax, xy1, xy2, color=GREY, style="-|>", lw=1.4, ls="solid", conn="arc3,rad=0.0"):
    a = FancyArrowPatch(xy1, xy2, arrowstyle=style, mutation_scale=12,
                        linewidth=lw, color=color, linestyle=ls,
                        connectionstyle=conn, zorder=1)
    ax.add_patch(a)


# ---------------------------------------------------------------------------
# Figure 1: student architecture -- encoder/middle/decoder with real channel
# counts, norm/clamp annotations, and the DegradationHead/FiLM insertion
# point.
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14.5, 5.2))
ax.set_xlim(0, 15.6)
ax.set_ylim(0, 5.2)
ax.axis("off")
ax.set_title("Student architecture — NAFNet (W16 SIDD, locked)",
             fontsize=13, weight="bold", loc="left", color=NAVY, pad=10)

enc_ch = [16, 32, 64, 128]
enc_blocks = [2, 2, 4, 8]
dec_ch = [128, 64, 32, 16]
dec_blocks = [2, 2, 2, 2]
stage_w = 1.15

# encoder (left to right, downsampling)
x = 0.3
enc_centers = []
for i, (ch, nb) in enumerate(zip(enc_ch, enc_blocks)):
    h = 1.0 + i * 0.35
    y = 2.6 - h / 2
    box(ax, x, y, stage_w, h, f"enc{i}\n{ch}ch\n×{nb} blk", fc="#dce8f5")
    enc_centers.append((x + stage_w, y + h / 2))
    x += stage_w + 0.35

# middle
mid_x = x
mid_h = 2.4
mid_y = 2.6 - mid_h / 2
box(ax, mid_x, mid_y, 1.3, mid_h, "middle_blks\n256ch @ 1/16\n×12 blk",
    fc="#fde9c8", ec=AMBER, fontsize=8.5, weight="bold")

# DegradationHead + FiLM, sitting on middle_blks output
dh_x, dh_y = mid_x + 0.15, mid_y + mid_h + 0.35
box(ax, dh_x, dh_y, 1.0, 0.85, "Degradation\nHead + FiLM\n(opt-in)",
    fc="#fbe4e1", ec=CORAL, fontsize=7.8)
arrow(ax, (mid_x + 0.65, mid_y + mid_h), (dh_x + 0.5, dh_y), color=CORAL,
      conn="arc3,rad=-0.15")
arrow(ax, (dh_x + 0.5, dh_y), (mid_x + 0.65, mid_y + mid_h), color=CORAL,
      ls=(0, (3, 2)), conn="arc3,rad=0.15")
ax.text(dh_x + 0.5, dh_y + 0.95, "scale·(1+x)+shift\n(FiLM, opt-in, aux_weight)",
        ha="center", va="bottom", fontsize=6.8, color=CORAL, style="italic")

x = mid_x + 1.3 + 0.35
dec_centers = []
for i, (ch, nb) in enumerate(zip(dec_ch, dec_blocks)):
    h = 1.0 + (3 - i) * 0.35
    y = 2.6 - h / 2
    box(ax, x, y, stage_w, h, f"dec{i}\n{ch}ch\n×{nb} blk", fc="#dce8f5")
    dec_centers.append((x, y + h / 2))
    x += stage_w + 0.35

# skip connections
for (ex, ey), (dx, dy) in zip(reversed(enc_centers), dec_centers):
    arrow(ax, (ex - 0.02, ey), (dx + 0.02, dy), color="#a9b7c6", lw=1.0,
          style="-", conn=f"arc3,rad=-0.35")

# main forward path
prev = (0.3, 2.6)
for c in enc_centers:
    arrow(ax, prev, (c[0] - stage_w, c[1]), color=NAVY, lw=1.6)
    prev = c
arrow(ax, prev, (mid_x, mid_y + mid_h / 2), color=NAVY, lw=1.6)
prev = (mid_x + 1.3, mid_y + mid_h / 2)
for c in dec_centers:
    arrow(ax, prev, (c[0], c[1]), color=NAVY, lw=1.6)
    prev = (c[0] + stage_w, c[1])
arrow(ax, prev, (x + 0.05, 2.6), color=NAVY, lw=1.6)
box(ax, x + 0.05, 2.6 - 0.35, 0.55, 0.7, "out\n3ch", fc="#dff5e5", ec=TEAL, fontsize=7.5)

# norm/clamp annotations
ax.annotate("LayerNorm2d (all blocks)\n+ affine_clamp(8.0) at full res (F9)\n"
            "+ enc3 deep clamp bound=32.0 (F10)",
            xy=(enc_centers[-1][0] - stage_w / 2, enc_centers[-1][1] - 1.1),
            xytext=(2.0, 0.35), fontsize=7.3, color=GREY,
            arrowprops=dict(arrowstyle="-", color=GREY, lw=0.8))
ax.text(0.3, 4.85,
        "7.37M params · 4.13 GMACs · 2.885 ms INT8 (S24) · SCA attention "
        "(ECA/GroupNorm variants tried, not adopted — see reports/student_arch/)",
        fontsize=7.6, color=GREY, style="italic")

fig.tight_layout()
fig.savefig(f"{OUT}/student_architecture.png", dpi=170, facecolor="white")
plt.close(fig)
print("wrote student_architecture.png")

# ---------------------------------------------------------------------------
# Figure 2: distillation pipeline -- teacher vs student, all loss terms.
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11.5, 6.4))
ax.set_xlim(0, 11.5)
ax.set_ylim(0, 6.4)
ax.axis("off")
ax.set_title("Distillation pipeline — current recipe",
             fontsize=13, weight="bold", loc="left", color=NAVY, pad=10)

# teacher (top row, frozen)
box(ax, 0.4, 5.0, 2.3, 0.9, "Teacher: AdaIR\n(adair3d.ckpt, frozen, eval)\n28.78M params",
    fc="#f0e6f6", ec="#7c5cbf", fontsize=8, weight="bold")
box(ax, 3.4, 5.0, 2.0, 0.9, "latent_pre\n384ch @ 1/8", fc="#f0e6f6", ec="#7c5cbf", fontsize=8)
box(ax, 6.1, 5.0, 2.0, 0.9, "response\n(teacher output)", fc="#f0e6f6", ec="#7c5cbf", fontsize=8)
arrow(ax, (2.7, 5.45), (3.4, 5.45), color="#7c5cbf")
arrow(ax, (2.7, 5.3), (6.1, 5.3), color="#7c5cbf", conn="arc3,rad=0.25")

# student (middle row)
box(ax, 0.4, 3.0, 2.3, 0.9, "Student: NAFNet\n(W16 SIDD, trainable)\n7.37M params",
    fc="#dce8f5", ec=NAVY, fontsize=8, weight="bold")
box(ax, 3.4, 3.0, 2.0, 0.9, "middle_blks capture\n256ch @ 1/16", fc="#dce8f5", ec=NAVY, fontsize=8)
box(ax, 6.1, 3.0, 2.0, 0.9, "pred\n(student output)", fc="#dce8f5", ec=NAVY, fontsize=8)
box(ax, 8.7, 3.0, 2.2, 0.9, "Degradation logits\n(opt-in, DegradationHead)",
    fc="#fbe4e1", ec=CORAL, fontsize=7.6)
arrow(ax, (2.7, 3.45), (3.4, 3.45), color=NAVY)
arrow(ax, (2.7, 3.3), (6.1, 3.3), color=NAVY, conn="arc3,rad=0.25")
arrow(ax, (5.4, 3.6), (8.7, 3.6), color=CORAL, ls=(0, (3, 2)))

# adapter bridging teacher/student feature scales
box(ax, 3.6, 4.05, 1.6, 0.6, "FeatureAdapter\n1x1 conv + upsample\n(train-time only)",
    fc="#fff2d6", ec=AMBER, fontsize=6.8)
arrow(ax, (4.4, 3.9), (4.4, 4.05), color=AMBER)
arrow(ax, (4.4, 5.0), (4.4, 4.65), color=AMBER)

# loss row (bottom)
box(ax, 0.4, 1.1, 2.3, 0.75, "pixel loss\nCharbonnier(pred, GT)", fc="#eafbea", ec=TEAL, fontsize=7.8)
box(ax, 3.0, 1.1, 2.6, 0.75, "response KD\nCharbonnier(pred, teacher)\nweight 1.0", fc="#eafbea", ec=TEAL, fontsize=7.5)
box(ax, 5.9, 1.1, 2.7, 0.75, "feature KD\nL1(adapter(mid), latent_pre)\nweight 0.01 (scale-checked)", fc="#eafbea", ec=TEAL, fontsize=7.3)
box(ax, 8.9, 1.1, 2.1, 0.75, "aux CE (opt-in)\nvs _provenance[task]\nweight 0.1 (scale-checked)", fc="#fbe4e1", ec=CORAL, fontsize=7.3)

arrow(ax, (1.55, 3.0), (1.55, 1.85), color=TEAL)
arrow(ax, (7.1, 3.0), (4.3, 1.85), color=TEAL, conn="arc3,rad=-0.15")
arrow(ax, (7.1, 5.0), (7.25, 1.85), color=TEAL, conn="arc3,rad=0.2")
arrow(ax, (4.4, 4.05), (7.25, 1.85), color=TEAL, conn="arc3,rad=-0.35")
arrow(ax, (9.8, 3.0), (9.95, 1.85), color=CORAL)

box(ax, 3.6, 0.0, 4.4, 0.6, "loss = pixel + 1.0·KD + 0.01·feat  ( + 0.1·aux, treatment arm only )",
    fc="white", ec="#1a1a1a", fontsize=8.4, weight="bold")

fig.tight_layout()
fig.savefig(f"{OUT}/distillation_pipeline.png", dpi=170, facecolor="white")
plt.close(fig)
print("wrote distillation_pipeline.png")

# ---------------------------------------------------------------------------
# Figure 3: DegradationHead + FiLM detail.
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9.5, 4.4))
ax.set_xlim(0, 9.5)
ax.set_ylim(0, 4.4)
ax.axis("off")
ax.set_title("DegradationHead + FiLM — conditioning mechanism (kd_feature_multitask)",
             fontsize=12.5, weight="bold", loc="left", color=NAVY, pad=10)

box(ax, 0.3, 1.7, 1.7, 0.9, "middle_blks\noutput\n(B,256,H,W)", fc="#dce8f5", ec=NAVY, fontsize=8)
box(ax, 2.4, 2.55, 1.5, 0.7, "GAP\n(B,256)", fc=LIGHT, fontsize=8)
box(ax, 4.2, 2.55, 1.7, 0.7, "Linear\n256→3\nclassifier", fc=LIGHT, fontsize=8)
box(ax, 6.2, 2.55, 1.6, 0.7, "softmax\nlogits", fc=LIGHT, fontsize=8)
box(ax, 4.2, 1.4, 1.7, 0.7, "Linear\n3→512\n(FiLM)", fc="#fbe4e1", ec=CORAL, fontsize=8)
box(ax, 6.2, 1.4, 1.6, 0.7, "scale, shift\n(B,256,1,1) each", fc="#fbe4e1", ec=CORAL, fontsize=7.8)
box(ax, 6.2, 0.15, 1.6, 0.7, "x·(1+scale)\n+ shift", fc="#fbe4e1", ec=CORAL, fontsize=8, weight="bold")
box(ax, 4.4, 3.6, 3.4, 0.6, "cross-entropy vs _provenance[\"task\"]\n(ground truth, already in the loader)",
    fc="#fff2d6", ec=AMBER, fontsize=7.8)

arrow(ax, (2.0, 2.15), (2.4, 2.9), color=NAVY)
arrow(ax, (3.9, 2.9), (4.2, 2.9), color=GREY)
arrow(ax, (5.9, 2.9), (6.2, 2.9), color=GREY)
arrow(ax, (7.0, 3.25), (6.0, 3.6), color=AMBER)
arrow(ax, (7.0, 2.55), (5.05, 2.1), color=CORAL, conn="arc3,rad=0.2")
arrow(ax, (5.05, 1.75), (7.0, 1.75), color=CORAL)
arrow(ax, (7.0, 1.4), (7.0, 0.85), color=CORAL)
arrow(ax, (2.15, 1.7), (6.2, 0.5), color=NAVY, ls=(0, (3, 2)), lw=1.1, conn="arc3,rad=0.3")
ax.text(4.2, 1.05, "same middle_blks tensor, elementwise-modulated\n"
                    "(deployment-safe: student conditions on its OWN\nprediction, no teacher at inference)",
        fontsize=6.8, color=GREY, ha="left", style="italic")

fig.tight_layout()
fig.savefig(f"{OUT}/degradation_head_detail.png", dpi=170, facecolor="white")
plt.close(fig)
print("wrote degradation_head_detail.png")

# ---------------------------------------------------------------------------
# Figure 4: results so far (dehaze-only ablation ladder, real numbers).
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.6, 5.0))
labels = ["GT-only", "+ response\nKD", "+ freq KD\n(stopped)", "+ feature KD\n(kd_feat)"]
vals = [32.8898, 33.0759, 33.190, 33.695]
colors = [GREY, TEAL, "#c9c9c9", AMBER]
bars = ax.bar(labels, vals, color=colors, edgecolor="#1a1a1a", linewidth=1.0, width=0.6)
ax.set_ylim(32.6, 33.9)
ax.set_ylabel("PSNR (dB), dehaze-only demo, best/mean per arm", fontsize=9.5)
ax.set_title("Single-task ablation ladder — established results\n"
              "(3-seed mean for GT-only/response-KD; single seed, full 60k iters, for freq/feature KD)",
              fontsize=10.5, weight="bold", color=NAVY, pad=12)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center",
            fontsize=9.5, weight="bold")
ax.annotate("+0.805 dB\nover GT-only", xy=(3, 33.70), xytext=(1.55, 33.62),
            fontsize=8, color=AMBER, weight="bold", ha="center",
            arrowprops=dict(arrowstyle="->", color=AMBER))
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(f"{OUT}/results_so_far.png", dpi=170, facecolor="white")
plt.close(fig)
print("wrote results_so_far.png")

print("\nALL FIGURES WRITTEN to", OUT)
