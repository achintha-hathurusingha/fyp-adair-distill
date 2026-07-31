"""Build ``reports/norm_quality.md`` from the Task 1.5b arm histories.

Plots the validation curves (PSNR, loss, gradient norm, per-level activation
magnitude) to PNG and applies the decision rule. Curve *shape* is the point:
divergence, oscillation and plateaus are what separate a trainability failure
from a capacity failure, and a single endpoint hides all of them.

    python -m scripts.norm_quality_report
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.utils.config import REPO_ROOT

#: Measured INT8 latency on w16_b8 (Samsung Galaxy S24), from Task 1.5c/Task 0.
LATENCY_MS = {"Q-A": 2.513, "Q-F": 1.580, "Q-E": 1.072,
              "Q-E1": 1.072, "Q-E2": 1.072, "Q-E3": 1.072}
SPEEDUP = {k: LATENCY_MS["Q-A"] / v for k, v in LATENCY_MS.items()}

#: Decision thresholds, in dB of BSD68 PSNR against Q-A.
TOL_LOCK_E = 0.10
TOL_BAND_HI = 0.30
TOL_F = 0.15


def load_arms(root: Path) -> dict[str, dict]:
    """Load each arm's history and final metrics from its run directory."""
    arms: dict[str, dict] = {}
    for arm_dir in sorted(root.glob("*")):
        if not arm_dir.is_dir():
            continue
        runs = sorted(arm_dir.glob("*/history.json"))
        if not runs:
            continue
        history = json.loads(runs[-1].read_text(encoding="utf-8"))
        if not history:
            continue
        metrics_path = runs[-1].parent / "metrics.json"
        metrics = (json.loads(metrics_path.read_text(encoding="utf-8"))
                   if metrics_path.exists() else {})
        arms[arm_dir.name] = {"history": history, "metrics": metrics,
                              "run_dir": runs[-1].parent}
    return arms


def plot_curves(arms: dict[str, dict], out_png: Path) -> bool:
    """Plot validation curves for every arm. Returns False if unavailable."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    panels = [("psnr", "BSD68 PSNR (dB)"), ("loss", "training loss"),
              ("grad_norm", "gradient norm"), ("act_enc0", "mean |activation| enc0")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (key, title) in zip(axes.ravel(), panels):
        for arm, data in sorted(arms.items()):
            xs = [r["iteration"] for r in data["history"] if key in r]
            ys = [r[key] for r in data["history"] if key in r]
            if xs:
                ax.plot(xs, ys, marker="o", markersize=3, label=arm)
        ax.set_title(title)
        ax.set_xlabel("iteration")
        ax.grid(alpha=0.3)
        if key == "loss":
            ax.set_yscale("log")
        ax.legend(fontsize=8)
    fig.suptitle("Task 1.5b — normalization ablation (w16_b8, denoising)")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return True


def decide(arms: dict[str, dict]) -> tuple[str, str]:
    """Apply the decision rule; return (locked_norm, explanation)."""
    def best(arm: str) -> float | None:
        d = arms.get(arm)
        return d["metrics"].get("best_psnr") if d else None

    qa = best("Q-A")
    qf = best("Q-F")
    # Best-trained Q-E variant across the escalation ladder.
    e_arms = {a: best(a) for a in ("Q-E", "Q-E1", "Q-E2", "Q-E3") if best(a)}
    if qa is None:
        return "UNDECIDED", "Q-A reference did not complete — cannot decide."
    if not e_arms:
        return "UNDECIDED", "no Q-E variant completed."

    qe_arm = max(e_arms, key=lambda a: e_arms[a])
    qe = e_arms[qe_arm]
    d_e = qe - qa
    rung = "" if qe_arm == "Q-E" else f" (via escalation rung {qe_arm})"

    if d_e >= -TOL_LOCK_E:
        return "N-E", (
            f"Best Q-E variant is `{qe_arm}` at {qe:.3f} dB vs Q-A {qa:.3f} dB "
            f"(Δ{d_e:+.3f} dB), within the {TOL_LOCK_E} dB band{rung}. "
            f"**Lock N-E** and take the {SPEEDUP['Q-E']:.2f}x speedup.")

    if d_e >= -TOL_BAND_HI:
        if qf is None:
            return "UNDECIDED", "Q-E is in the 0.10-0.30 dB band and Q-F is missing."
        d_f = qf - qa
        if d_f >= -TOL_LOCK_E:
            return "N-F", (
                f"Q-E is {abs(d_e):.3f} dB below Q-A (in the 0.10-0.30 band), but "
                f"Q-F is {d_f:+.3f} dB — within {TOL_LOCK_E} dB. **Lock N-F** for "
                f"{SPEEDUP['Q-F']:.2f}x at lower risk.")
        return "STOP", (
            f"**STOP AND REPORT.** Q-E is {abs(d_e):.3f} dB below Q-A (0.10-0.30 "
            f"band) and Q-F is also {abs(d_f):.3f} dB below — beyond "
            f"{TOL_LOCK_E} dB. No clean winner; this is a judgement call.")

    d_f = (qf - qa) if qf is not None else None
    if d_f is not None and d_f < -TOL_F:
        return "N-A", (
            f"Q-E is {abs(d_e):.3f} dB below Q-A (>{TOL_BAND_HI} dB) and Q-F is "
            f"{abs(d_f):.3f} dB below (>{TOL_F} dB). **Lock N-A**: keep "
            "normalization and accept the latency.")
    if d_f is not None:
        return "N-F", (
            f"Q-E is {abs(d_e):.3f} dB below Q-A, but Q-F holds at {d_f:+.3f} dB. "
            f"**Lock N-F** for {SPEEDUP['Q-F']:.2f}x.")
    return "UNDECIDED", "Q-E degraded and Q-F is missing."


def build_report(arms: dict[str, dict], has_plot: bool) -> str:
    locked, why = decide(arms)
    L = [
        "# Normalization quality ablation — Task 1.5b", "",
        "Latency was measured in Task 1.5c on **untrained** weights; this task "
        "measures the quality axis on **trained** models. Config `w16_b8`, "
        "denoising only (σ ∈ {15, 25, 50}), 30k iterations at batch 32 and "
        "patch 128, single seed, identical data/augmentation/optimiser/schedule "
        "across every arm. Validation on BSD68 through the locked harness "
        "(`src/eval/evaluate.py`).", "",
        f"## Decision: **{locked}**", "", why, "",
        "## Final results", "",
        "| arm | normalization | BSD68 PSNR | ΔPSNR vs Q-A | INT8 ms | speedup | peak VRAM | iters |",
        "|---|---|---|---|---|---|---|---|",
    ]
    qa = arms.get("Q-A", {}).get("metrics", {}).get("best_psnr")
    desc = {
        "Q-A": "LayerNorm2d everywhere",
        "Q-F": "affine @ full-res, LayerNorm deeper",
        "Q-E": "affine everywhere",
        "Q-E1": "affine, half LR + long warmup",
        "Q-E2": "affine, half LR + clip 1.0",
        "Q-E3": "affine, half LR + clip + resid init 0.1",
    }
    for arm in ("Q-A", "Q-F", "Q-E", "Q-E1", "Q-E2", "Q-E3"):
        d = arms.get(arm)
        if not d:
            continue
        m = d["metrics"]
        psnr = m.get("best_psnr")
        delta = f"{psnr - qa:+.3f}" if (psnr and qa) else "—"
        final = m.get("final", {})
        L.append(
            f"| **{arm}** | {desc.get(arm,'')} | {psnr:.3f} | {delta} | "
            f"{LATENCY_MS.get(arm,float('nan')):.3f} | {SPEEDUP.get(arm,0):.2f}x | "
            f"{final.get('peak_vram_gb',0):.2f} GB | {m.get('iterations','—')} |")
    L.append("")

    if has_plot:
        L += ["## Validation curves", "",
              "![validation curves](norm_quality_curves.png)", "",
              "Panels: BSD68 PSNR, training loss (log), gradient norm, and mean "
              "activation magnitude at encoder level 0. The latter two are the "
              "trainability diagnostics — a capacity failure shows as a stable "
              "curve that plateaus lower, whereas a trainability failure shows as "
              "diverging or oscillating gradients and activations.", ""]

    L += ["## Diagnostics", ""]
    for arm, d in sorted(arms.items()):
        h = d["history"]
        if not h:
            continue
        gn = [r["grad_norm"] for r in h if "grad_norm" in r]
        acts = {k: r for k in h[-1] if k.startswith("act_") for r in [h[-1]]}
        diverged = any(r.get("diverged") for r in h)
        L.append(
            f"- **{arm}**: final loss {h[-1].get('loss', float('nan')):.5f}, "
            f"grad norm {gn[-1] if gn else float('nan'):.3f} "
            f"(min {min(gn) if gn else 0:.3f}, max {max(gn) if gn else 0:.3f}), "
            f"activations " +
            ", ".join(f"{k[4:]}={h[-1][k]:.3f}" for k in sorted(acts)) +
            ("  **DIVERGED**" if diverged else ""))
    L.append("")

    L += ["## Caveats", "",
          "- Ablation is on **w16_b8 (width 16) and denoising only**. Wider "
          "models have more channels per normalization and may behave "
          "differently; derain/dehaze may respond differently to the loss of "
          "normalization. If the decision were close, a spot-check on `w24_b8` "
          "and one non-denoising task would be warranted before locking.",
          "- **30k iterations, not to convergence.** A ranking at 30k is not "
          "guaranteed to hold at full training length.",
          "- Single seed. Differences smaller than seed-to-seed variance should "
          "not be over-read.", ""]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the Task 1.5b report.")
    ap.add_argument("--runs", default="runs/1p5b")
    ap.add_argument("--out", default="reports/norm_quality.md")
    args = ap.parse_args()

    arms = load_arms(REPO_ROOT / args.runs)
    if not arms:
        raise SystemExit(f"no completed arms under {args.runs}")
    has_plot = plot_curves(arms, REPO_ROOT / "reports" / "norm_quality_curves.png")
    (REPO_ROOT / args.out).write_text(build_report(arms, has_plot), encoding="utf-8")
    locked, why = decide(arms)
    print(f"[report] arms: {sorted(arms)}")
    print(f"[report] decision: {locked}")
    print(f"[report] {why}")
    print(f"[report] -> {args.out}")


if __name__ == "__main__":
    main()
