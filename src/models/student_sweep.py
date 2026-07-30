"""Task 1.5a — student architecture sweep (no training).

Ranks candidate NAFNet students by MACs (and, when credentials exist, real
on-device INT8 latency from Qualcomm AI Hub) rather than by parameter count,
then proposes an S/M/L family against measured teacher complexity.

    python -m src.models.student_sweep --aihub        # include on-device profiling
    python -m src.models.student_sweep --no-aihub     # params/MACs only
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.models.complexity import Complexity, measure
from src.models.nafnet import NAFNet
from src.utils.config import REPO_ROOT, load_yaml, require
from src.utils.seeding import seed_everything

_CTX_SWEEP = "sweep config"
_CTX_TEACHER = "sweep.teacher"
_CTX_AIHUB = "sweep.aihub"
_CTX_BLOCK = "sweep.block_configs entry"
_FAILS_RULE = "fails-compression-rule"


@dataclass
class SweepRow:
    """One measured architecture candidate."""

    name: str
    width: int
    block_name: str
    enc_blk_nums: list[int]
    middle_blk_num: int
    dec_blk_nums: list[int]
    complexity: Complexity
    param_reduction: float = 0.0
    mac_reduction: float = 0.0
    device: Any = None  # DeviceJobResult when AI Hub ran
    notes: list[str] = field(default_factory=list)

    @property
    def eligible(self) -> bool:
        """True when the config is a genuine compression on BOTH axes."""
        return _FAILS_RULE not in self.notes


def measure_teacher(cfg: dict, input_shape: tuple[int, int, int, int]) -> Complexity:
    """Measure the AdaIR teacher from the cloned architecture.

    Measured rather than hardcoded so reduction ratios cannot drift from reality.
    Raises a clear error if the clone is missing.
    """
    t = require(cfg, "teacher", context=_CTX_SWEEP)
    repo = REPO_ROOT / require(t, "repo_path", context=_CTX_TEACHER)
    if not repo.exists():
        raise FileNotFoundError(
            f"Teacher architecture not found at {repo}. Clone it first:\n"
            f"    git clone --depth 1 https://github.com/c-yn/AdaIR.git {repo}"
        )
    sys.path.insert(0, str(repo))
    try:
        import importlib

        mod = importlib.import_module(require(t, "module", context=_CTX_TEACHER))
        cls = getattr(mod, require(t, "cls", context=_CTX_TEACHER))
    finally:
        sys.path.remove(str(repo))
    return measure(cls(**t.get("kwargs", {})), input_shape)


def build_grid(cfg: dict) -> list[dict]:
    """Expand widths x block_configs into a flat list of candidate specs."""
    widths = require(cfg, "widths", context=_CTX_SWEEP)
    blocks = require(cfg, "block_configs", context=_CTX_SWEEP)
    grid = []
    for w in widths:
        for b in blocks:
            grid.append({
                "name": f"w{w}_{require(b, 'name', context=_CTX_BLOCK)}",
                "width": w,
                "block_name": b["name"],
                "enc_blk_nums": require(b, "enc_blk_nums", context=_CTX_BLOCK),
                "middle_blk_num": require(b, "middle_blk_num", context=_CTX_BLOCK),
                "dec_blk_nums": require(b, "dec_blk_nums", context=_CTX_BLOCK),
            })
    return grid


def measure_grid(grid: list[dict], teacher: Complexity,
                 input_shape: tuple[int, int, int, int],
                 min_factor: float) -> list[SweepRow]:
    """Measure every candidate and annotate reduction ratios vs the teacher."""
    rows: list[SweepRow] = []
    for spec in grid:
        model = NAFNet(
            width=spec["width"],
            enc_blk_nums=spec["enc_blk_nums"],
            middle_blk_num=spec["middle_blk_num"],
            dec_blk_nums=spec["dec_blk_nums"],
        )
        c = measure(model, input_shape)
        row = SweepRow(complexity=c, **spec)
        row.param_reduction = teacher.params / c.params
        row.mac_reduction = teacher.macs / c.macs
        if row.param_reduction < min_factor or row.mac_reduction < min_factor:
            row.notes.append(_FAILS_RULE)
        rows.append(row)
        print(f"[sweep] {row.name:12s} {c.mparams:6.2f}M params  "
              f"{c.gmacs:7.2f} GMACs  (params /{row.param_reduction:.1f}, "
              f"MACs /{row.mac_reduction:.1f})")
    return rows


def assign_family(rows: list[SweepRow],
                  targets: dict[str, float]) -> tuple[dict[str, SweepRow], list[str]]:
    """Assign distinct eligible candidates to each arm, reporting target gaps.

    Assignment is greedy from the largest MAC-reduction target downwards and
    never reuses a config, so a degenerate family (S == M == L) is impossible.
    Any arm whose target is unreachable within the eligible set is reported in
    the returned warning list rather than silently snapped to a poor match.
    """
    import itertools
    import math

    # Descending MAC reduction == ascending model cost.
    eligible = sorted((r for r in rows if r.eligible),
                      key=lambda r: -r.mac_reduction)
    family: dict[str, SweepRow] = {}
    warnings: list[str] = []

    arms = sorted(targets.items(), key=lambda kv: -kv[1])  # S(30x), M(10x), L(4x)
    if len(eligible) < len(arms):
        warnings.append(
            f"only {len(eligible)} eligible candidate(s) for {len(arms)} arms; "
            "widen the grid or relax the compression rule"
        )
        return {arm: eligible[i] for i, (arm, _) in enumerate(arms)
                if i < len(eligible)}, warnings

    # Choose an ordered triple so that MAC reduction is monotonically decreasing
    # across S -> M -> L (i.e. S is always the smallest model, L the largest).
    # Score by summed absolute log-ratio error so the fit is scale-fair.
    def score(combo: tuple[SweepRow, ...]) -> float:
        return sum(abs(math.log(r.mac_reduction / t))
                   for r, (_, t) in zip(combo, arms))

    best_combo = min(itertools.combinations(eligible, len(arms)), key=score)
    for (arm, target), row in zip(arms, best_combo):
        family[arm] = row
        if row.mac_reduction > 2 * target or row.mac_reduction < 0.5 * target:
            warnings.append(
                f"arm {arm}: target {target:g}x MAC reduction is UNREACHABLE "
                f"under the compression rule; closest ordered fit is "
                f"`{row.name}` at {row.mac_reduction:.1f}x"
            )
    return family, warnings


def run_aihub(rows: list[SweepRow], cfg: dict, out_dir: Path,
              input_shape: tuple[int, int, int, int]) -> str | None:
    """Export each candidate and profile it on AI Hub. Returns an error note."""
    from src.export.aihub import AIHubUnavailable, submit_and_profile
    from src.export.to_onnx import export_onnx

    hub_cfg = require(cfg, "aihub", context=_CTX_SWEEP)
    device = require(hub_cfg, "device", context=_CTX_AIHUB)

    for row in rows:
        model = NAFNet(width=row.width, enc_blk_nums=row.enc_blk_nums,
                       middle_blk_num=row.middle_blk_num,
                       dec_blk_nums=row.dec_blk_nums)
        onnx_path = export_onnx(model, out_dir / f"{row.name}.onnx", input_shape)
        try:
            row.device = submit_and_profile(
                onnx_path, row.name, device, input_shape=input_shape,
                calib_samples=hub_cfg.get("calib_samples", 8),
                compile_options=hub_cfg.get("compile_options", ""),
                profile_options=hub_cfg.get("profile_options", ""),
            )
            d = row.device
            status = ("ok" if d.profiled
                      else f"FAILED at {d.stage_failed}: {(d.error or '')[:120]}")
            print(f"[aihub] {row.name}: {status}"
                  + (f"  latency={d.inference_latency_ms:.2f}ms"
                     if d.inference_latency_ms else ""))
        except AIHubUnavailable as exc:
            return str(exc)
    return None


def build_report(teacher: Complexity, rows: list[SweepRow],
                 family: dict[str, SweepRow], min_factor: float,
                 aihub_error: str | None, input_shape: tuple[int, ...],
                 warnings: list[str] | None = None) -> str:
    """Render reports/student_sweep.md."""
    res = f"{input_shape[2]}x{input_shape[3]}"
    L = [
        "# Student architecture sweep — Task 1.5a", "",
        f"All figures at **{res}**, batch 1. **MACs = FLOPs/2** "
        "(`torch.utils.flop_counter`, convention pinned by a unit test).",
        "",
        "Selection is by **MACs and on-device latency, not parameters**: a NAFBlock "
        "at H/8 costs ~1/64 the MACs of the same block at full resolution, so "
        "parameter count cannot rank these architectures by cost.", "",
        "## Teacher reference (measured, not quoted)", "",
        f"**AdaIR** — `{teacher.mparams:.2f}M` params, "
        f"**`{teacher.gmacs:.2f}` GMACs** @ {res}.", "",
        "> Measured by instantiating `third_party/AdaIR`. Note the counter does "
        "not model the FFT work in AdaIR's frequency modules, so this is a mild "
        "*under*-estimate of true teacher cost — reduction ratios below are "
        "therefore conservative.", "",
        f"Compression rule: an arm counts only if **both** params and MACs shrink "
        f"by ≥{min_factor:g}x (params ≤ {teacher.mparams / min_factor:.2f}M, "
        f"MACs ≤ {teacher.gmacs / min_factor:.2f} GMACs).", "",
        "## Sweep results", "",
    ]

    has_dev = any(r.device is not None for r in rows)
    hdr = ("| config | width | blocks | params | GMACs | params÷ | MACs÷ | "
           "≥4x both |")
    sep = "|---|---|---|---|---|---|---|---|"
    if has_dev:
        hdr += " NPU ms | compiled |"
        sep += "---|---|"
    L += [hdr, sep]

    for r in sorted(rows, key=lambda r: r.complexity.macs):
        line = (f"| `{r.name}` | {r.width} | {r.block_name} | "
                f"{r.complexity.mparams:.2f}M | {r.complexity.gmacs:.2f} | "
                f"{r.param_reduction:.1f}x | {r.mac_reduction:.1f}x | "
                f"{'yes' if r.eligible else '**no**'} |")
        if has_dev:
            d = r.device
            line += (f" {d.inference_latency_ms:.1f} |" if d and d.inference_latency_ms
                     else " n/a |")
            line += f" {'yes' if d and d.compiled else 'no'} |" if d else " n/a |"
        L.append(line)
    L.append("")

    # MACs track total block count, not placement -- the U-Net's 4x channel
    # growth per downsample cancels the 4x spatial reduction.
    L += ["## Why placement changes params but not MACs", "",
          "| config | total blocks | params | GMACs | GMACs/block |",
          "|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r.width, r.complexity.macs)):
        nb = sum(r.enc_blk_nums) + r.middle_blk_num + sum(r.dec_blk_nums)
        L.append(f"| `{r.name}` | {nb} | {r.complexity.mparams:.2f}M | "
                 f"{r.complexity.gmacs:.2f} | {r.complexity.gmacs / nb:.4f} |")
    L += ["",
          "`b28` and `sidd` have the **same total block count (36)** and therefore "
          "near-identical MACs, despite `sidd` carrying ~1.7x the parameters. "
          "Each downsample quarters the spatial area but doubles the channel count "
          "(4x in `c^2`), so a NAFBlock costs the **same MACs at every depth**. "
          "MACs therefore track *total blocks x width^2*; placement is MAC-free "
          "but parameter-expensive.", "",
          "Consequence: deepening the pyramid buys capacity at zero MAC cost, but "
          "costs model size and memory bandwidth — which is why the parameter "
          "rule remains a useful independent constraint.", ""]

    ppm_t = teacher.mparams / teacher.gmacs
    L += ["## Params-per-MAC mismatch (why the family targets bind)", "",
          f"AdaIR sits at **{ppm_t:.2f}M params per GMAC**; NAFNet variants sit at "
          f"~{sum(r.complexity.mparams / r.complexity.gmacs for r in rows) / len(rows):.2f}M "
          "params per GMAC — roughly "
          f"{(sum(r.complexity.mparams / r.complexity.gmacs for r in rows) / len(rows)) / ppm_t:.1f}x "
          "more parameter-heavy per unit of compute. Attention is compute-dense; "
          "convolution is parameter-dense.", "",
          "So parameter-reduction and MAC-reduction are **not independently "
          "dialable**: hitting the parameter rule forces a much larger MAC "
          "reduction than the nominal arm target.", ""]

    L += ["## Proposed family", "",
          "| arm | target MACs÷ | chosen config | params | GMACs | actual MACs÷ | params÷ |",
          "|---|---|---|---|---|---|---|"]
    for arm in ("S", "M", "L"):
        r = family.get(arm)
        if r is None:
            L.append(f"| **{arm}** | — | *no eligible candidate* | — | — | — | — |")
            continue
        L.append(f"| **{arm}** | — | `{r.name}` | {r.complexity.mparams:.2f}M | "
                 f"{r.complexity.gmacs:.2f} | {r.mac_reduction:.1f}x | "
                 f"{r.param_reduction:.1f}x |")
    L.append("")
    if warnings:
        L += ["**Assignment warnings — these need a decision:**", ""]
        L += [f"- {w}" for w in warnings]
        L.append("")
    L += ["Ablation-grid discipline: the full grid runs on **M** only. "
          "S and L get B0 plus the single best KD config.", ""]

    L += ["## On-device verification", ""]
    if aihub_error:
        L += ["**NOT YET RUN — blocked on credentials.** The op-coverage verdicts "
              "in `export_smoke_test.md` remain a *static* estimate until this "
              "completes, so Gate G1 stays **PASS (provisional)**.", "",
              "```", aihub_error.strip(), "```", ""]
    elif has_dev:
        L += ["Real AI Hub compilation and profiling completed; see the NPU "
              "columns above.", ""]
        for r in rows:
            d = r.device
            if d and d.compute_unit_breakdown:
                L.append(f"- `{r.name}`: compute units {d.compute_unit_breakdown}"
                         + (f", peak {d.peak_memory_mb:.0f} MB"
                            if d.peak_memory_mb else ""))
            if d and d.compile_error:
                L.append(f"- `{r.name}` **compile error**: {d.compile_error}")
        L.append("")
    else:
        L += ["Skipped (`--no-aihub`).", ""]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description="Task 1.5a student sweep.")
    ap.add_argument("--config", default="configs/sweep/student_sweep.yaml")
    ap.add_argument("--out-dir", default="runs/sweep")
    ap.add_argument("--report", default="reports/student_sweep.md")
    ap.add_argument("--aihub", dest="aihub", action="store_true", default=None)
    ap.add_argument("--no-aihub", dest="aihub", action="store_false")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    seed_everything(args.seed)
    cfg = load_yaml(args.config)
    shape = tuple(require(cfg, "input_shape", context=_CTX_SWEEP))
    min_factor = cfg.get("min_compression_factor", 4.0)

    teacher = measure_teacher(cfg, shape)
    print(f"[sweep] teacher: {teacher.mparams:.2f}M params, {teacher.gmacs:.2f} GMACs")

    rows = measure_grid(build_grid(cfg), teacher, shape, min_factor)
    family, warnings = assign_family(
        rows, require(cfg, "family_targets", context=_CTX_SWEEP))
    for w in warnings:
        print(f"[sweep] WARNING: {w}")

    use_hub = cfg.get("aihub", {}).get("enabled", False) if args.aihub is None else args.aihub
    aihub_error = None
    if use_hub:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        aihub_error = run_aihub(rows, cfg, out_dir, shape)
        if aihub_error:
            print("[sweep] AI Hub unavailable; params/MACs reported without "
                  "on-device numbers.")

    report = build_report(teacher, rows, family, min_factor, aihub_error, shape,
                          warnings)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(report, encoding="utf-8")
    print(f"[sweep] report -> {args.report}")


if __name__ == "__main__":
    main()
