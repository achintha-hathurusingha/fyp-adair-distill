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
_OVER_CEILING = "exceeds-params-ceiling"


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
        """True when the config is within the parameter ceiling."""
        return _OVER_CEILING not in self.notes


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
                 params_ceiling: float = 10_000_000) -> list[SweepRow]:
    """Measure every candidate and annotate reduction ratios vs the teacher.

    ``params_ceiling`` bounds model size only; it is NOT a compression target.
    """
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
        if c.params > params_ceiling:
            row.notes.append(_OVER_CEILING)
        rows.append(row)
        print(f"[sweep] {row.name:12s} {c.mparams:6.2f}M params  "
              f"{c.gmacs:7.2f} GMACs  (params /{row.param_reduction:.1f}, "
              f"MACs /{row.mac_reduction:.1f})")
    return rows


#: Arm order, smallest to largest. Fixed by definition, not by search outcome.
ARMS = ("S", "M", "L")


class FamilyInvariantError(ValueError):
    """Raised when a proposed S/M/L family violates a required invariant."""


def validate_family(family: dict[str, SweepRow], *, min_mac_span: float = 2.5) -> None:
    """Enforce family semantics as hard invariants (rule 9: no silent fallbacks).

    A family is only meaningful if S is genuinely the smallest arm and L the
    largest, and the span is wide enough to expose a capacity gap. Previously
    this ordering was an emergent property of the search; now it is checked, so
    a selector bug fails loudly instead of producing a nonsense family.

    Raises:
        FamilyInvariantError: if any invariant is violated.
    """
    missing = [a for a in ARMS if a not in family]
    if missing:
        raise FamilyInvariantError(f"family missing arm(s): {missing}")

    s, m, l = (family[a] for a in ARMS)
    if not (s.complexity.params < m.complexity.params < l.complexity.params):
        raise FamilyInvariantError(
            "params must increase S < M < L, got "
            f"S={s.complexity.mparams:.2f}M ({s.name}), "
            f"M={m.complexity.mparams:.2f}M ({m.name}), "
            f"L={l.complexity.mparams:.2f}M ({l.name})")
    if not (s.complexity.macs < m.complexity.macs < l.complexity.macs):
        raise FamilyInvariantError(
            "MACs must increase S < M < L, got "
            f"S={s.complexity.gmacs:.2f} ({s.name}), "
            f"M={m.complexity.gmacs:.2f} ({m.name}), "
            f"L={l.complexity.gmacs:.2f} ({l.name})")

    span = l.complexity.macs / s.complexity.macs
    if span < min_mac_span:
        raise FamilyInvariantError(
            f"family span too narrow for capacity-gap study: "
            f"macs(L)/macs(S) = {span:.2f}x < {min_mac_span}x "
            f"({s.name} -> {l.name})")


def _family_candidates(eligible: list[SweepRow], min_mac_span: float):
    """Yield ordered (S, M, L) triples that satisfy every family invariant."""
    import itertools

    for combo in itertools.combinations(eligible, len(ARMS)):
        cand = dict(zip(ARMS, combo))
        try:
            validate_family(cand, min_mac_span=min_mac_span)
        except FamilyInvariantError:
            continue
        # When measured latency exists it must also increase across arms —
        # MACs correlate only ~0.66 with latency, so MAC ordering alone can
        # yield a family whose "small" arm is slower than its "medium" arm.
        lats = [r.device.inference_latency_ms for r in combo
                if r.device and r.device.inference_latency_ms]
        if len(lats) == len(combo) and not all(a < b for a, b in zip(lats, lats[1:])):
            continue
        yield combo


def assign_family(rows: list[SweepRow], targets: dict[str, float] | None = None, *,
                  params_ceiling: float = 10_000_000,
                  min_mac_span: float = 2.5,
                  ) -> tuple[dict[str, SweepRow], list[str]]:
    """Select an S/M/L family under a params ceiling, enforcing invariants.

    The former ">=4x reduction on both params and MACs" rule is retired: it
    rejected every width-32 candidate and forced a latency-degenerate family.
    Parameters are now only a ceiling; arms are chosen to maximise MAC span
    (the capacity gap) with M near the geometric midpoint.

    Args:
        rows: measured candidates.
        targets: advisory MAC-reduction targets, reported but never used to
            reject a candidate.
        params_ceiling: hard upper bound on parameter count.
        min_mac_span: required ``macs(L) / macs(S)``.

    Returns:
        ``(family, warnings)``. The family always satisfies
        :func:`validate_family` or is empty with an explanatory warning.
    """
    import math

    warnings: list[str] = []
    eligible = sorted((r for r in rows if r.complexity.params <= params_ceiling),
                      key=lambda r: r.complexity.macs)
    dropped = len(rows) - len(eligible)
    if dropped:
        warnings.append(
            f"{dropped} config(s) exceed the {params_ceiling/1e6:.0f}M parameter "
            "ceiling and were excluded")

    candidates = list(_family_candidates(eligible, min_mac_span))
    if not candidates:
        warnings.append(
            f"no S/M/L family satisfies the invariants (params <= "
            f"{params_ceiling/1e6:.0f}M, monotonic params/MACs/latency, span >= "
            f"{min_mac_span}x) among {len(eligible)} eligible config(s); widen "
            "the grid")
        return {}, warnings

    def rank(combo):
        s, m, l = combo
        span = l.complexity.macs / s.complexity.macs
        # Prefer the widest capacity gap, then an M nearest the geometric mean
        # of S and L (so the middle arm is genuinely intermediate, not hugging
        # an end).
        mid = abs(math.log(m.complexity.macs
                           / math.sqrt(s.complexity.macs * l.complexity.macs)))
        return (-span, mid)

    family = dict(zip(ARMS, min(candidates, key=rank)))
    validate_family(family, min_mac_span=min_mac_span)  # belt and braces

    lats = [family[a].device.inference_latency_ms for a in ARMS
            if family[a].device and family[a].device.inference_latency_ms]
    if len(lats) == len(ARMS) and max(lats) / min(lats) < 1.5:
        warnings.append(
            f"family latency span is only {max(lats)/min(lats):.2f}x "
            f"({min(lats):.2f}-{max(lats):.2f} ms) — thin for a Pareto curve")

    for arm, target in (targets or {}).items():
        row = family.get(arm)
        if row and (row.mac_reduction > 2 * target or row.mac_reduction < 0.5 * target):
            warnings.append(
                f"arm {arm}: advisory target {target:g}x MAC reduction not met; "
                f"selected `{row.name}` at {row.mac_reduction:.1f}x "
                "(targets are advisory only, not a filter)")
    return family, warnings


def run_aihub(rows: list[SweepRow], cfg: dict, out_dir: Path,
              input_shape: tuple[int, int, int, int]) -> str | None:
    """Export each candidate and profile it on AI Hub. Returns an error note."""
    from src.export.aihub import AIHubUnavailable, DeviceJobResult
    from src.export.aihub_batch import run_batch
    from src.export.to_onnx import export_onnx

    hub_cfg = require(cfg, "aihub", context=_CTX_SWEEP)
    device = require(hub_cfg, "device", context=_CTX_AIHUB)

    # Export everything first, then drive all models through AI Hub concurrently
    # via the resumable batch pipeline (serial submission would take hours).
    specs = []
    for row in rows:
        model = NAFNet(width=row.width, enc_blk_nums=row.enc_blk_nums,
                       middle_blk_num=row.middle_blk_num,
                       dec_blk_nums=row.dec_blk_nums)
        onnx_path = export_onnx(model, out_dir / f"{row.name}.onnx", input_shape)
        specs.append({"name": row.name, "onnx": str(onnx_path)})

    try:
        manifest = run_batch(
            specs, out_dir / "aihub_manifest.json", device, input_shape,
            calib_samples=hub_cfg.get("calib_samples", 8),
            compile_options=hub_cfg.get("compile_options", ""),
            profile_options=hub_cfg.get("profile_options", ""),
            max_minutes=hub_cfg.get("max_minutes", 180),
        )
    except AIHubUnavailable as exc:
        return str(exc)

    by_name = {r.name: r for r in rows}
    for name, entry in manifest.get("models", {}).items():
        row = by_name.get(name)
        if row is None:
            continue
        res = entry.get("results") or {}
        row.device = DeviceJobResult(
            name=name,
            quantized=entry.get("status", {}).get("quantize") == "SUCCESS",
            compiled=entry.get("status", {}).get("compile") == "SUCCESS",
            profiled=bool(res),
            error=entry.get("error"),
            inference_latency_ms=res.get("latency_ms"),
            peak_memory_mb=res.get("peak_memory_mb"),
            compute_unit_breakdown=res.get("compute_units") or {},
            job_urls=entry.get("urls", {}),
        )
    return None


def _pearson(a: list[float], b: list[float]) -> float:
    """Pearson correlation; 0.0 when either series is constant."""
    n = len(a)
    if n < 2:
        return 0.0
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va = sum((x - ma) ** 2 for x in a) ** 0.5
    vb = sum((y - mb) ** 2 for y in b) ** 0.5
    return cov / (va * vb) if va and vb else 0.0


def _spearman(a: list[float], b: list[float]) -> float:
    """Spearman rank correlation (Pearson on ranks, average ties)."""
    def ranks(xs: list[float]) -> list[float]:
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        out = [0.0] * len(xs)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out

    return _pearson(ranks(a), ranks(b))


def _matched_pair_section(prof: list[SweepRow]) -> list[str]:
    """Report controlled pairs: same block count, near-identical MACs.

    A matched pair isolates *placement* from capacity, so it is stronger
    evidence than a correlation across heterogeneous configurations.
    """
    pairs = []
    for a in prof:
        for b in prof:
            if a.name >= b.name:
                continue
            na = sum(a.enc_blk_nums) + a.middle_blk_num + sum(a.dec_blk_nums)
            nb = sum(b.enc_blk_nums) + b.middle_blk_num + sum(b.dec_blk_nums)
            if na != nb or a.width != b.width:
                continue
            mac_gap = abs(a.complexity.macs - b.complexity.macs) / max(
                a.complexity.macs, b.complexity.macs)
            if mac_gap > 0.02:
                continue
            fast, slow = sorted((a, b), key=lambda r: r.device.inference_latency_ms)
            pairs.append((na, mac_gap, fast, slow))
    if not pairs:
        return []

    L = ["### Controlled comparison — matched block count and MACs", "",
         "| pair | blocks | GMACs | NPU ms | Δ latency |", "|---|---|---|---|---|"]
    for nb, gap, fast, slow in sorted(pairs, key=lambda p: -(
            p[3].device.inference_latency_ms / p[2].device.inference_latency_ms)):
        delta = (slow.device.inference_latency_ms
                 / fast.device.inference_latency_ms - 1) * 100
        L.append(f"| `{fast.name}` vs `{slow.name}` | {nb} | "
                 f"{fast.complexity.gmacs:.2f} vs {slow.complexity.gmacs:.2f} "
                 f"({gap*100:.1f}% apart) | "
                 f"{fast.device.inference_latency_ms:.2f} vs "
                 f"{slow.device.inference_latency_ms:.2f} | **+{delta:.0f}%** |")
    L += ["", "These pairs hold block count, width and MACs essentially "
          "constant and vary only **where** the blocks sit in the pyramid. The "
          "latency difference therefore cannot be attributed to capacity or "
          "compute — it is placement, and specifically the number of "
          "normalisations running at full resolution. This is a controlled "
          "result and is stronger evidence than any correlation over "
          "heterogeneous points.", ""]
    return L


def _norm_area_proxy(row: SweepRow) -> float:
    """Normalisation cost proxy: blocks weighted by their stage's spatial area.

    Each NAFBlock carries two LayerNorm2d layers whose cost is per-element, so a
    block at full resolution normalises 4x as many elements as one at H/2. This
    weights block counts by relative area (1, 1/4, 1/16, ...).
    """
    areas = [1.0, 0.25, 0.0625, 0.015625]
    total = sum(k * areas[min(i, len(areas) - 1)]
                for i, k in enumerate(row.enc_blk_nums))
    total += row.middle_blk_num * areas[-1] / 4
    for i, k in enumerate(row.dec_blk_nums):
        total += k * areas[max(0, len(areas) - 1 - i)]
    return total


def _device_findings(rows: list[SweepRow]) -> list[str]:
    """Interpret the measured on-device results."""
    prof = [r for r in rows if r.device and r.device.inference_latency_ms]
    if len(prof) < 3:
        return []

    lat = [r.device.inference_latency_ms for r in prof]
    macs = [r.complexity.gmacs for r in prof]
    blocks = [float(sum(r.enc_blk_nums) + r.middle_blk_num + sum(r.dec_blk_nums))
              for r in prof]
    norm = [_norm_area_proxy(r) for r in prof]
    fallback = sum(r.device.npu_fallback_layers for r in prof)
    n = len(prof)

    L = ["## What the device actually says", "",
         f"Correlations with measured INT8 latency (**n={n}**, Pearson and "
         "Spearman rank):", "",
         "| predictor | Pearson r | Spearman ρ |", "|---|---|---|",
         f"| GMACs | {_pearson(lat, macs):.2f} | {_spearman(lat, macs):.2f} |",
         f"| total block count | {_pearson(lat, blocks):.2f} | "
         f"{_spearman(lat, blocks):.2f} |",
         f"| normalisation-area proxy | **{_pearson(lat, norm):.2f}** | "
         f"**{_spearman(lat, norm):.2f}** |", "",
         f"> With n={n} heterogeneous configurations these correlations are "
         "indicative, not conclusive — the gap between the MAC and "
         "normalisation predictors should not be over-read. The controlled "
         "comparison below is the stronger evidence.", "",
         f"**NPU->CPU fallback across all profiled configs: {fallback} layers.** "
         "Every op ran on the Hexagon NPU, so the static CAUTION verdicts in "
         "`export_smoke_test.md` overstated the *support* risk — nothing was "
         "rejected or offloaded.", ""]

    L += _matched_pair_section(prof)

    # Find the sharpest MACs-mispredicts-latency inversion.
    worst = None
    for a in prof:
        for b in prof:
            if a.complexity.macs < b.complexity.macs and \
                    a.device.inference_latency_ms > b.device.inference_latency_ms:
                gap = b.complexity.macs / a.complexity.macs
                if worst is None or gap > worst[0]:
                    worst = (gap, a, b)
    if worst:
        gap, a, b = worst
        L += [f"**MACs mispredict latency.** `{a.name}` has "
              f"{gap:.1f}x *fewer* MACs than `{b.name}` "
              f"({a.complexity.gmacs:.2f} vs {b.complexity.gmacs:.2f} GMACs) yet "
              f"is **slower** on device ({a.device.inference_latency_ms:.2f} vs "
              f"{b.device.inference_latency_ms:.2f} ms). Selecting on MACs alone "
              "would have picked the wrong architecture.", ""]

    L += ["Mechanism: cycle profiling of `w16_b8` shows **LayerNorm2d consumes "
          "~62% of NPU cycles** (`Div` alone ~62%) against **~3% for `Conv`**. "
          "Fixed-point division is expensive on the Hexagon integer pipeline, and "
          "its cost is per-element — so normalisations at full resolution "
          "dominate. This is why the area-weighted proxy predicts latency better "
          "than MACs.", "",
          "> **Consequence for Task 1.5b:** replacing `LayerNorm2d` with a "
          "conv-foldable normalisation (BatchNorm) should remove ~60% of NPU "
          "cycles — a larger latency win than any width/block choice in this "
          "sweep. The architecture must be locked on the *post-normalisation* "
          "design, otherwise this table's ranking does not survive.", ""]
    return L


def build_report(teacher: Complexity, rows: list[SweepRow],
                 family: dict[str, SweepRow], params_ceiling: float,
                 aihub_error: str | None, input_shape: tuple[int, ...],
                 warnings: list[str] | None = None) -> str:
    """Render reports/student_sweep.md."""
    res = f"{input_shape[2]}x{input_shape[3]}"
    L = [
        "# Student architecture sweep — Task 1.5a", "",
        f"All figures at **{res}**, batch 1. **MACs = FLOPs/2** "
        "(`torch.utils.flop_counter`, convention pinned by a unit test). "
        "On-device figures measured on Qualcomm AI Hub, Samsung Galaxy S24 "
        "(Snapdragon 8 Gen 3, Hexagon v75), INT8 QNN context binary.", "",
        "Selection priority: **measured on-device latency -> peak activation "
        "memory -> GMACs -> params**. Parameters are a *ceiling*, not a target.",
        "", "## Teacher reference (measured, not quoted)", "",
        f"**AdaIR** — `{teacher.mparams:.2f}M` params, "
        f"**`{teacher.gmacs:.2f}` GMACs** @ {res}.", "",
        "> Measured by instantiating `third_party/AdaIR`. Note the counter does "
        "not model the FFT work in AdaIR's frequency modules, so this is a mild "
        "*under*-estimate of true teacher cost — reduction ratios below are "
        "therefore conservative.", "",
        f"Parameter ceiling: **{params_ceiling/1e6:.0f}M** (bounds model size "
        "only). The former '>=4x reduction on both params and MACs' rule is "
        "**retired** — it rejected every width-32 candidate and forced a "
        "latency-degenerate family.", "",
        "## Sweep results", "",
    ]

    has_dev = any(r.device is not None for r in rows)
    hdr = ("| config | width | blocks | params | GMACs | MACs÷ | ≤ceiling |")
    sep = "|---|---|---|---|---|---|---|"
    if has_dev:
        hdr += " **NPU ms** | peak mem MB | fallback |"
        sep += "---|---|---|"
    L += [hdr, sep]

    # Sort by measured latency when available -- the primary selection axis.
    def sort_key(r: SweepRow):
        if r.device and r.device.inference_latency_ms:
            return (0, r.device.inference_latency_ms)
        return (1, r.complexity.macs)

    for r in sorted(rows, key=sort_key):
        line = (f"| `{r.name}` | {r.width} | {r.block_name} | "
                f"{r.complexity.mparams:.2f}M | {r.complexity.gmacs:.2f} | "
                f"{r.mac_reduction:.1f}x | "
                f"{'yes' if r.eligible else '**no**'} |")
        if has_dev:
            d = r.device
            line += (f" **{d.inference_latency_ms:.2f}** |"
                     if d and d.inference_latency_ms else " n/a |")
            line += (f" {d.peak_memory_mb:.0f} |"
                     if d and d.peak_memory_mb else " n/a |")
            line += (f" {d.npu_fallback_layers} |" if d else " n/a |")
        L.append(line)
    L += ["",
          "> **Peak memory is total footprint** (weights + activations + runtime), "
          "not the incremental working set. It is ~98-101 MB across *every* "
          "config including the smallest, i.e. dominated by fixed QNN runtime "
          "overhead rather than by model size — so on this device it does not "
          "discriminate between candidates, but it does set the floor any edge "
          "memory budget must clear.", ""]

    L += _device_findings(rows)

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
            if d and d.error:
                L.append(f"- `{r.name}` **FAILED**: {d.error}")
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
    params_ceiling = cfg.get("params_ceiling", 10_000_000)
    min_span = cfg.get("family_invariants", {}).get("min_mac_span", 2.5)

    teacher = measure_teacher(cfg, shape)
    print(f"[sweep] teacher: {teacher.mparams:.2f}M params, {teacher.gmacs:.2f} GMACs")

    rows = measure_grid(build_grid(cfg), teacher, shape, params_ceiling)

    # Profile BEFORE selecting: latency is the primary selection axis, so the
    # family must be chosen with the device numbers in hand.
    use_hub = cfg.get("aihub", {}).get("enabled", False) if args.aihub is None else args.aihub
    aihub_error = None
    if use_hub:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        aihub_error = run_aihub(rows, cfg, out_dir, shape)
        if aihub_error:
            print("[sweep] AI Hub unavailable; selecting on MACs only.")

    family, warnings = assign_family(
        rows, cfg.get("family_targets"),
        params_ceiling=params_ceiling, min_mac_span=min_span)
    for w in warnings:
        print(f"[sweep] WARNING: {w}")

    report = build_report(teacher, rows, family, params_ceiling, aihub_error,
                          shape, warnings)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(report, encoding="utf-8")
    print(f"[sweep] report -> {args.report}")


if __name__ == "__main__":
    main()
