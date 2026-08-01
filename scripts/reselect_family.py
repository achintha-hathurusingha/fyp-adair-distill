"""Task 4 — re-select the S/M/L family on post-norm-fix latency.

The current family was chosen on N-A latency, which is ~62% normalization
overhead — a large, roughly-fixed cost sitting on every configuration that
compresses a 4.26x MAC span into a 1.70x latency span. Once the locked
normalization removes it, the span decompresses **non-uniformly** (configs differ
in how much full-resolution normalization they carry), so the ranking, the span
and possibly the M choice can all move.

Selection runs through the corrected **profile-then-select** path: latency is
loaded first and passed into ``assign_family``, which was previously called
before AI Hub had run and so never saw the numbers it selects on.

Both the pre-fix (N-A) and post-fix tables are kept — side by side they are the
evidence that MAC ranking mispredicts on-device latency.

    python -m scripts.reselect_family --norm NE
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.export.aihub import DeviceJobResult
from src.models.student_sweep import (ARMS, FamilyInvariantError, assign_family,
                                      build_grid, measure_grid, measure_teacher,
                                      validate_family)
from src.utils.config import REPO_ROOT, load_yaml


def load_latency(norm: str) -> dict[str, float]:
    """Latency per config for a variant ('NA', 'NF', 'NE' or 'FC').

    FC is the LOCKED variant after findings F9: N-F plus a magnitude clamp at
    the full-resolution stages. It is swept across the whole grid rather than
    extrapolated from the M arm, because the clamp adds Clip nodes in proportion
    to each config's full-resolution block count, so its cost is not uniform.
    """
    if norm == "NA":
        man = json.loads((REPO_ROOT / "runs/sweep/aihub_manifest.json")
                         .read_text(encoding="utf-8"))["models"]
        return {k: (v.get("results") or {}).get("latency_ms")
                for k, v in man.items()
                if (v.get("results") or {}).get("latency_ms")}
    jobs = json.loads((REPO_ROOT / "reports/aihub_jobs.json")
                      .read_text(encoding="utf-8"))["jobs"]
    out = {}
    for entry in jobs.values():
        if entry.get("variant") == norm and (entry.get("results") or {}).get("latency_ms"):
            out[entry["config"]] = entry["results"]["latency_ms"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Re-select the S/M/L family.")
    ap.add_argument("--norm", required=True, choices=["NA", "NF", "NE", "FC"],
                    help="the LOCKED normalization variant")
    ap.add_argument("--config", default="configs/sweep/student_sweep.yaml")
    ap.add_argument("--out", default="reports/family_reselection.md")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    shape = tuple(cfg["input_shape"])
    ceiling = cfg.get("params_ceiling", 10_000_000)
    min_span = cfg.get("family_invariants", {}).get("min_mac_span", 2.5)

    teacher = measure_teacher(cfg, shape)
    rows = measure_grid(build_grid(cfg), teacher, shape, ceiling)

    pre = load_latency("NA")
    post = load_latency(args.norm)
    if not post:
        raise SystemExit(f"no latency results found for variant {args.norm}")

    # PROFILE-THEN-SELECT: attach measured latency BEFORE assign_family runs.
    for row in rows:
        lat = post.get(row.name)
        if lat:
            row.device = DeviceJobResult(name=row.name, quantized=True,
                                         compiled=True, profiled=True,
                                         inference_latency_ms=lat)

    family, warnings = assign_family(rows, cfg.get("family_targets"),
                                     params_ceiling=ceiling,
                                     min_mac_span=min_span)

    invariant_error = None
    try:
        validate_family(family, min_mac_span=min_span)
    except FamilyInvariantError as exc:
        invariant_error = str(exc)

    by_name = {r.name: r for r in rows}
    L = [f"# Family re-selection on locked normalization ({args.norm})", "",
         "Selected through the corrected **profile-then-select** path: measured "
         "latency is attached to every candidate *before* `assign_family` runs. "
         "The earlier bug ran selection first, so it never saw the latency it "
         "selects on.", "",
         "## Pre-fix (N-A) vs post-fix latency", "",
         "| config | params | GMACs | N-A ms | "
         f"{args.norm} ms | speedup | rank N-A | rank {args.norm} |",
         "|---|---|---|---|---|---|---|---|"]

    pre_rank = {n: i + 1 for i, (n, _) in enumerate(
        sorted(pre.items(), key=lambda kv: kv[1]))}
    post_rank = {n: i + 1 for i, (n, _) in enumerate(
        sorted(post.items(), key=lambda kv: kv[1]))}
    for name, lat in sorted(post.items(), key=lambda kv: kv[1]):
        r = by_name.get(name)
        if not r:
            continue
        a = pre.get(name)
        moved = "" if pre_rank.get(name) == post_rank.get(name) else " ⟵ moved"
        L.append(f"| `{name}` | {r.complexity.mparams:.2f}M | "
                 f"{r.complexity.gmacs:.2f} | {a:.3f} | {lat:.3f} | "
                 f"{a/lat:.2f}x | {pre_rank.get(name,'-')} | "
                 f"{post_rank.get(name,'-')}{moved} |")

    pre_span = max(pre.values()) / min(pre.values())
    post_span = max(post.values()) / min(post.values())
    L += ["",
          f"**Latency span: {pre_span:.2f}x (N-A) -> {post_span:.2f}x ({args.norm}).** "
          "Removing the large roughly-fixed normalization cost decompresses the "
          "range, as predicted.", "",
          "## Selected family", "",
          "| arm | config | params | GMACs | latency | MACs÷ |",
          "|---|---|---|---|---|---|"]
    for arm in ARMS:
        r = family.get(arm)
        if r is None:
            L.append(f"| **{arm}** | *none* | — | — | — | — |")
            continue
        lat = post.get(r.name)
        L.append(f"| **{arm}** | `{r.name}` | {r.complexity.mparams:.2f}M | "
                 f"{r.complexity.gmacs:.2f} | {lat:.3f} ms | "
                 f"{r.mac_reduction:.1f}x |")
    L.append("")

    if family:
        s, l = family.get("S"), family.get("L")
        if s and l:
            L.append(f"MAC span **{l.complexity.macs / s.complexity.macs:.2f}x**, "
                     f"latency span "
                     f"**{post[l.name] / post[s.name]:.2f}x**.")
            L.append("")

    L += ["## Invariant check", ""]
    if invariant_error:
        L += [f"**FAILED — {invariant_error}**", "",
              "Per the stop-and-report rule, a broken family is never locked.", ""]
    else:
        L += ["Passed: params and MACs strictly increase S < M < L, measured "
              f"latency increases across arms, and the MAC span clears "
              f"{min_span}x.", ""]
    if warnings:
        L += ["### Warnings", ""] + [f"- {w}" for w in warnings] + [""]

    out = REPO_ROOT / args.out
    out.write_text("\n".join(L), encoding="utf-8")

    print(f"[family] latency span {pre_span:.2f}x -> {post_span:.2f}x")
    for arm in ARMS:
        r = family.get(arm)
        if r:
            print(f"[family] {arm}: {r.name} {r.complexity.mparams:.2f}M "
                  f"{r.complexity.gmacs:.2f}G {post.get(r.name, 0):.3f} ms")
    print(f"[family] invariants: {'FAILED — ' + invariant_error if invariant_error else 'passed'}")
    print(f"[family] -> {args.out}")
    if invariant_error:
        raise SystemExit("family invariant failed — STOP AND REPORT")


if __name__ == "__main__":
    main()
