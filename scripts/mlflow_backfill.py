"""Import every existing run directory into MLflow.

    python scripts/mlflow_backfill.py --dry-run
    python scripts/mlflow_backfill.py --tracking-uri http://<mlflow-host>:5000
    python scripts/mlflow_backfill.py --tracking-uri file:./mlruns    # local test

**Verified end to end** against devon's 16 real experiments into a
SQLite-backed store: full curves, params, artifacts and tags all land.
Note MLflow 3.15 has deprecated the plain file store, so the backend must
be SQLite or Postgres even for a local test.

**Reads what the project already writes.** Every run directory carries a
resolved ``config.yaml``, ``git_commit.txt``, ``metrics.json``, ``history.json``
and ``train.log``. That is already an experiment record; it is simply not
rendered anywhere. This imports it, so MLflow starts with the project's whole
history rather than only whatever runs after it is installed.

**Idempotent.** Each MLflow run is tagged with the absolute run-directory path;
re-running skips directories already imported, so this can be re-run after new
experiments finish without creating duplicates.

**The run directory stays the source of truth.** MLflow is a view over it, not
a replacement — if the two ever disagree, the files on disk are correct, and
that ordering matters because a tracking server is one more thing that can be
misconfigured.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from src.utils.config import REPO_ROOT

#: Runs are grouped by their OUT-ROOT -- the directory chosen at launch time
#: (`--out-root runs/b0_final`) -- not by the leaf directory name. Prefix
#: matching on the leaf was the first attempt and it put 16 of 33 runs into
#: `misc` while giving three different runs the same name, because
#: `B0_seed0_<timestamp>` is not unique across out-roots.
#:
#: Anything matching one of these is scratch work rather than an experiment:
#: determinism checks, divergence probes, smoke tests. Imported only with
#: --include-scratch, so the tracking UI is not swamped by them.
SCRATCH = {"determinism", "divtest", "divtest_laptop", "smoke", "b0_smoke",
           "trace", "fp32check", "export", "export_probe", "_freq_smoke",
           "prelaunch", "freshcheck"}

#: Config values worth having as searchable MLflow params. The full resolved
#: config is attached as an artifact regardless, so nothing is lost by keeping
#: this list short -- these are the ones worth filtering and sorting by.
PARAM_KEYS = [
    ("arch", "width"), ("arch", "middle_blk_num"),
    ("arch", "norm_type"), ("arch", "full_res_norm_type"),
    ("arch", "clamp_bound"), ("arch", "deep_clamp_bound"),
    ("data", "patch_size"), ("data", "batch_size"), ("data", "mixed_task"),
    ("data", "sigma_range"), ("data", "clean_prob"),
    ("optim", "lr"), ("optim", "grad_clip"),
    ("schedule", "total_iters"), ("schedule", "warmup_iters"),
    ("train", "accum_steps"), ("train", "ema_decay"),
    ("train", "track_clamp_engagement"),
    ("loss", "name"),
    # Both spellings: runs before the paths fix carry `teacher` (an
    # absolute path), runs after carry `teacher_task`. Logging only the
    # new one would make older KD runs unfilterable by teacher.
    ("distill", "teacher_task"), ("distill", "teacher"),
    ("distill", "weight"),
    ("distill", "freq_weight"), ("distill", "freq_mode"),
    ("eval", "val_task"),
]


def _relative(run_dir: Path, roots: list[Path]) -> Path:
    """Path of ``run_dir`` beneath whichever root contains it."""
    for root in roots:
        try:
            return run_dir.resolve().relative_to(root.resolve())
        except ValueError:
            continue
    return Path(run_dir.name)


def _experiment_for(run_dir: Path, roots: list[Path]) -> str:
    """First path component under the runs root -- i.e. the launch out-root."""
    rel = _relative(run_dir, roots)
    return rel.parts[0] if rel.parts else "misc"


def _run_name(run_dir: Path, roots: list[Path]) -> str:
    """Unique, readable name: the path relative to the repository root.

    Relative to the RUNS root would not be unique -- `runs/` and the devon
    mirror `runs_devon/runs/` contain the same out-root names, and
    `B0_seed0_20260801_150903` already occurs three times across out-roots with
    different histories. Repo-relative disambiguates all of it.
    """
    try:
        return run_dir.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return run_dir.as_posix()


def _read(path: Path):
    try:
        if path.suffix == ".json":
            return json.loads(path.read_text(encoding="utf-8"))
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return None


def find_runs(roots: list[Path]) -> list[Path]:
    """Run directories, identified by carrying a history.json."""
    out = []
    for root in roots:
        if root.exists():
            out += [p.parent for p in root.rglob("history.json")]
    return sorted(set(out))


def import_run(run_dir: Path, roots: list[Path], dry: bool) -> str:
    cfg = _read(run_dir / "config.yaml")
    if isinstance(cfg, str):
        cfg = yaml.safe_load(cfg) or {}
    cfg = cfg or {}
    metrics = _read(run_dir / "metrics.json") or {}
    history = _read(run_dir / "history.json") or []
    commit = _read(run_dir / "git_commit.txt")
    seed = _read(run_dir / "seed.txt")

    exp = _experiment_for(run_dir, roots)
    name = _run_name(run_dir, roots)
    tag = str(run_dir.resolve())

    if dry:
        # No mlflow import on this path: --dry-run must work before the
        # dependency exists, which is exactly when it is most useful.
        return (f"  [dry] {exp:<16} {name:<44} "
                f"{len(history):>3} points  best={metrics.get('best_psnr')}")

    import mlflow

    mlflow.set_experiment(exp)
    # Idempotency: the run-directory path is the identity.
    existing = mlflow.search_runs(
        experiment_names=[exp],
        filter_string=f"tags.run_dir = '{tag}'", output_format="list")
    if existing:
        return f"  skip  {name} (already imported)"

    with mlflow.start_run(run_name=name):
        mlflow.set_tags({
            "run_dir": tag,
            "git_commit": commit or "unknown",
            "arm": metrics.get("arm", "unknown"),
            "seed": seed or "unknown",
            "diverged": str(metrics.get("diverged", "unknown")),
            "backfilled": "true",
        })
        params = {}
        for section, key in PARAM_KEYS:
            v = (cfg.get(section) or {}).get(key)
            if v is not None:
                params[f"{section}.{key}"] = str(v)
        if params:
            mlflow.log_params(params)

        # Full curves, stepped by iteration -- this is what makes runs
        # comparable in the UI rather than just a final number.
        for row in history:
            step = row.get("iteration")
            if step is None:
                continue
            for k, v in row.items():
                if k != "iteration" and isinstance(v, (int, float)):
                    mlflow.log_metric(k, float(v), step=int(step))

        final = metrics.get("final") or {}
        for k, v in {**metrics, **final}.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                mlflow.log_metric(f"final_{k}" if k in final else k, float(v))

        for fname in ("config.yaml", "metrics.json", "history.json",
                      "env.txt", "git_commit.txt", "resumes.jsonl", "train.log"):
            f = run_dir / fname
            if f.exists() and f.stat().st_size < 20 * 2 ** 20:
                mlflow.log_artifact(str(f))

    return (f"  ok    {exp:<16} {name:<44} "
            f"{len(history):>3} points  best={metrics.get('best_psnr')}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tracking-uri", default=None,
                    help="MLflow server, or file:./mlruns for a local store")
    # Most specific first: `_relative` returns the first match, and
    # runs_devon/runs must win over runs_devon or every mirrored run reports
    # its out-root as the literal string "runs".
    ap.add_argument("--roots", nargs="+", type=Path,
                    default=[REPO_ROOT / "runs_devon" / "runs",
                             REPO_ROOT / "runs"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--include-scratch", action="store_true",
                    help="also import determinism/smoke/probe runs")
    args = ap.parse_args()

    runs = find_runs(args.roots)
    if not args.include_scratch:
        kept = [r for r in runs
                if _experiment_for(r, args.roots) not in SCRATCH]
        if len(kept) != len(runs):
            print(f"skipping {len(runs) - len(kept)} scratch runs "
                  f"(determinism/smoke/probe); --include-scratch to import them)")
        runs = kept
    print(f"found {len(runs)} run directories under "
          f"{', '.join(str(r) for r in args.roots)}\n")
    if not runs:
        return 0

    if not args.dry_run:
        import mlflow
        if args.tracking_uri:
            mlflow.set_tracking_uri(args.tracking_uri)
        print(f"tracking uri: {mlflow.get_tracking_uri()}\n")

    counts: dict[str, int] = {}
    for r in runs:
        print(import_run(r, args.roots, args.dry_run))
        e = _experiment_for(r, args.roots)
        counts[e] = counts.get(e, 0) + 1

    print("\nby experiment:")
    for exp, n in sorted(counts.items()):
        print(f"  {exp:<18} {n:>3}")
    if "misc" in counts:
        print("\n  NOTE: runs landed in 'misc' -- add their prefix to "
              "EXPERIMENTS so they are grouped rather than silently pooled.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
