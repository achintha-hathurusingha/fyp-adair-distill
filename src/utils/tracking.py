"""Optional MLflow logging, wrapped so it can never take a training run down.

A tracking server is a **convenience**, not a dependency. Runs here take 6-22
hours and the run directory — resolved config, git hash, pip freeze,
``metrics.json``, ``history.json`` — remains the source of truth. If MLflow is
unreachable, misconfigured, or the server is restarted mid-run, training must
continue and the files must still be written.

So every call here is best-effort: the first failure logs a warning, sets a flag
and disables further attempts, and nothing propagates. That is deliberate
asymmetry — losing a tracking record costs a re-import via
``scripts/mlflow_backfill.py``; losing a 20-hour run costs a day of GPU time.

Enabled by ``tracking.mlflow_uri`` in the training config, or by the standard
``MLFLOW_TRACKING_URI`` environment variable. Absent, everything here is a
no-op with zero import cost — ``mlflow`` is only imported when a URI is set.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any


class RunTracker:
    """Best-effort MLflow run. Every method is safe to call unconditionally."""

    def __init__(self, cfg: dict, run_dir: Path, log) -> None:
        self.log = log
        self.run_dir = Path(run_dir)
        self._active = False
        self._mlflow = None

        uri = (cfg.get("tracking") or {}).get("mlflow_uri") \
            or os.environ.get("MLFLOW_TRACKING_URI")
        if not uri:
            return

        try:
            import mlflow

            # MLflow's REST client does not fail fast against a down or
            # unreachable server. Two knobs compound: MLFLOW_HTTP_REQUEST_TIMEOUT
            # bounds ONE request (default 120s), but MLFLOW_HTTP_REQUEST_MAX_RETRIES
            # defaults to 7 with exponential backoff (factor 2) -- so even at a
            # 5s per-request timeout, 7 retries of doubling backoff is several
            # MINUTES (observed: ~260s with only the timeout reduced, i.e. the
            # retry count was still the dominant term). "Best-effort" must mean
            # fast-fail, not eventually-fail, so both are set -- and set before
            # set_tracking_uri, which does not itself make the first request but
            # is the earliest point these are read from.
            os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "5")
            os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
            mlflow.set_tracking_uri(uri)
            # Experiment = the out-root, matching how mlflow_backfill.py groups
            # historical runs, so live and backfilled runs land together
            # instead of in two parallel hierarchies.
            experiment = (cfg.get("tracking") or {}).get("experiment") \
                or self.run_dir.parent.parent.name
            mlflow.set_experiment(experiment)
            mlflow.start_run(run_name=self.run_dir.name)
            self._mlflow = mlflow
            self._active = True
            self.log.info(f"mlflow: {uri} | experiment {experiment}")
        except Exception as exc:                              # noqa: BLE001
            self.log.warning(f"mlflow disabled ({type(exc).__name__}: {exc}); "
                             "training continues, run directory is unaffected")

    def _safe(self, fn, *a, **kw) -> None:
        if not self._active:
            return
        try:
            fn(*a, **kw)
        except Exception as exc:                              # noqa: BLE001
            self._active = False
            self.log.warning(f"mlflow logging failed ({type(exc).__name__}: "
                             f"{exc}); disabled for the rest of this run")

    def log_start(self, cfg: dict, commit: str | None, seed: int | None) -> None:
        if not self._active:
            return
        params: dict[str, str] = {}
        for section in ("arch", "data", "optim", "schedule", "train", "loss",
                        "distill", "eval"):
            for k, v in (cfg.get(section) or {}).items():
                if isinstance(v, (str, int, float, bool)) or v is None:
                    params[f"{section}.{k}"] = str(v)
        self._safe(self._mlflow.log_params, params)
        self._safe(self._mlflow.set_tags, {
            "run_dir": str(self.run_dir.resolve()),
            "git_commit": commit or "unknown",
            "seed": str(seed) if seed is not None else "unknown",
            "backfilled": "false",
        })

    def log_metrics(self, row: dict[str, Any], step: int) -> None:
        """One validation row. Non-numeric and None values are skipped."""
        if not self._active:
            return
        clean = {k: float(v) for k, v in row.items()
                 if k != "iteration" and isinstance(v, (int, float))
                 and not isinstance(v, bool)}
        if clean:
            self._safe(self._mlflow.log_metrics, clean, step=step)

    def finish(self, metrics: dict | None = None) -> None:
        """Attach the run directory's own records, then close."""
        if not self._active:
            return
        if metrics:
            self._safe(self._mlflow.set_tag, "diverged",
                       str(metrics.get("diverged", "unknown")))
        for name in ("config.yaml", "metrics.json", "history.json",
                     "env.txt", "git_commit.txt", "train.log"):
            f = self.run_dir / name
            if f.exists() and f.stat().st_size < 20 * 2 ** 20:
                self._safe(self._mlflow.log_artifact, str(f))
        self._safe(self._mlflow.end_run)
        self._active = False
