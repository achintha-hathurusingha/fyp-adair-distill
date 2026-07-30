"""Logging setup: console + per-run file handler."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str = "adair", *, run_dir: str | Path | None = None,
               level: int = logging.INFO) -> logging.Logger:
    """Return a logger writing to stdout and, if given, ``run_dir/train.log``.

    Idempotent: repeated calls with the same name do not duplicate handlers.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(sh)

    if run_dir is not None:
        log_path = Path(run_dir) / "train.log"
        already = any(
            isinstance(h, logging.FileHandler)
            and Path(getattr(h, "baseFilename", "")) == log_path.resolve()
            for h in logger.handlers
        )
        if not already:
            fh = logging.FileHandler(log_path, encoding="utf-8")
            fh.setFormatter(logging.Formatter(_FORMAT))
            logger.addHandler(fh)

    return logger
