"""Every key in a reviewed training config must actually be READ by the code.

WHY THIS EXISTS. Twice now a config key has looked authoritative while being
silently ignored:

  * `warmup_iters` -- 2000 in code, 5000 in the reviewed YAML. Caught by making
    the YAML authoritative (commit 82e1ba0).
  * `mixed_task: true` -- in `b0_baseline.yaml` since the first scaffold commit,
    documented as "balance degradation types WITHIN each batch", and read by no
    code at all. B0 trained on denoise only for its entire run (findings F11).

The first fix cannot catch the second. It verified config values MATCH between
YAML and code; a key the code never reads matches trivially, because nothing
contradicts it. The check has to be inverted: assert every key is CONSUMED.

A config key is a claim about behaviour. This test makes the claim falsifiable.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from src.utils.config import REPO_ROOT

#: Configs that drive a training run, and so must not carry dead keys.
TRAINING_CONFIGS = [
    "configs/train/b0_baseline.yaml",
    "configs/train/b0_qa_control.yaml",
    "configs/train/b0_fixc.yaml",
    "configs/train/b0v2_multitask.yaml",
]

#: Source consuming those configs. A key must appear in at least one of these.
CONSUMERS = [
    "src/train/train.py",
    "src/train/trainer.py",
    "src/data/build.py",
    "src/models/nafnet.py",
    "src/models/norms.py",
    "src/losses/reconstruction.py",
]

#: Documentation-only keys, each with the reason it is not read by code.
#: Anything added here is a deliberate, reviewed exception -- not a way to
#: silence the test.
DOCUMENTED_NON_CODE_KEYS = {
    "experiment": "run label, recorded in the run directory",
    "model": "path to the locked architecture file, read by the reader not the code",
    "arch": "mirror of the locked geometry; the drift guard compares it explicitly",
    "eval": "declares which harness/test sets a human should use",
    "harness": "documentation of the locked evaluation entry point",
    "test_sets": "documentation of the evaluation sets",
    "name": "human-readable scheduler/optimiser label",
    "seeds": "the seed list is passed on the CLI, one process per seed",
}

#: Keys KNOWN to be dead, with the finding that records it. Present so the test
#: passes today while making the debt explicit and greppable. Removing a key
#: from this set must make the test fail until the key is genuinely wired up.
#:
#: EMPTY, and it should stay that way. `mixed_task` was the only entry; it was
#: removed when `build_multitask_loader` landed and the tripwire below fired
#: exactly as intended.
KNOWN_DEAD: dict[str, str] = {}


def _flatten(node, out: set[str]) -> set[str]:
    if isinstance(node, dict):
        for k, v in node.items():
            out.add(str(k))
            _flatten(v, out)
    elif isinstance(node, list):
        for v in node:
            _flatten(v, out)
    return out


def _consumer_source() -> str:
    return "\n".join((REPO_ROOT / p).read_text(encoding="utf-8")
                     for p in CONSUMERS if (REPO_ROOT / p).exists())


@pytest.mark.parametrize("cfg_path", TRAINING_CONFIGS)
def test_no_dead_config_keys(cfg_path: str) -> None:
    """Fail if a config key is never referenced by the consuming code."""
    p = REPO_ROOT / cfg_path
    if not p.exists():
        pytest.skip(f"{cfg_path} not present")
    keys = _flatten(yaml.safe_load(p.read_text(encoding="utf-8")), set())
    src = _consumer_source()

    dead = []
    for k in sorted(keys):
        if k in DOCUMENTED_NON_CODE_KEYS or k in KNOWN_DEAD:
            continue
        # A key counts as read if it appears as a string literal or attribute.
        if not re.search(rf"""["']{re.escape(k)}["']|\b{re.escape(k)}\b""", src):
            dead.append(k)

    assert not dead, (
        f"{cfg_path} carries key(s) no code reads: {dead}\n"
        "A config key is a claim about behaviour. Either wire it up, or add it "
        "to DOCUMENTED_NON_CODE_KEYS / KNOWN_DEAD with a reason.\n"
        "This is how B0 trained on denoise only for a full run (F11).")


def test_mixed_task_actually_selects_the_loader() -> None:
    """`mixed_task` must decide which loader is built, not merely be mentioned.

    The weaker check -- that the string appears in the source -- would pass on a
    comment, and a comment is what F11 was. This asserts the branch exists and
    that the true case reaches the multi-task loader.
    """
    src = (REPO_ROOT / "src/train/train.py").read_text(encoding="utf-8")
    branch = re.search(r"if cfg\[.data.\]\[.mixed_task.\]:(.*?)\n    else:",
                       src, re.S)
    assert branch, "mixed_task no longer selects a loader in train.py"
    assert "build_multitask_loader" in branch.group(1), (
        "the mixed_task=true branch does not build the multi-task loader")
    assert "mixed_task" not in KNOWN_DEAD, "the key is wired up; stop excusing it"


def test_known_dead_entries_carry_a_reason() -> None:
    for key, reason in {**KNOWN_DEAD, **DOCUMENTED_NON_CODE_KEYS}.items():
        assert reason and len(reason) > 10, f"{key} needs a real justification"
