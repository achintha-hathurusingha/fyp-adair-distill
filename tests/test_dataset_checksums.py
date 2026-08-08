"""The dataset checksum registry: catches corruption, not just presence.

A directory-level hash tells you *that* something changed, never *what* --
these tests exist to prove the per-file design actually names the exact file,
and that generate/verify agree with each other rather than each silently doing
something slightly different.
"""
from __future__ import annotations

import importlib
import json

import numpy as np
import pytest
from PIL import Image

dc = importlib.import_module("scripts.dataset_checksums")


def _write_img(path, seed, size=(8, 8)):
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (*size, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


@pytest.fixture
def fake_dataset(tmp_path, monkeypatch):
    """One dataset, 5 images, wired through paths.yaml's real resolution path."""
    root = tmp_path / "data" / "denoise" / "toy"
    for i in range(5):
        _write_img(root / f"img{i:02d}.png", seed=i)

    monkeypatch.setattr(dc, "_dataset_roots", lambda: {"toy": root})
    monkeypatch.setattr(dc, "_OUT_DIR", tmp_path / "checksums")
    return root


def test_generate_writes_one_entry_per_file(fake_dataset) -> None:
    assert dc.generate(None) == 0
    manifest = json.loads((dc._OUT_DIR / "toy.json").read_text(encoding="utf-8"))
    assert manifest["n_files"] == 5
    assert len(manifest["files"]) == 5
    assert all("sha256" in v and "size" in v for v in manifest["files"].values())


def test_verify_passes_against_an_unmodified_dataset(fake_dataset) -> None:
    dc.generate(None)
    assert dc.verify(None) == 0


def test_verify_catches_a_corrupted_file(fake_dataset, capsys) -> None:
    """The exact case a directory-level hash cannot distinguish from bit rot
    on an unrelated file: one image silently replaced with different content."""
    dc.generate(None)
    _write_img(fake_dataset / "img02.png", seed=999)   # same name, new content
    rc = dc.verify(None)
    out = capsys.readouterr().out
    assert rc == 1
    assert "FAIL" in out
    assert "img02.png" in out, "must name the exact corrupted file"


def test_verify_catches_a_missing_file(fake_dataset, capsys) -> None:
    dc.generate(None)
    (fake_dataset / "img03.png").unlink()
    rc = dc.verify(None)
    out = capsys.readouterr().out
    assert rc == 1
    assert "1 missing" in out
    assert "img03.png" in out


def test_verify_reports_untracked_extras_without_failing(fake_dataset, capsys) -> None:
    """A NEW file appearing is not corruption -- it just was not in the manifest
    when generated. Reported for visibility, not treated as a failure."""
    dc.generate(None)
    _write_img(fake_dataset / "img_new.png", seed=42)
    rc = dc.verify(None)
    out = capsys.readouterr().out
    assert rc == 0, "an added file alone must not fail verify"
    assert "1 untracked-extra" in out


def test_verify_without_a_manifest_is_skipped_not_a_false_pass(fake_dataset, capsys) -> None:
    rc = dc.verify(None)
    out = capsys.readouterr().out
    assert "SKIP" in out
    assert rc == 0, "nothing to verify is not itself a failure"


def test_empty_or_absent_dataset_is_never_silently_skipped_at_generate(
        tmp_path, monkeypatch, capsys) -> None:
    """The RESIDE-specific failure mode (empty/absent dir) must be caught for
    every dataset here, not only in reside_manifest.py."""
    empty = tmp_path / "empty_dataset"
    monkeypatch.setattr(dc, "_dataset_roots", lambda: {"empty": empty})
    monkeypatch.setattr(dc, "_OUT_DIR", tmp_path / "checksums")
    dc.generate(None)
    out = capsys.readouterr().out
    assert "SKIP" in out and "empty" in out
    assert not (dc._OUT_DIR / "empty.json").exists(), \
        "must not write a manifest claiming zero files is a verified state"


def test_unknown_dataset_name_is_rejected(fake_dataset) -> None:
    with pytest.raises(SystemExit, match="unknown dataset"):
        dc.generate("does-not-exist")
    with pytest.raises(SystemExit, match="unknown dataset"):
        dc.verify("does-not-exist")


def test_size_change_alone_is_caught_even_with_no_hash_mismatch(fake_dataset) -> None:
    """A truncated file (partial download/disk error) may not always land on a
    hash collision path in a naive check -- assert size is compared too, not
    only the hash, so a truncated-but-hash-uncomputed edge case is impossible."""
    dc.generate(None)
    manifest_path = dc._OUT_DIR / "toy.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Corrupt the recorded size only, simulating a manifest/reality mismatch.
    first = next(iter(manifest["files"]))
    manifest["files"][first]["size"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert dc.verify(None) == 1
