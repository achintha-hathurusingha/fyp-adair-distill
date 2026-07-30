"""Write inference-only copies of the AdaIR checkpoints.

The released files are full Lightning training checkpoints (~346 MB), of which
roughly two thirds is Adam moment state we will never use. Stripping to weights
alone gives ~115 MB files that load faster, cache smaller, and — importantly —
**cannot be used to accidentally resume training from a teacher checkpoint**.

Keys are also un-prefixed here (``net.`` removed) so the output loads directly
into a bare ``AdaIR`` with ``strict=True``.

    python -m scripts.strip_checkpoints --ckpt-dir data/ckpt --out-dir data/ckpt/inference
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

from scripts.audit_checkpoints import _build_reference_model, _strip, sha256_file


def strip_checkpoint(src: Path, dst: Path, reference: torch.nn.Module) -> dict:
    """Write an inference-only checkpoint and verify it loads strictly.

    Returns a summary dict. Raises if the stripped file does not load with
    ``strict=True`` — a silent partial load here would poison every downstream
    number.
    """
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    sd, prefix = _strip(sd)
    sd = {k: v for k, v in sd.items() if torch.is_tensor(v)}

    dst.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": sd,
        "source_file": src.name,
        "source_sha256": sha256_file(src),
        "stripped_prefix": prefix,
        "epoch": ckpt.get("epoch"),
        "global_step": ckpt.get("global_step"),
        "note": "inference-only: optimizer and scheduler state removed",
    }
    torch.save(payload, dst)

    # Verify strictly -- this must not be lenient.
    reference.load_state_dict(sd, strict=True)

    return {
        "name": dst.name,
        "src_mb": src.stat().st_size / 1e6,
        "dst_mb": dst.stat().st_size / 1e6,
        "n_params": sum(v.numel() for v in sd.values()),
        "sha256": sha256_file(dst),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Strip optimizer state from checkpoints.")
    ap.add_argument("--ckpt-dir", default="data/ckpt")
    ap.add_argument("--out-dir", default="data/ckpt/inference")
    args = ap.parse_args()

    src_dir, out_dir = Path(args.ckpt_dir), Path(args.out_dir)
    files = sorted(p for p in src_dir.glob("*.ckpt") if p.is_file())
    if not files:
        raise FileNotFoundError(f"no .ckpt files in {src_dir}")

    reference = _build_reference_model()
    ref_params = sum(p.numel() for p in reference.parameters())

    total_src = total_dst = 0
    for src in files:
        dst = out_dir / f"{src.stem}.pth"
        info = strip_checkpoint(src, dst, reference)
        total_src += info["src_mb"]
        total_dst += info["dst_mb"]
        status = "ok" if info["n_params"] == ref_params else "PARAM MISMATCH"
        print(f"[strip] {src.name:28s} {info['src_mb']:6.0f} MB -> "
              f"{info['dst_mb']:6.0f} MB  params={info['n_params']:,}  {status}")
        if info["n_params"] != ref_params:
            raise SystemExit(
                f"{src.name}: {info['n_params']:,} params != reference {ref_params:,}")

    print(f"[strip] total {total_src:.0f} MB -> {total_dst:.0f} MB "
          f"({100 * (1 - total_dst / total_src):.0f}% smaller) -> {out_dir}")


if __name__ == "__main__":
    main()
