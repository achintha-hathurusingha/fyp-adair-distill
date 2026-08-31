"""Dump every run's history plus the EVALUATION REGIME it was scored under.

The regime matters more than the curve. Three incompatible regimes exist in
runs/ and putting them on one axis would be the same class of error as the
train/test leak:

  denoise-only  : the 300k B0/B0V2 runs validated on BSD68 alone, so their
                  "psnr" is a denoise number, not a 3-task mean
  multitask-leaked : 90k arms validated on test/derain/demo + test/dehaze/demo,
                  which were carved out of the training corpora
  single-task   : the M-DEHAZE / M-DERAIN demo runs, one task each
  corrected     : re-scored on BSD68 / Rain100L-100 / SOTS-clean-417

Only curves within the same regime are comparable to each other.
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import yaml


def regime(run_dir: Path, hist: list) -> tuple[str, str]:
    cfg_p = run_dir / "config.yaml"
    val = ""
    if cfg_p.exists():
        try:
            cfg = yaml.safe_load(cfg_p.read_text())
            ev = (cfg or {}).get("eval", {}) or {}
            vt = ev.get("val_tasks")
            if vt:
                val = ",".join(f"{k}:{v}" for k, v in sorted(vt.items()))
            elif ev.get("val_root"):
                val = str(ev["val_root"])
        except Exception:
            pass
    has_pt = any(e.get("psnr_derain") is not None for e in hist)
    n_tasks = 0
    if has_pt:
        last = next(e for e in reversed(hist) if e.get("psnr_derain") is not None)
        n_tasks = sum(last.get(f"psnr_{t}") is not None
                      for t in ("denoise", "derain", "dehaze"))
    if not has_pt:
        # single-task runs carry only a val_root; the task is in its path
        low = val.lower()
        for t in ("dehaze", "derain"):
            if t in low:
                return f"single-task-{t}", val
        return "denoise-only", val
    if n_tasks == 3:
        return ("multitask-leaked" if "demo" in val or not val
                else "multitask-clean"), val
    return "single-task", val


def main():
    out = []
    for f in sorted(glob.glob("runs/**/history.json", recursive=True)):
        try:
            h = json.load(open(f))
        except Exception:
            continue
        if not h:
            continue
        d = Path(os.path.dirname(f))
        reg, val = regime(d, h)
        pts = [{"it": e.get("iteration"), "p": e.get("psnr"),
                "dn": e.get("psnr_denoise"), "rn": e.get("psnr_derain"),
                "hz": e.get("psnr_dehaze")}
               for e in h if e.get("iteration") is not None]
        name = str(d).replace("runs/", "")
        arm = name.split("/")[1] if "/" in name else name
        out.append({"run": name, "arm": arm, "regime": reg, "val": val,
                    "n": len(pts), "last_it": pts[-1]["it"] if pts else None,
                    "pts": pts})
    out.sort(key=lambda r: -(r["last_it"] or 0))
    Path("reports").mkdir(exist_ok=True)
    Path("reports/run_histories.json").write_text(json.dumps(out))
    print(f"{len(out)} runs -> reports/run_histories.json")
    for r in out:
        print(f"  {r['regime']:<18}{r['arm']:<22}{r['last_it']:>7}  n={r['n']:<3}")


if __name__ == "__main__":
    main()
