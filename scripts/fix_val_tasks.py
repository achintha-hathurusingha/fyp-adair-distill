"""Point every multi-task config at leak-free evaluation sets.

WHY
---
`test/derain/demo` (40) and `test/dehaze/demo` (150) were carved out of the
TRAINING corpora by make_derain_split.py / make_dehaze_split.py. That is only
safe if training then excludes them via the `list:` key. The single-task m_*
configs do that. The multi-task configs do NOT -- they pass bare paths, so
build.py's `_listed_images` falls through to the whole directory. Verified:
all 40 derain test files are byte-identical to training files, and all 150
dehaze test stems are among the 2,061 training clear stems.

THE FIX IS EVALUATION-ONLY. Our TRAINING data already matches the published
all-in-one protocol exactly (AdaIR, ICLR 2025, sec. 4):
    denoise  BSD400 + WED   = 5,144   -> we have 5,144
    derain   Rain100L train =   200   -> we have 200
    dehaze   RESIDE OTS     = 72,135  -> we have 72,135
so nothing about training changes, and no training data is sacrificed. In
particular do NOT add `list: reports/derain_train_list.txt` here: that 160/40
split existed only because the single-task demo runs had no real test set. The
real Rain100L test split (100 pairs, verified 0 content overlap) makes it
unnecessary, and using it would deviate from the protocol AND discard 40
training pairs for nothing.

DEHAZE: WHICH SOTS
------------------
RESIDE's own published split is not disjoint: 75 of SOTS-outdoor's 492 clear
scenes also occur in OTS training (make_sots_clean.py verified this by content
correlation at 1.0000, not by filename). So there are two defensible sets and
they answer different questions:
  * test/dehaze/sots_clean          417 scenes, genuinely unseen -> honest
  * dehaze/RESIDE/SOTS/outdoor      500 images, what the literature reports
                                     -> comparable, but 75 scenes are memorised
`sots_clean` is wired in as the validation set because model selection must not
run on memorised scenes. Full SOTS is reported alongside at the end, with the
caveat, for comparability with AdaIR's published 31.06 dB.
"""
from __future__ import annotations

import sys
from pathlib import Path

OLD = """  val_tasks:
    denoise: test/denoise/bsd68
    derain:  test/derain/demo
    dehaze:  test/dehaze/demo"""

NEW = """  val_tasks:
    # Leak-free sets. The former test/{derain,dehaze}/demo sets were carved out
    # of the TRAINING corpora and multi-task runs never excluded them -- see
    # scripts/fix_val_tasks.py. Training data is unchanged and still matches the
    # published protocol (BSD400+WED / Rain100L-200 / OTS-72135).
    denoise: test/denoise/bsd68      # 68 imgs; verified disjoint from BSD400+WED
    derain:  test/derain/rain100L    # Rain100L TEST, 100 pairs; 0/200 content overlap
    dehaze:  test/dehaze/sots_clean  # SOTS-outdoor minus the 75 scenes RESIDE leaks into OTS"""

TARGETS = [
    "b0v2_kd_denoise_only.yaml", "b0v2_kd_feat_cached.yaml",
    "b0v2_kd_feat_cond_decfilm.yaml", "b0v2_kd_feat_cond.yaml",
    "b0v2_kd_feat.yaml", "b0v2_multitask.yaml", "b0v3_kd_feat.yaml",
    "b0v3m.yaml", "b0v3.yaml",
]


def main() -> int:
    root = Path("configs/train")
    changed = 0
    for name in TARGETS:
        p = root / name
        if not p.exists():
            print(f"  SKIP (missing) {name}")
            continue
        src = p.read_text(encoding="utf-8")
        if "test/derain/rain100L" in src:
            print(f"  already fixed  {name}")
            continue
        n = src.count(OLD)
        if n != 1:
            print(f"  !! {name}: val_tasks block matched {n} times, NOT patched")
            continue
        src = src.replace(OLD, NEW)
        src = src.replace("test_sets: [bsd68, derain_demo, dehaze_demo]",
                          "test_sets: [bsd68, rain100L, sots_clean]")
        p.write_text(src, encoding="utf-8")
        print(f"  patched        {name}")
        changed += 1
    print(f"\n{changed} config(s) updated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
