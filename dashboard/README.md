# kd_feature_multitask dashboard

Replaces the previous `dashboard/` entirely (both the kd_freq-era single-page
one and the `unified/` one from the ECA/kd_feat/GroupNorm phase). Rebuilt for
the current phase: **B0V2-KD-FEAT** (control) vs **B0V2-KD-FEAT-COND**
(treatment) — does degradation-conditioning prevent multi-task interference?

## What it shows

- **Control vs Treatment** panel: per-task PSNR delta (denoise/derain/dehaze
  + combined) between the two arms, once both have at least one checkpoint.
  This is the actual result the experiment is asking for — a single combined
  PSNR number can hide interference concentrated on one task.
- One card per arm: progress, PSNR (combined + per-task), loss/grad-norm/clip
  diagnostics, clamp engagement (the F9/F10 divergence-risk telemetry), and a
  live log tail.

## Architecture

- `remote_status.py` — deployed to `/tmp/remote_status.py` on each training
  host. Stdlib only. Reads `history.json` (Trainer's own structured
  per-checkpoint dump) and tails `train.log` straight from each run
  directory — never touches the training process itself.
- `local_server.py` — runs on THIS machine (not on either training host), so
  it can SSH into both devon and qbits without either needing to trust the
  other. Polls every 6s, serves the merged JSON at `/api/status` plus the
  static page.
- `index.html` — single page, vanilla JS, polls `/api/status` every 6s.

## Run

    python local_server.py
    # -> http://127.0.0.1:8092

Deploy `remote_status.py` to each host first:

    scp -i <key> remote_status.py <user>@<host>:/tmp/remote_status.py

`HOSTS` in `local_server.py` configures which arms (as `"out_root/ARM_NAME"`)
to track on which host. Update it when an arm moves hosts or a new arm is
added — this bit the previous dashboard (`unified/`) repeatedly.
