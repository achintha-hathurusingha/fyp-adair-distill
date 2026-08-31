# Training board

Live view of the training runs. Replaces the kd_feature_multitask dashboard,
rebuilt for the S3.3 block ablation.

## Run it

    python dashboard/local_server.py     ->  http://127.0.0.1:8092

Deploy the remote reader once per training host first (already done on devon):

    scp -i <key> dashboard/remote_status.py <user>@<host>:/tmp/remote_status.py

## Architecture (unchanged, it was the right shape)

- `remote_status.py` — lives at `/tmp/remote_status.py` on each training host.
  Stdlib only. Reads each run's `history.json` and tails its `train.log`
  straight off disk; it never touches the training process.
- `local_server.py` — runs on the Windows machine, so it can SSH into devon and
  qbits without either host trusting the other. Polls every 6 s, serves the
  merged JSON at `/api/status` and the page at `/`.
- `index.html` — vanilla JS, polls `/api/status` every 6 s. No build step.

## What it shows

Per host: GPU utilisation, VRAM, temperature — including idle capacity on
qbits, which is how you tell whether a run can be placed there.

Per arm: progress against 90k, combined and per-task PSNR, loss / grad-norm /
clip-rate / non-finite-skip / VRAM diagnostics, a PSNR-over-iterations chart
built from the run's own history, and a live log tail.

## The thing to not forget

**Two evaluation regimes are on this board and every card says which it is.**
The finished arms (B0V3-KD-FEAT, B0V3, B0V2-KD-FEAT) were validated on
`test/derain/demo` + `test/dehaze/demo`, which had been carved out of the
TRAINING corpora — their curves run ~1.9 dB high. The S3.3 arms validate on
BSD68 / Rain100L-100 / SOTS-clean-417 and are leak-free.

Curves are comparable **within a regime only**. The finished arms' honest
numbers are the re-scored ones in `reports/clean_eval_rescore.json`, not the
values their own history shows.

## Tracked arms

Edit `HOSTS` in `local_server.py` when an arm moves host or a new one starts —
a stale arm list is the failure mode that bit every previous version of this
dashboard.

| arm | role | eval |
|---|---|---|
| B0V3-KD-K11 | plain 11×11 depthwise — receptive-field control | leak-free |
| B0V3-KD-ORI | reparameterizable oriented block | leak-free |
| B0V3-KD-ORI-MID | oriented block + middle placement | leak-free |
| B0V3-KD-FEAT | StudentV3 + KD — best arm, no-block control | leaked |
| B0V3 | StudentV3, GT-only — isolates KD | leaked |
| B0V2-KD-FEAT | NAFNet + KD — isolates architecture | leaked |
