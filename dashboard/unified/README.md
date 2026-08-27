# Unified dashboard (devon + qbits)

Single view across both training hosts. Unlike `dashboard/server.py` (which
runs *on* devon and only sees devon's own runs/), this one runs **locally**
(your machine, not either GPU box) and SSHes into both hosts each poll —
neither host needs to trust the other.

## How it works

- `remote_status.py` — stdlib-only, deployed to `/tmp/remote_status.py` on
  each host. Reads that host's own `runs/` + log file, prints one JSON line.
- `local_server.py` — runs locally. Each poll: SSHes into devon and qbits,
  runs `remote_status.py` on each, merges both results, serves the combined
  JSON at `/api/status` plus `unified.html` at `/`.

## Run it

    python local_server.py
    # -> http://127.0.0.1:8091

No tunnel needed — this listens on localhost directly, since it's already
running on your machine.

## Arms tracked (edit `HOSTS` in local_server.py to change)

| arm | host |
|---|---|
| M-DEHAZE (GT only) | devon |
| M-DEHAZE-KD (response KD) | devon |
| M-DEHAZE-KD-FREQ (+ frequency KD) | devon |
| M-DEHAZE-KD-FEAT (+ feature KD, latent_pre) | qbits |
| M-DEHAZE-ECA (SCA -> ECA) | qbits |
| M-DEHAZE-GROUPNORM (LayerNorm2d -> GroupNorm) | qbits |

## Known gotcha (hit and fixed once already)

`remote_status.py` picks the most-recently-modified run directory matching
`<arm>_seed<N>_*`. Stale directories from smoke tests or killed launch
attempts can have a newer mtime than a real, actively-training run whose
directory was created earlier (or whose mtime was preserved by a file
transfer) — this silently makes the dashboard show the wrong run's
progress. If a resumed/migrated run's progress looks wrong, check for
duplicate `<arm>_seed<N>_*` directories on that host and delete the stale
ones.
