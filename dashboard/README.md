# kd_freq live dashboard

Single-page live view of the M-DEHAZE-KD-FREQ 3-seed run (see
) against the completed GT-only and response-KD
baselines. Stdlib-only Python — no pip dependencies, runs identically inside
Docker or directly on the host.

## Run directly (no Docker)

    RUNS_ROOT=~/fyp-adair-distill/runs LAUNCH_LOG=/tmp/kd_freq_3seed.log       python3 dashboard/server.py
    # -> http://localhost:8080

## Run with Docker

    cd dashboard && docker compose up -d --build
    # -> http://localhost:8080

Requires Docker installed (`sudo apt install docker.io docker-compose-plugin`)
and the invoking user in the `docker` group — neither was available
non-interactively when this was built (sudo needs a password on devon), so
the live instance currently runs as a plain background process, not a
container. Swap to `docker compose up -d --build` once Docker is installed;
no code changes needed.

To expose outside an SSH tunnel: `sudo ufw allow 8080/tcp` on devon (also
blocked on the same sudo password today).
