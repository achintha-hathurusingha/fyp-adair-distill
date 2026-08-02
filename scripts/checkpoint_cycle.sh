#!/usr/bin/env bash
# Periodic stop -> verify -> resume for a running B0 seed.
#
#   ./scripts/checkpoint_cycle.sh <seed> [out-root]
#
# WHY. This is not belt-and-braces: it is a standing test of `--resume` on the
# actual production run, and it bounds how long a silent problem can go
# unnoticed. Today's NVML driver drift was invisible to the training logs for
# hours -- the runs kept going while nothing new could start. A periodic verify
# would have caught it at the next cycle instead of at the next launch attempt.
#
# The stop is GRACEFUL (SIGTERM, wait, escalate only if needed) so the current
# step and any checkpoint write complete. A SIGKILL mid-write is exactly how a
# checkpoint gets corrupted, which would turn a safety measure into the thing it
# is meant to protect against.
#
# Refuses to resume if anything looks wrong. Reporting a problem is always
# better than resuming blind into it.
set -uo pipefail

SEED="${1:?usage: checkpoint_cycle.sh <seed> [out-root]}"
OUT_ROOT="${2:-runs/b0_final}"
cd ~/fyp-adair-distill || exit 1

PAT="--seed ${SEED} --out-root ${OUT_ROOT}"
RUN_DIR=$(ls -td "${OUT_ROOT}"/B0/*seed${SEED}_*/ 2>/dev/null | head -1)
say() { echo "[cycle seed${SEED}] $*"; }

if [ -z "$RUN_DIR" ]; then say "FAIL: no run dir under ${OUT_ROOT}"; exit 1; fi
CKPT="${RUN_DIR}last.pth"

# --- 1. graceful stop -------------------------------------------------------
PIDS=$(pgrep -f "$PAT" || true)
if [ -z "$PIDS" ]; then say "FAIL: no running process matching '$PAT'"; exit 1; fi
BEFORE=$(stat -c %Y "$CKPT" 2>/dev/null || echo 0)
say "stopping $(echo "$PIDS" | wc -w) processes (SIGTERM)"
kill -TERM $PIDS 2>/dev/null
for _ in $(seq 1 60); do
  pgrep -f "$PAT" >/dev/null || break
  sleep 1
done
if pgrep -f "$PAT" >/dev/null; then
  say "did not exit in 60s; escalating to SIGKILL"
  kill -9 $(pgrep -f "$PAT") 2>/dev/null
  sleep 5
fi
say "stopped"

# --- 2. verify --------------------------------------------------------------
if ! nvidia-smi --query-gpu=driver_version --format=csv,noheader >/dev/null 2>&1; then
  say "ABORT: nvidia-smi is broken (driver/library mismatch?). NOT resuming."
  nvidia-smi 2>&1 | head -3
  exit 2
fi
say "driver ok: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"

LEFTOVER=$(pgrep -cf "$PAT" || true)
if [ "${LEFTOVER:-0}" -ne 0 ]; then
  say "ABORT: $LEFTOVER leftover processes still alive. NOT resuming."
  exit 2
fi

AFTER=$(stat -c %Y "$CKPT" 2>/dev/null || echo 0)
[ "$AFTER" -gt "$BEFORE" ] && say "checkpoint was rewritten during shutdown (good)"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate adair-distill
ITER=$(python - "$CKPT" <<'PY'
import sys, torch
try:
    c = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
    need = {"model", "optimizer", "ema", "iteration",
            "rng_python", "rng_numpy", "rng_torch", "rng_cuda"}
    missing = need - set(c)
    if missing:
        print(f"BAD missing:{sorted(missing)}")
    else:
        print(c["iteration"])
except Exception as exc:                       # noqa: BLE001
    print(f"BAD {type(exc).__name__}: {exc}")
PY
)
case "$ITER" in
  BAD*) say "ABORT: checkpoint failed to verify -- $ITER. NOT resuming."; exit 2 ;;
esac
say "checkpoint verified: iteration $ITER, all RNG streams present"

# --- 3. resume --------------------------------------------------------------
say "resuming"
setsid nohup ./run_b0_devon.sh --arm B0 --seed "$SEED" --out-root "$OUT_ROOT" \
  --num-workers 8 --resume "$CKPT" \
  --resume-reason "periodic checkpoint-verify cycle at ${ITER}" \
  >> "logs/b0_seed${SEED}.log" 2>&1 &
disown
sleep 30
NOW=$(pgrep -cf "$PAT" || echo 0)
if [ "$NOW" -eq 0 ]; then say "FAIL: did not come back up"; exit 1; fi
say "resumed from $ITER ($NOW processes)"
