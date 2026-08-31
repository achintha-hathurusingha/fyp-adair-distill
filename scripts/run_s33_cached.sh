#!/usr/bin/env bash
# S3.3 cached pair. BOTH arms cached so the cache's offset vs a live teacher
# sits on both sides and cancels; comparing a cached arm to the LIVE
# B0V3-KD-FEAT would charge that offset to the block.
# Control first, so if the sequence derails the reference exists.
set -u
cd /home/minura/fyp-adair-distill || exit 1
source ~/miniforge3/etc/profile.d/conda.sh
conda activate adair-distill
LOG=/tmp/s33_cached_$(date +%Y%m%d_%H%M%S).log
echo "S3.3 cached pair started $(date)" | tee "$LOG"
for ARM in S33-CTRL-CACHED S33-K11-CACHED; do
  OUT="runs/$(echo "$ARM" | tr 'A-Z-' 'a-z_')"
  echo "" | tee -a "$LOG"
  echo "=== $ARM -> $OUT  started $(date) ===" | tee -a "$LOG"
  taskset -c 0-7,12-31 python -m src.train.train --arm "$ARM" --out-root "$OUT" >> "$LOG" 2>&1
  RC=$?
  [ $RC -ne 0 ] && echo "=== $ARM FAILED rc=$RC $(date) ===" | tee -a "$LOG" \
                || echo "=== $ARM done $(date) ===" | tee -a "$LOG"
done
echo "S3.3 CACHED PAIR COMPLETE $(date)" | tee -a "$LOG"
