#!/usr/bin/env bash
# S3.3 sequential long run. One GPU, three arms, ~12h each.
#
# _v2: the FIRST ATTEMPT WAS INVALID. Both blocks zero-initialised the 1x1 fuse,
# which left the depthwise kernel with ZERO gradient (dL/d(conv) is proportional
# to fuse.weight = 0), so early training amplified a frozen random Kaiming kernel
# into the decoder -- a random blur. B0V3-KD-K11 was 1.335 dB behind its control
# at 15k with -3.026 on dehaze and -0.037 on denoise. Fixed by delta-initialising
# the kernel so conv(x) = x at init; see runs/s33_b0v3_kd_k11/WHY_THIS_FAILED.md.
# Output dirs are suffixed _v2 so the failed run stays on disk as evidence.
#
#   B0V3-KD-K11      plain 11x11 depthwise   <- receptive-field CONTROL
#   B0V3-KD-ORI      oriented block          <- treatment
#   B0V3-KD-ORI-MID  oriented + middle       <- placement
#
# vs the already-trained B0V3-KD-FEAT (no block, 90k) as control A.
# K11 runs FIRST: K11-vs-ORI is the decisive pair, so if anything derails the
# sequence the two arms that isolate orientation are the ones already done.
set -u
cd /home/minura/fyp-adair-distill || exit 1
source ~/miniforge3/etc/profile.d/conda.sh
conda activate adair-distill

STAMP=$(date +%Y%m%d_%H%M%S)
LOG=/tmp/s33_sequence_${STAMP}.log
echo "S3.3 sequence started $(date)" | tee "$LOG"

for ARM in B0V3-KD-K11 B0V3-KD-ORI B0V3-KD-ORI-MID; do
  OUT="runs/s33_$(echo "$ARM" | tr 'A-Z-' 'a-z_')_v2"
  echo "" | tee -a "$LOG"
  echo "=== $ARM -> $OUT  started $(date) ===" | tee -a "$LOG"

  taskset -c 0-7,12-31 python -m src.train.train \
      --arm "$ARM" --out-root "$OUT" >> "$LOG" 2>&1
  RC=$?

  if [ $RC -ne 0 ]; then
    echo "=== $ARM FAILED rc=$RC at $(date) -- continuing to next arm ===" | tee -a "$LOG"
  else
    echo "=== $ARM done $(date) ===" | tee -a "$LOG"
  fi
  # surface the last validation line for this arm
  find "$OUT" -name train.log -newermt '-13 hours' -exec tail -1 {} \; 2>/dev/null | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "S3.3 SEQUENCE COMPLETE $(date)" | tee -a "$LOG"
