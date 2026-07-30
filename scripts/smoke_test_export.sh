#!/usr/bin/env bash
# Gate G1: export NAFNet-w32 (with and without the channel gate) to ONNX +
# INT8, benchmark, and write reports/export_smoke_test.md.
set -euo pipefail

cd "$(dirname "$0")/.."

OUT_DIR="${1:-runs/g1}"
SEED="${SEED:-0}"

python -m src.export.smoke_test \
    --model-config configs/model/nafnet_w32.yaml \
    --export-config configs/export/qnn_int8.yaml \
    --out-dir "${OUT_DIR}" \
    --report reports/export_smoke_test.md \
    --seed "${SEED}"
