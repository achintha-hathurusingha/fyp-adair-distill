# fyp-adair-distill

Edge-deployable all-in-one image restoration via **cross-architecture knowledge
distillation**. Undergraduate final-year research project.

- **Teacher:** [AdaIR](https://github.com/c-yn/AdaIR) (ICLR 2025) — Restormer-style
  transformer, ~29M params, frozen throughout.
- **Student:** NAFNet (width 32) — CNN chosen for clean mobile-NPU export.
- **Goal:** distill AdaIR → NAFNet, quantize to INT8, deploy to Qualcomm QNN /
  TFLite / Jetson TensorRT, benchmark on-device latency.
- **Protocol:** AirNet/PromptIR **3-degradation** setting — denoise, derain,
  dehaze. Fixed input/output resolution. No super-resolution.

## Status

**Phase 01 (weeks 1–4): de-risk the deployment path, validate the data/eval
pipeline against published numbers, establish a frozen baseline.** No
distillation code in this phase.

Gates: **G1** export smoke test · **G2/G3** teacher reproduction (±0.10 dB) ·
**G4** B0 baseline frozen.

## Engineering rules

Config-driven (no hardcoded paths/magic numbers) · deterministic (`--seed`
everywhere) · one eval harness (`src/eval/evaluate.py`) · documented metric
conventions · checkpoint-resumable · every run writes a run directory · typed +
docstringed · tested numbers · **no silent fallbacks** (missing checkpoint /
shape mismatch / absent config key → raise).

## Setup

```bash
conda create -y -n adair python=3.11 && conda activate adair
# optional CUDA torch first (see requirements.txt), then:
pip install -r requirements.txt
cp configs/paths.yaml configs/paths.local.yaml   # edit local roots
```

## Layout

See the directory tree; each `src/` subpackage owns one concern (data, models,
losses, train, eval, export, cache, utils). Configs live under `configs/`,
deliverables under `reports/`, tests under `tests/`.
