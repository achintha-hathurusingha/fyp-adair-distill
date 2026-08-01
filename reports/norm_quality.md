# Normalization quality ablation — Task 1.5b

Latency was measured in Task 1.5c on **untrained** weights; this task measures the quality axis on **trained** models. Config `w16_b8`, denoising only (σ ∈ {15, 25, 50}), 30k iterations at batch 32 and patch 128, single seed, identical data/augmentation/optimiser/schedule across every arm. Validation on BSD68 through the locked harness (`src/eval/evaluate.py`).

## Decision: **N-F**

Q-E is 4.296 dB below Q-A, but Q-F holds at -0.005 dB. **Lock N-F** for 1.59x.

## Final results

| arm | normalization | BSD68 PSNR | ΔPSNR vs Q-A | INT8 ms | speedup | peak VRAM | iters |
|---|---|---|---|---|---|---|---|
| **Q-A** | LayerNorm2d everywhere | 31.019 | +0.000 | 2.513 | 1.00x | 2.14 GB | 30000 |
| **Q-F** | affine @ full-res, LayerNorm deeper | 31.014 | -0.005 | 1.580 | 1.59x | 1.83 GB | 30000 |
| **Q-E** | affine everywhere | 25.080 | -5.939 | 1.072 | 2.34x | 0.00 GB | 3040 |
| **Q-E1** | affine, half LR + long warmup | 26.695 | -4.324 | 1.072 | 2.34x | 0.00 GB | 4926 |
| **Q-E2** | affine, half LR + clip 1.0 | 26.695 | -4.324 | 1.072 | 2.34x | 0.00 GB | 4926 |
| **Q-E3** | affine, half LR + clip + resid init 0.1 | 26.723 | -4.296 | 1.072 | 2.34x | 0.00 GB | 5899 |

## Validation curves

![validation curves](norm_quality_curves.png)

Panels: BSD68 PSNR, training loss (log), gradient norm, and mean activation magnitude at encoder level 0. The latter two are the trainability diagnostics — a capacity failure shows as a stable curve that plateaus lower, whereas a trainability failure shows as diverging or oscillating gradients and activations.

## Diagnostics

- **M-A**: final loss 0.02103, grad norm 0.029 (min 0.029, max 0.319), activations enc0=0.126, enc1=0.095, enc2=0.100, enc3=0.138, middle=0.117
- **M-F**: final loss 0.02087, grad norm 0.040 (min 0.031, max 0.245), activations enc0=0.091, enc1=0.087, enc2=0.105, enc3=0.116, middle=0.066
- **Q-A**: final loss 0.01990, grad norm 0.024 (min 0.020, max 0.158), activations enc0=0.127, enc1=0.082, enc2=0.100, enc3=0.202, middle=0.150
- **Q-E**: final loss nan, grad norm 0.117 (min 0.117, max 0.117), activations   **DIVERGED**
- **Q-E1**: final loss nan, grad norm 0.111 (min 0.111, max 0.153), activations   **DIVERGED**
- **Q-E2**: final loss nan, grad norm 0.111 (min 0.111, max 0.153), activations   **DIVERGED**
- **Q-E3**: final loss nan, grad norm 0.063 (min 0.063, max 0.160), activations   **DIVERGED**
- **Q-F**: final loss 0.01989, grad norm 0.016 (min 0.015, max 0.123), activations enc0=0.103, enc1=0.079, enc2=0.104, enc3=0.204, middle=0.148

## Caveats

- Ablation is on **w16_b8 (width 16) and denoising only**. Wider models have more channels per normalization and may behave differently; derain/dehaze may respond differently to the loss of normalization. If the decision were close, a spot-check on `w24_b8` and one non-denoising task would be warranted before locking.
- **30k iterations, not to convergence.** A ranking at 30k is not guaranteed to hold at full training length.
- Single seed. Differences smaller than seed-to-seed variance should not be over-read.
