# Student architecture sweep — Task 1.5a

All figures at **256x256**, batch 1. **MACs = FLOPs/2** (`torch.utils.flop_counter`, convention pinned by a unit test).

Selection is by **MACs and on-device latency, not parameters**: a NAFBlock at H/8 costs ~1/64 the MACs of the same block at full resolution, so parameter count cannot rank these architectures by cost.

## Teacher reference (measured, not quoted)

**AdaIR** — `28.78M` params, **`161.75` GMACs** @ 256x256.

> Measured by instantiating `third_party/AdaIR`. Note the counter does not model the FFT work in AdaIR's frequency modules, so this is a mild *under*-estimate of true teacher cost — reduction ratios below are therefore conservative.

Compression rule: an arm counts only if **both** params and MACs shrink by ≥4x (params ≤ 7.20M, MACs ≤ 40.44 GMACs).

## Sweep results

| config | width | blocks | params | GMACs | params÷ | MACs÷ | ≥4x both |
|---|---|---|---|---|---|---|---|
| `w16_b8` | 16 | b8 | 2.44M | 2.13 | 11.8x | 76.1x | yes |
| `w16_b14` | 16 | b14 | 3.15M | 2.74 | 9.1x | 58.9x | yes |
| `w16_b28` | 16 | b28 | 4.35M | 4.09 | 6.6x | 39.6x | yes |
| `w16_sidd` | 16 | sidd | 7.37M | 4.13 | 3.9x | 39.2x | **no** |
| `w24_b8` | 24 | b8 | 5.44M | 4.67 | 5.3x | 34.6x | yes |
| `w24_b14` | 24 | b14 | 7.02M | 6.05 | 4.1x | 26.7x | yes |
| `w32_b8` | 32 | b8 | 9.62M | 8.21 | 3.0x | 19.7x | **no** |
| `w24_b28` | 24 | b28 | 9.68M | 9.05 | 3.0x | 17.9x | **no** |
| `w24_sidd` | 24 | sidd | 16.46M | 9.11 | 1.7x | 17.7x | **no** |
| `w32_b14` | 32 | b14 | 12.42M | 10.66 | 2.3x | 15.2x | **no** |
| `w32_b28` | 32 | b28 | 17.11M | 15.96 | 1.7x | 10.1x | **no** |
| `w32_sidd` | 32 | sidd | 29.16M | 16.05 | 1.0x | 10.1x | **no** |

## Why placement changes params but not MACs

| config | total blocks | params | GMACs | GMACs/block |
|---|---|---|---|---|
| `w16_b8` | 17 | 2.44M | 2.13 | 0.1251 |
| `w16_b14` | 23 | 3.15M | 2.74 | 0.1193 |
| `w16_b28` | 36 | 4.35M | 4.09 | 0.1135 |
| `w16_sidd` | 36 | 7.37M | 4.13 | 0.1147 |
| `w24_b8` | 17 | 5.44M | 4.67 | 0.2750 |
| `w24_b14` | 23 | 7.02M | 6.05 | 0.2633 |
| `w24_b28` | 36 | 9.68M | 9.05 | 0.2513 |
| `w24_sidd` | 36 | 16.46M | 9.11 | 0.2532 |
| `w32_b8` | 17 | 9.62M | 8.21 | 0.4831 |
| `w32_b14` | 23 | 12.42M | 10.66 | 0.4634 |
| `w32_b28` | 36 | 17.11M | 15.96 | 0.4432 |
| `w32_sidd` | 36 | 29.16M | 16.05 | 0.4457 |

`b28` and `sidd` have the **same total block count (36)** and therefore near-identical MACs, despite `sidd` carrying ~1.7x the parameters. Each downsample quarters the spatial area but doubles the channel count (4x in `c^2`), so a NAFBlock costs the **same MACs at every depth**. MACs therefore track *total blocks x width^2*; placement is MAC-free but parameter-expensive.

Consequence: deepening the pyramid buys capacity at zero MAC cost, but costs model size and memory bandwidth — which is why the parameter rule remains a useful independent constraint.

## Params-per-MAC mismatch (why the family targets bind)

AdaIR sits at **0.18M params per GMAC**; NAFNet variants sit at ~1.30M params per GMAC — roughly 7.3x more parameter-heavy per unit of compute. Attention is compute-dense; convolution is parameter-dense.

So parameter-reduction and MAC-reduction are **not independently dialable**: hitting the parameter rule forces a much larger MAC reduction than the nominal arm target.

## Proposed family

| arm | target MACs÷ | chosen config | params | GMACs | actual MACs÷ | params÷ |
|---|---|---|---|---|---|---|
| **S** | — | `w24_b14` | 7.02M | 6.05 | 26.7x | 4.1x |
| **M** | — | `w24_b8` | 5.44M | 4.67 | 34.6x | 5.3x |
| **L** | — | `w16_b28` | 4.35M | 4.09 | 39.6x | 6.6x |

**Assignment warnings — these need a decision:**

- arm M: target 10x MAC reduction is UNREACHABLE within the compression rule; closest eligible is `w24_b8` at 34.6x
- arm L: target 4x MAC reduction is UNREACHABLE within the compression rule; closest eligible is `w16_b28` at 39.6x

Ablation-grid discipline: the full grid runs on **M** only. S and L get B0 plus the single best KD config.

## On-device verification

Skipped (`--no-aihub`).
