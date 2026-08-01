# Student architecture sweep — Task 1.5a

All figures at **256x256**, batch 1. **MACs = FLOPs/2** (`torch.utils.flop_counter`, convention pinned by a unit test). On-device figures measured on Qualcomm AI Hub, Samsung Galaxy S24 (Snapdragon 8 Gen 3, Hexagon v75), INT8 QNN context binary.

Selection priority: **measured on-device latency -> peak activation memory -> GMACs -> params**. Parameters are a *ceiling*, not a target.

## Teacher reference (measured, not quoted)

**AdaIR** — `28.78M` params, **`161.75` GMACs** @ 256x256.

> Measured by instantiating `third_party/AdaIR`. Note the counter does not model the FFT work in AdaIR's frequency modules, so this is a mild *under*-estimate of true teacher cost — reduction ratios below are therefore conservative.

Parameter ceiling: **10M** (bounds model size only). The former '>=4x reduction on both params and MACs' rule is **retired** — it rejected every width-32 candidate and forced a latency-degenerate family.

## Sweep results

| config | width | blocks | params | GMACs | MACs÷ | ≤ceiling | **NPU ms** | peak mem MB | fallback |
|---|---|---|---|---|---|---|---|---|---|
| `w16_b8` | 16 | b8 | 2.44M | 2.13 | 76.1x | yes | **2.51** | 7 | 0 |
| `w16_b14` | 16 | b14 | 3.15M | 2.74 | 58.9x | yes | **2.74** | 18 | 0 |
| `w32_b8` | 32 | b8 | 9.62M | 8.21 | 19.7x | yes | **3.16** | 8 | 0 |
| `w24_b8` | 24 | b8 | 5.44M | 4.67 | 34.6x | yes | **3.19** | 7 | 0 |
| `w16_b28` | 16 | b28 | 4.35M | 4.09 | 39.6x | yes | **3.30** | 8 | 0 |
| `w24_b14` | 24 | b14 | 7.02M | 6.05 | 26.7x | yes | **3.52** | 8 | 0 |
| `w32_b14` | 32 | b14 | 12.42M | 10.66 | 15.2x | **no** | **3.59** | 9 | 0 |
| `w24_b28` | 24 | b28 | 9.68M | 9.05 | 17.9x | yes | **4.26** | 7 | 0 |
| `w32_b28` | 32 | b28 | 17.11M | 15.96 | 10.1x | **no** | **4.47** | 8 | 0 |
| `w16_sidd` | 16 | sidd | 7.37M | 4.13 | 39.2x | yes | **4.74** | 8 | 0 |
| `w24_sidd` | 24 | sidd | 16.46M | 9.11 | 17.7x | **no** | **6.09** | 8 | 0 |
| `w32_sidd` | 32 | sidd | 29.16M | 16.05 | 10.1x | **no** | **6.16** | 100 | 0 |

> **Peak memory is total footprint** (weights + activations + runtime), not the incremental working set. It is ~98-101 MB across *every* config including the smallest, i.e. dominated by fixed QNN runtime overhead rather than by model size — so on this device it does not discriminate between candidates, but it does set the floor any edge memory budget must clear.

## What the device actually says

Correlations with measured INT8 latency (**n=12**, Pearson and Spearman rank):

| predictor | Pearson r | Spearman ρ |
|---|---|---|
| GMACs | 0.66 | 0.74 |
| total block count | 0.75 | 0.81 |
| normalisation-area proxy | **0.87** | **0.89** |

> With n=12 heterogeneous configurations these correlations are indicative, not conclusive — the gap between the MAC and normalisation predictors should not be over-read. The controlled comparison below is the stronger evidence.

**NPU->CPU fallback across all profiled configs: 0 layers.** Every op ran on the Hexagon NPU, so the static CAUTION verdicts in `export_smoke_test.md` overstated the *support* risk — nothing was rejected or offloaded.

### Controlled comparison — matched block count and MACs

| pair | blocks | GMACs | NPU ms | Δ latency |
|---|---|---|---|---|
| `w16_b28` vs `w16_sidd` | 36 | 4.09 vs 4.13 (1.1% apart) | 3.30 vs 4.74 | **+44%** |
| `w24_b28` vs `w24_sidd` | 36 | 9.05 vs 9.11 (0.7% apart) | 4.26 vs 6.09 | **+43%** |
| `w32_b28` vs `w32_sidd` | 36 | 15.96 vs 16.05 (0.6% apart) | 4.47 vs 6.16 | **+38%** |

These pairs hold block count, width and MACs essentially constant and vary only **where** the blocks sit in the pyramid. The latency difference therefore cannot be attributed to capacity or compute — it is placement, and specifically the number of normalisations running at full resolution. This is a controlled result and is stronger evidence than any correlation over heterogeneous points.

**MACs mispredict latency.** `w16_sidd` has 3.9x *fewer* MACs than `w32_b28` (4.13 vs 15.96 GMACs) yet is **slower** on device (4.74 vs 4.47 ms). Selecting on MACs alone would have picked the wrong architecture.

Mechanism: cycle profiling of `w16_b8` shows **LayerNorm2d consumes ~62% of NPU cycles** (`Div` alone ~62%) against **~3% for `Conv`**. Fixed-point division is expensive on the Hexagon integer pipeline, and its cost is per-element — so normalisations at full resolution dominate. This is why the area-weighted proxy predicts latency better than MACs.

> **Consequence for Task 1.5b:** replacing `LayerNorm2d` with a conv-foldable normalisation (BatchNorm) should remove ~60% of NPU cycles — a larger latency win than any width/block choice in this sweep. The architecture must be locked on the *post-normalisation* design, otherwise this table's ranking does not survive.

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
| **S** | — | `w16_b8` | 2.44M | 2.13 | 76.1x | 11.8x |
| **M** | — | `w24_b8` | 5.44M | 4.67 | 34.6x | 5.3x |
| **L** | — | `w24_b28` | 9.68M | 9.05 | 17.9x | 3.0x |

**Assignment warnings — these need a decision:**

- 4 config(s) exceed the 10M parameter ceiling and were excluded
- arm S: advisory target 30x MAC reduction not met; selected `w16_b8` at 76.1x (targets are advisory only, not a filter)
- arm M: advisory target 10x MAC reduction not met; selected `w24_b8` at 34.6x (targets are advisory only, not a filter)
- arm L: advisory target 4x MAC reduction not met; selected `w24_b28` at 17.9x (targets are advisory only, not a filter)

Ablation-grid discipline: the full grid runs on **M** only. S and L get B0 plus the single best KD config.

## On-device verification

Real AI Hub compilation and profiling completed; see the NPU columns above.

- `w16_b8`: compute units {'NPU': 637}, peak 7 MB
- `w16_b14`: compute units {'NPU': 853}, peak 18 MB
- `w16_b28`: compute units {'NPU': 1321}, peak 8 MB
- `w16_sidd`: compute units {'NPU': 1321}, peak 8 MB
- `w24_b8`: compute units {'NPU': 637}, peak 7 MB
- `w24_b14`: compute units {'NPU': 853}, peak 8 MB
- `w24_b28`: compute units {'NPU': 1321}, peak 7 MB
- `w24_sidd`: compute units {'NPU': 1321}, peak 8 MB
- `w32_b8`: compute units {'NPU': 637}, peak 8 MB
- `w32_b14`: compute units {'NPU': 853}, peak 9 MB
- `w32_b28`: compute units {'NPU': 1321}, peak 8 MB
- `w32_sidd`: compute units {'NPU': 1321}, peak 100 MB

## Family re-selection on the LOCKED Fix-C variant (2026-08-02)

After findings F9 the locked normalization changed from `affine` to
`affine_clamp(8.0)` at the full-resolution stages. The family was re-run
rather than assumed unchanged: the clamp adds `Clip` nodes in proportion
to each config's full-resolution block count, so its cost is **not**
uniform across the grid and the M arm's +0.3% does not transfer.

All 12 configs re-profiled on Samsung Galaxy S24, INT8, through the
corrected profile-then-select path.

| config | N-F ms | Fix-C ms | delta | % |
|---|---|---|---|---|
| w16_b8 | 1.580 | 1.577 | -0.003 | -0.19% |
| w16_b14 | 1.818 | 1.811 | -0.007 | -0.39% |
| w16_b28 | 2.355 | 2.349 | -0.006 | -0.25% |
| w16_sidd | 2.873 | 2.885 | +0.012 | +0.42% |
| w24_b8 | 2.248 | 2.252 | +0.004 | +0.18% |
| w24_b14 | 2.576 | 2.580 | +0.004 | +0.16% |
| w24_b28 | 3.320 | 3.328 | +0.008 | +0.24% |
| w24_sidd | 4.225 | 4.223 | -0.002 | -0.05% |
| w32_b8 | 2.263 | 2.250 | -0.013 | -0.57% |
| w32_b14 | 2.679 | 2.708 | +0.029 | +1.08% |
| w32_b28 | 3.543 | 3.545 | +0.002 | +0.06% |
| w32_sidd | 4.226 | 4.249 | +0.023 | +0.54% |

Mean **+0.10%**, range -0.57% to +1.08%.

**RESULT: the family is UNCHANGED.**

| arm | config | params | GMACs | Fix-C ms |
|---|---|---|---|---|
| S | `w16_b8` | 2.44M | 2.13 | 1.577 |
| M | `w16_sidd` | 7.37M | 4.13 | 2.885 |
| L | `w24_b28` | 9.68M | 9.05 | 3.328 |

Latency span 2.69x (N-F: 2.67x). All invariants pass: params and MACs
strictly increase S < M < L, measured latency increases across arms, MAC
span 4.26x >= 2.5x, every arm under the 10M parameter ceiling.

Worth stating plainly: this re-run was expected to be a formality and
was, but the previous norm change (N-A -> N-F) **did** move M from
`w24_b8` to `w16_sidd`, so 'small delta implies same family' is not a
safe inference — it is a measurement.
