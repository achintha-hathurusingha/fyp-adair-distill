# Family re-selection on locked normalization (FC)

Selected through the corrected **profile-then-select** path: measured latency is attached to every candidate *before* `assign_family` runs. The earlier bug ran selection first, so it never saw the latency it selects on.

## Pre-fix (N-A) vs post-fix latency

| config | params | GMACs | N-A ms | FC ms | speedup | rank N-A | rank FC |
|---|---|---|---|---|---|---|---|
| `w16_b8` | 2.44M | 2.13 | 2.513 | 1.577 | 1.59x | 1 | 1 |
| `w16_b14` | 3.15M | 2.74 | 2.742 | 1.811 | 1.51x | 2 | 2 |
| `w32_b8` | 9.62M | 8.21 | 3.160 | 2.250 | 1.40x | 3 | 3 |
| `w24_b8` | 5.44M | 4.67 | 3.195 | 2.252 | 1.42x | 4 | 4 |
| `w16_b28` | 4.35M | 4.09 | 3.299 | 2.349 | 1.40x | 5 | 5 |
| `w24_b14` | 7.02M | 6.05 | 3.519 | 2.580 | 1.36x | 6 | 6 |
| `w32_b14` | 12.42M | 10.66 | 3.590 | 2.708 | 1.33x | 7 | 7 |
| `w16_sidd` | 7.37M | 4.13 | 4.742 | 2.885 | 1.64x | 10 | 8 ⟵ moved |
| `w24_b28` | 9.68M | 9.05 | 4.256 | 3.328 | 1.28x | 8 | 9 ⟵ moved |
| `w32_b28` | 17.11M | 15.96 | 4.469 | 3.545 | 1.26x | 9 | 10 ⟵ moved |
| `w24_sidd` | 16.46M | 9.11 | 6.091 | 4.223 | 1.44x | 11 | 11 |
| `w32_sidd` | 29.16M | 16.05 | 6.158 | 4.249 | 1.45x | 12 | 12 |

**Latency span: 2.45x (N-A) -> 2.69x (FC).** Removing the large roughly-fixed normalization cost decompresses the range, as predicted.

## Selected family

| arm | config | params | GMACs | latency | MACs÷ |
|---|---|---|---|---|---|
| **S** | `w16_b8` | 2.44M | 2.13 | 1.577 ms | 76.1x |
| **M** | `w16_sidd` | 7.37M | 4.13 | 2.885 ms | 39.2x |
| **L** | `w24_b28` | 9.68M | 9.05 | 3.328 ms | 17.9x |

MAC span **4.26x**, latency span **2.11x**.

## Invariant check

Passed: params and MACs strictly increase S < M < L, measured latency increases across arms, and the MAC span clears 2.5x.

### Warnings

- 4 config(s) exceed the 10M parameter ceiling and were excluded
- arm S: advisory target 30x MAC reduction not met; selected `w16_b8` at 76.1x (targets are advisory only, not a filter)
- arm M: advisory target 10x MAC reduction not met; selected `w16_sidd` at 39.2x (targets are advisory only, not a filter)
- arm L: advisory target 4x MAC reduction not met; selected `w24_b28` at 17.9x (targets are advisory only, not a filter)
