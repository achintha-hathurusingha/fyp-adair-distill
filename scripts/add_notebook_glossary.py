"""Insert notation guides as MARKDOWN cells only.

Markdown cells carry no outputs, so every executed result in the notebook is
preserved untouched — no re-run needed.
"""
import nbformat as nbf

GLOSSARY = """---
## Notation guide

Every name in this notebook follows one of five schemes. Read this once and the
rest is unambiguous.

### Normalization variants — `N-*`

Which normalization the network uses. This is the variable Task 1.5b tested.

| id | meaning | inference cost |
|---|---|---|
| **N-A** | `LayerNorm2d` **everywhere** — the reference | many ops: ReduceMean, Sub, Pow, Sqrt, Div |
| **N-F** | `LayerNorm2d` deep, **affine-only at full resolution** | the Div/Sqrt removed where they cost most |
| **N-E** | affine **everywhere** — no statistics at all | foldable, nearly free |
| N-A′ | N-A rewritten to avoid `Div` — **rejected**, 14% *slower* (F3/F4) | — |
| N-B / N-C | BatchNorm / Identity — latency datapoints only | — |

Why "full resolution" matters: normalization cost is **per element**, so the
stages at full resolution dominate even though they are few (F1).

### Training arms — `Q-*`, `M-*`, `B0*`

An *arm* is one training run with everything held fixed except the thing being
tested.

| prefix | geometry | purpose |
|---|---|---|
| **Q-A / Q-F / Q-E** | S arm, `w16_b8` | the normalization ablation: N-A / N-F / N-E |
| **Q-E1 / Q-E2 / Q-E3** | S arm | escalation rungs trying to rescue N-E (half LR, + clipping, + residual init) — all diverged |
| **M-A / M-F** | M arm, `w16_sidd` | spot-check that N-F's result holds on the larger config |
| **B0** | M arm | **the baseline.** Ground truth only, *no distillation* |
| B0-QA | M arm | control: B0's schedule with full LayerNorm |
| B0-FIXC | M arm | validation: B0's schedule with the clamp |

So `Q-A` and `M-A` are the same normalization (N-A) on different geometries;
`Q-F`, `M-F` and `B0` all use the full-resolution-affine family.

### Fix candidates — `Fix-*`

Answers to the F9 divergence. All three bound the failure; they differ in cost.

| id | change | measured cost |
|---|---|---|
| **Fix-B** | restore real `LayerNorm2d` at full resolution | **1.65x latency** — gives back what N-F won |
| **Fix-C** | affine **+ hard clamp** at ±8 | **+0.3% latency** — **LOCKED** |
| Fix-A | clamp `dec3` only, not both full-res stages | not built — `full_res_norm_type` applies symmetrically |

### Family arms — `S` / `M` / `L`

Three model sizes for the deployment sweep. Named by geometry:
`w16_b8` = width 16, "b8" block layout. **M = `w16_sidd` is the arm B0 trains.**

### Findings — `F1`–`F9`

Numbered results in `reports/findings.md`. The ones used here:

| id | finding |
|---|---|
| **F1** | normalization is 62% of NPU cycles; convolution is 3.4% |
| F3 / F4 | per-op cycle attribution misleads in fused graphs; `rsqrt` rewrite is a net loss |
| F6 | removing normalization fails by activation-scale growth, not gradient spikes |
| **F7** | the AdaIR teacher **cannot be exported at all** |
| F8 | batch-size policy: effective batch 32 via gradient accumulation |
| **F9** | N-F's unnormalized full-resolution stage turns a rare input into a fatal gradient |

### Metrics in the training logs

| name | meaning |
|---|---|
| `maxgn` | **max gradient norm** in the interval — measured *before* clipping |
| `clip` | how many optimizer steps had their gradient clipped |
| `skip` | steps **dropped** because the gradient was Inf/NaN — should always be 0 |
| `clampeng` | % of forward passes where the Fix-C clamp actually engaged |
| `premax` | largest **pre-clamp** magnitude seen — says whether the bound has headroom |
| σ (sigma) | Gaussian noise standard deviation, in 0–255 units |

`clampeng` and `premax` answer different questions: *how often* the clamp fires
versus *how hard* it is being pushed. Neither alone is sufficient.
"""

NOTES = {
    2:  "> **Notation.** *AdaIR* is the teacher — the 28.8M model this project distils from.\n"
        "> *Opset* is the ONNX operator-set version; different opsets support different ops,\n"
        "> which is why the export was retried at 17, 18 and 20.\n",
    5:  "> **Notation.** *INT8* = 8-bit integer quantization, required by the NPU.\n"
        "> *QNN* is Qualcomm's Neural Network runtime; a *context binary* is the compiled\n"
        "> artefact that actually runs on the phone. All latencies are on a Galaxy S24\n"
        "> (Snapdragon 8 Gen 3, Hexagon v75 NPU).\n",
    8:  "> **Notation.** Arms are `Q-*` on the S geometry (`w16_b8`). `Q-A` = N-A,\n"
        "> `Q-F` = N-F, `Q-E` = N-E. `Q-E1/2/3` are rescue attempts for N-E.\n"
        "> `M-A`/`M-F` repeat the A-vs-F comparison on the larger M geometry.\n",
    11: "> **Notation.** `dec3` is the **full-resolution decoder stage** — the last stage\n"
        "> before the output, and one of the two places N-F replaces LayerNorm with\n"
        "> affine. `enc0` is its encoder-side counterpart. `middle_blks` is the deepest,\n"
        "> lowest-resolution part of the U-Net.\n",
    14: "> **Notation.** *Clamp bound* = the magnitude limit, here ±8. *Containment* is\n"
        "> the max output magnitude on the crop that caused the divergence — lower is\n"
        "> better. *Mann-Kendall* is a rank-based trend test, chosen because the series\n"
        "> is heavy-tailed and an ordinary regression slope would be dominated by single\n"
        "> large values.\n",
    17: "> **Notation.** *GMACs* = billions of multiply-accumulate operations, the\n"
        "> compute measure that predicts latency better than parameter count.\n"
        "> *Seed* = the RNG seed; three seeds are run so the reported figure can carry\n"
        "> a mean and spread rather than a single number.\n",
    19: "> **Notation.** *BSD68* is the standard 68-image denoising benchmark.\n"
        "> *PSNR* (dB, higher better) measures pixel fidelity; *SSIM* (0–1) measures\n"
        "> structural similarity. Both go through the locked harness so conventions\n"
        "> (RGB vs Y-channel, border crop, data range) are identical everywhere.\n",
    22: "> **Notation.** *Synthetic* = our own Gaussian noise added to a clean image, so\n"
        "> ground truth exists and PSNR is meaningful. *Native* = the downloaded JPEG\n"
        "> as-is, carrying real capture noise and compression artefacts — **no ground\n"
        "> truth exists, so no PSNR is reported for it.**\n",
    25: "> **Notation.** *FP32* = 32-bit float, how the model trains. *INT8* = the\n"
        "> quantized deployment form. *PTQ* (post-training quantization) converts one to\n"
        "> the other without retraining; *calibration* is the sample pass that chooses\n"
        "> the numeric ranges.\n",
}

nb = nbf.read('notebooks/fyp_demo.ipynb', as_version=4)
before_out = sum(len(c.get('outputs', [])) for c in nb.cells)

cells, added = [], 0
for i, c in enumerate(nb.cells):
    if i in NOTES:
        cells.append(nbf.v4.new_markdown_cell(NOTES[i]))
        added += 1
    cells.append(c)
    if i == 0:                       # glossary straight after the title
        cells.append(nbf.v4.new_markdown_cell(GLOSSARY))
        added += 1

nb.cells = cells
nbf.write(nb, 'notebooks/fyp_demo.ipynb')

after_out = sum(len(c.get('outputs', [])) for c in nb.cells)
print(f"added {added} markdown cells -> {len(nb.cells)} total")
print(f"outputs before {before_out}, after {after_out}  "
      f"{'PRESERVED' if before_out == after_out else 'CHANGED — problem'}")
