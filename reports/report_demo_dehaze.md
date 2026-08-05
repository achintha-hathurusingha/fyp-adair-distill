# Dehaze gap demo — does AdaIR's advantage transfer through supervision?

**Result: yes, partially.** Response distillation from AdaIR closed **11.5%** of
the teacher–student gap on held-out dehazing, at zero inference cost to the
student.

| model | params | PSNR | SSIM | gap to teacher |
|---|---|---|---|---|
| AdaIR (teacher) | 28,784,824 | **34.5056** | 0.9878 | — |
| M, ground truth only | 7,371,923 | 32.8898 | 0.9821 | **+1.6159 dB** |
| M, ground truth + KD | 7,371,923 | **33.0759** | 0.9828 | **+1.4298 dB** |

**Distillation delta: +0.1861 dB, +0.0007 SSIM.** Same architecture, same data,
same schedule, same seed — the two student runs differ in exactly one thing, the
added loss term.

---

## Scope — read this before quoting any number above

This is a **feasibility demo**, not a Phase 02 result. Specifically:

* **Dehazing only.** One of the three protocol degradations.
* **One seed.** B0-denoise and B0-v2 use three seeds and report seed-to-seed
  variance (0.0079 dB across three B0-denoise seeds). This does not. The
  +0.1861 dB delta is **not** established as larger than seed noise, because
  seed noise was not measured here. It is consistent across the whole second
  half of training (see the curve below), which is weaker evidence than a
  multi-seed interval and is stated as such.
* **A 4,000-image subset** of RESIDE-OTS's 72,135, seed 1234, recorded in
  `reports/dehaze_train_list.txt`.
* **60,000 iterations**, chosen for a deadline. Convergence is evidenced below,
  not asserted.
* **KD weight 1.0, untuned.** Equal weighting of the two loss terms, chosen
  rather than searched. A sweep would be a second experiment.

---

## Why this experiment, and why dehazing (F7)

AdaIR's advantage comes substantially from **FreModule**, which mines the
frequency domain via FFT. That same mechanism is exactly why the teacher
**cannot deploy**: `aten::fft_fft2` has no ONNX export path, and finding F7
records all five attempted routes failing —

| attempt | result |
|---|---|
| unpatched / TorchScript / opset 17 | `SymbolicValueError` |
| patched / TorchScript / 17 and 20 | `UnsupportedOperatorError: aten::fft_fft2` |
| unpatched / dynamo / 18 and 20 | `TorchExportError` |

So the teacher is 3.90x the student's parameters *and* undeployable, while the
student exports cleanly and runs at 2.885 ms INT8 on a Galaxy S24.

The question this demo asks: **can the advantage that frequency mining buys be
transferred through supervision alone?** If it can, the student stays a pure
convolutional network that computes no frequencies at inference, exports
unchanged, and the teacher's undeployability stops being a blocker.

A detail that sharpens the point, found while building this: the teacher could
not run in the student's training precision either. `aten::fft_fft2` raised
`Unsupported dtype BFloat16` the moment the KD term first executed, so the
teacher runs in fp32 outside autocast. **The same frequency machinery that
blocks export also blocks mixed precision.**

---

## Step 0 — safety gate before training

Dehazing is a distribution this architecture had never been trained on alone,
and F10 established that low-variance input is where it is fragile. Two checks,
both clean, before any GPU time was committed.

### Haze-shaped stress test

`scripts/stress_test_norm.py --dehaze` adds cases from the atmospheric model
`I = J·t + A·(1−t)`: transmission `t` swept 0.9 → 0, pure airlight, a vertical
veiling gradient, flat bright fog, a desaturated grey-out, plus the three
lowest-variance **real** RESIDE-OTS crops.

Haze is the **bright mirror** of the crop that killed B0 (finding F9). Both are
low-variance, but haze compresses the scene toward a bright constant rather than
a dark one. Since F10 showed low variance is the fragile regime, the bright
branch needed testing explicitly rather than assuming the dark one covered it.

Run against **trained** B0-v2 weights, not untrained — the script's own
docstring records that untrained weights prove nothing here, because NAFNet
initialises `beta`/`gamma` to zero and every block is the identity at init.

**22 cases, 0 failures.** Locked N-F / Fix-C outputs land at 0.86–1.00 across
every haze case and are monotone in `t`.

### Per-task clamp telemetry

B0-v2 reports one engagement rate over a batch that is one third of each task,
so 12% on one task reads identically to 4% everywhere. `scripts/clamp_by_task.py`
feeds single-task batches through the finished checkpoint, 1,600 samples each:

| task | dec3 engage | dec3 premax | enc3 engage | enc3 premax |
|---|---|---|---|---|
| denoise | 2.812% | 18.64 | 0.000% | 11.14 |
| derain | 4.062% | 11.62 | 0.000% | 11.10 |
| **dehaze** | **0.125%** | **8.458** | **0.000%** | 11.26 |

**Dehaze is the calmest task by a wide margin**, and the `enc3` clamp is inert on
all three — as F9 and F12 predicted, since LayerNorm already bounds its own
output structurally. Nothing new to resolve. Gate: **PROCEED**.

Both dehaze runs subsequently confirmed this: `dec3` finished at **0.00%**
engagement on both, and `enc3` at 0.00% with pre-clamp magnitude 11.7–12.3
against its bound of 32.

---

## Data — the split, and why it splits where it does

RESIDE-OTS renders roughly **35 hazy variants from each clear source image**
(`0025_0.8_0.04.jpg`, `0025_0.9_0.20.jpg`, … all from `0025`). Splitting on hazy
filenames would put different hazings of the **same scene** on both sides, and
the held-out score would measure memorisation rather than generalisation.

`scripts/make_dehaze_split.py` therefore splits by **clear source stem**, and
asserts no leaked stem rather than trusting its own logic:

```
source: 72,135 hazy images from 2,061 clear stems (35.0 per stem)
seed 1234 -> 1,911 train stems / 150 held-out stems, disjoint
train:    4,000 hazy images
held-out:   150 hazy images, one per reserved stem (150 distinct scenes)
no leakage: 150 held-out scenes share no clear source with the 4,000 training images
```

Both lists are committed with the seed in their headers. The held-out set is
materialised as `data/test/dehaze/demo/{input,target}/` so the existing
`PairedTestDataset` reads it with no new code, and targets keep the clear-stem
name so pairing uses the real `name.split('_')[0]` rule rather than a special
case invented for this directory.

---

## Step 1 & 4 — the two runs

Identical but for the loss. `configs/train/m_dehaze_baseline.yaml` and
`m_dehaze_kd.yaml` are asserted section-by-section identical except `distill`
(`test_kd_config_differs_from_its_baseline_only_by_distill`), because otherwise
the delta would not be attributable to distillation.

| | GT only | GT + KD |
|---|---|---|
| arm | `M-DEHAZE` | `M-DEHAZE-KD` |
| commit | `ec389a9` | `932504f` |
| best PSNR | 32.8898 @ 54k | **33.0759 @ 54k** |
| diverged | false | false |
| wall clock | 5.82 h | 8.73 h |
| VRAM | 2.01 GB | 4.28 GB |

Both ran concurrently on one RTX 4090, pinned to `taskset -c 0-7,12-31`
(see `reports/devon_cpu_mitigation.md`).

### The KD method

```
loss = Charbonnier(student(x), ground_truth)
     + 1.0 * Charbonnier(student(x), teacher(x))
```

**Response distillation — one added term.** No feature hooks, no adapters or
projectors, no attention transfer, no temperature (a softmax concept that does
not apply to dense regression; the "soft target" here is literally the teacher's
restored image), no multi-loss schedule.

The teacher is `adair-single-dehaze.ckpt` loaded through `FrozenTeacher`, which
enforces zero missing and zero unexpected keys, `eval()` mode, and
`requires_grad=False` on every parameter, re-asserted on every forward.

**Computed live, not cached.** The student's input is a random crop chosen per
index, so a per-image cache would have to be cropped after the fact — and
`teacher(crop) ≠ crop(teacher(image))` once the receptive field crosses the crop
boundary. Live is exact and costs throughput: the KD run took 1.50x the wall
clock of the baseline.

### Convergence evidence

Judged from the curve, not from having picked 60,000 in advance.

| iteration | GT only | GT + KD | KD − GT |
|---|---|---|---|
| 2,000 | 25.4906 | 25.5075 | +0.0168 |
| 6,000 | 30.0003 | 29.9596 | −0.0408 |
| 12,000 | 31.0945 | 31.2636 | +0.1691 |
| 18,000 | 31.6688 | 31.9551 | +0.2863 |
| 24,000 | 32.1006 | 32.3727 | +0.2722 |
| 30,000 | 32.3129 | 32.6207 | +0.3078 |
| 36,000 | 32.6517 | 32.9002 | +0.2485 |
| 42,000 | 32.8243 | 32.9626 | +0.1382 |
| 48,000 | 32.8847 | 32.9895 | +0.1048 |
| 54,000 | **32.8898** | **33.0759** | +0.1861 |
| 60,000 | 32.8697 | 33.0676 | +0.1979 |

Over the **final 8,000 iterations** the GT-only curve spans **0.0201 dB** and the
KD curve **0.0259 dB** — both flat. Both were within 0.25 dB of their eventual
best by iteration 36,000. Neither is still improving materially at 60,000, so the
schedule is long enough for the comparison being made.

Both peak at 54,000 and dip slightly at 60,000, which is ordinary
end-of-cosine-schedule behaviour, not a divergence.

---

## Step 3 & 5 — the gap, and what distillation did to it

All three models are evaluated on **byte-identical inputs**: one materialised
`PairedTestDataset` replayed per model, so loading, cropping and ordering cannot
differ between them. Metrics go through the locked harness
(`src/eval/evaluate.py`, `ADAIR_DEFAULT`); `scripts/task_gap.py --task dehaze` computes none
of its own.

```
AdaIR (teacher)        psnr 34.5056  ssim 0.9878  n=150
M GT-only              psnr 32.8898  ssim 0.9821  n=150
M GT+KD                psnr 33.0759  ssim 0.9828  n=150

gap  teacher - M GT-only          +1.6159 dB  +0.0057 ssim
gap  teacher - M GT+KD            +1.4298 dB  +0.0049 ssim
```

**Distillation recovered 0.1861 dB of a 1.6159 dB gap — 11.5% of it — for zero
inference cost.** The student's architecture, parameter count, export graph and
INT8 latency are all unchanged; the entire difference is in the weights.

![hazy / teacher / GT-only / GT+KD / ground truth](dehaze_gap_strip.png)

Visually the three restorations are close, which is what a 1.6 dB spread at
33–34 dB looks like. The hazy inputs are recovered convincingly by all three.

### Honest reading

**What this supports.** Response distillation transfers *some* of the teacher's
advantage without the student computing any frequencies. The direction is
consistent from iteration 12,000 onward — KD leads at every validation point
after 6,000 — which is what makes the result worth reporting from a single seed.

**What this does not support.** That +0.1861 dB exceeds seed variance, because
seed variance was not measured for this configuration. B0-denoise's three seeds
spanned 0.0079 dB on a different task and a 300k schedule; if dehazing behaved
similarly the delta would be ~24x that spread, but transferring a variance
estimate across task and schedule is not evidence, and it is not claimed here.
The clean way to settle it is two more seeds per arm.

**Also not supported:** that response KD is the *best* method. The weight was not
tuned and no other KD variant was tried. This shows the simplest possible
approach produces a measurable gain, which is the useful thing to know before
building Phase 02's grid.

---

## Reproduction

```bash
python scripts/make_dehaze_split.py --seed 1234 --train-pairs 4000 --heldout-images 150
python scripts/stress_test_norm.py --dehaze --geometry "M(w16_sidd)" --weights <b0v2>/best.pth
python scripts/clamp_by_task.py <b0v2>/best.pth --batches 200

./scripts/run_b0_devon.sh --arm M-DEHAZE    --seed 0 --out-root runs/demo_dehaze    --num-workers 8
./scripts/run_b0_devon.sh --arm M-DEHAZE-KD --seed 0 --out-root runs/demo_dehaze_kd --num-workers 8

python scripts/task_gap.py --task dehaze \
  --student runs/demo_dehaze/M-DEHAZE/*/best.pth       --label "M GT-only" \
  --student runs/demo_dehaze_kd/M-DEHAZE-KD/*/best.pth --label "M GT+KD"
```

Machine: devon, RTX 4090, `taskset -c 0-7,12-31`. Teacher checkpoint
`adair-single-dehaze.ckpt`, 28,784,824 params.

---

## Defects found while building this

Recorded because two of them would have produced a confident wrong answer.

1. **The YAML merge dropped `eval` and `distill`.** `_apply_yaml_overrides`
   merged five sections and neither was among them. The dehaze run validated on
   **BSD68 denoising** for 8,000 iterations, and — far worse — `M-DEHAZE-KD`
   would have **loaded no teacher and trained a plain GT-only run**. Step 5
   would then have compared a baseline against itself and reported "distillation
   changed nothing". Nothing would have errored.

   Caught because two instruments disagreed on the same checkpoint: the training
   log said 18.75 dB and `task_gap.py --task dehaze` said 30.63 dB. The tell was
   `psnr_dehaze: null` beside a non-null `psnr`.

   The B3 dead-config-key check could not catch this — it greps for key *names*,
   and both names are referenced downstream. The new test asserts the **values
   arrive**: every key of every reviewed YAML section must equal what
   `build_config` resolves.

2. **The F10 gate scored a destroyed image as perfect.** `psnr()` takes
   `data_range = 1.0` with `clip = True`; passing `uint8` clipped everything to
   1.0, made both images identical and returned `inf` — which clears a
   `>= 25 dB` threshold. Now non-finite PSNR raises rather than grading. Written
   up as its own section of finding F12.

3. **The teacher loaded before the logger existed**, and **AdaIR cannot run in
   bfloat16**. Both caught by a 40-iteration smoke test rather than by a
   multi-hour run failing at startup.

4. **AdaIR was not vendored on devon at all** — caught by running the gap
   pipeline early on a partial checkpoint instead of at the end.

The pattern in 1 and 2 is the same one F12 records: **a validation instrument
that silently agrees with whatever it is shown**, in the specific case where
disagreement matters most.
