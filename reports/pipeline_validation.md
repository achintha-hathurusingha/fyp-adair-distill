# Pipeline validation — Gates G2 and G3

**Verdict: G3 PASSES.** Our harness reproduces AdaIR's published 3-degradation
table to within **±0.01 dB** on every task, against a ±0.10 dB tolerance.

The evaluation conventions are now **locked** (`src/eval/metrics.py`,
`ADAIR_DEFAULT`) and must not be changed for the remainder of the project.

---

## Environment

| | |
|---|---|
| checkpoint | `adair3d.pth` (stripped from `adair3d.ckpt`), 28,784,824 params, epoch 149, step 650,700 |
| architecture | `AdaIR(decoder=True)` from `third_party/AdaIR` @ `ccb8b98`, MIT licence |
| load | zero missing, zero unexpected keys after stripping the `net.` prefix |
| harness | `src/eval/evaluate.py` — per-image metrics then mean, matching AdaIR's `AverageMeter` |
| device | RTX 3050 6GB, torch 2.5.1+cu121, ~1.2 s/image |
| noise seeding | filename-derived (`sha256(name)[:8]`), **not** AdaIR's global stream |
| date | 2026-07-31 |

---

## G3 — reproduction through *our* harness

| test set | n | our PSNR | published | **ΔPSNR** | our SSIM | published | ΔSSIM |
|---|---|---|---|---|---|---|---|
| denoise BSD68 σ=15 | 68 | 34.12 | 34.12 | **−0.00** | 0.9352 | 0.935 | +0.0002 |
| denoise BSD68 σ=25 | 68 | 31.46 | 31.45 | **+0.01** | 0.8919 | 0.892 | −0.0001 |
| denoise BSD68 σ=50 | 68 | 28.19 | 28.19 | **−0.00** | 0.8020 | 0.802 | −0.0000 |
| derain Rain100L | 100 | 38.64 | 38.64 | **+0.00** | 0.9830 | 0.983 | +0.0000 |
| dehaze SOTS-outdoor | 500 | 31.06 | 31.06 | **+0.00** | 0.9802 | 0.980 | +0.0002 |

804 images total. Maximum deviation **0.01 dB PSNR**, **0.0002 SSIM**.

### What this jointly validates

An exact match across three degradation types simultaneously constrains every
component of the pipeline. Any one of these being wrong would have shown up:

* the **conventions** traced in `reports/eval_conventions.md` (RGB not Y, no
  border crop, `data_range=1`, clipped-not-rounded, skimage SSIM at
  `win_size=7` uniform, `channel_axis=-1`)
* the **crop geometry** — `crop_img(base=16)` centre-crop with its asymmetric
  offset, applied to degraded and clean alike
* the **checkpoint load** — a partial load would not produce the published number
* the **pairing rules** — derain by identical basename, dehaze by
  `name.split('_')[0]` with 500 hazy images over 492 scenes
* the **noise synthesis** — 255-space, uint8-quantised, σ on the 0–255 scale

### Answering R3: filename seeding costs nothing

`reports/eval_conventions.md` R3 flagged that AdaIR draws denoise noise from a
single global seed consumed in unsorted `os.listdir` order, making its
realisation filesystem-dependent, and that our filename-derived seeding would
deviate.

**Measured deviation: ≤0.01 dB** across all three sigmas — the largest denoise
delta is +0.01 dB at σ=25 and the other two are 0.00. Well inside the ±0.10 dB
tolerance and comparable to rounding in the published table's 2-decimal PSNR.

We therefore keep filename-derived seeding, which is order-independent and
reproducible across machines, at no measurable cost in comparability. Noise is
generated deterministically per image rather than cached, since determinism
makes caching unnecessary.

---

## Metric correctness, validated independently

Before G3, `src/eval/metrics.py` was validated directly against AdaIR's own
`compute_psnr_ssim` — not against the aggregate published numbers.

`scripts/make_golden_metrics.py` ran their function inside a rebuilt legacy
environment (**Python 3.8.20, scikit-image 0.19.3, scikit-video 1.1.11,
torch 1.13.1**, matching `env.yaml`) over 8 seeded cases spanning four shapes,
including BSD68's post-crop 320×480 geometry and identical-image cases. Results
are committed at `tests/golden/adair_metrics.json`.

`tests/test_golden_metrics.py`: **4/4 pass**, agreement within 1e-4 dB PSNR and
1e-6 SSIM.

This matters for attribution: metric correctness was established *independently*
of datasets, teacher and dataloaders, so G3's success is not a coincidence of
compensating errors, and a G3 failure would not have been ambiguous across four
subsystems.

A detail worth recording: scikit-image 0.19.3 emits
`FutureWarning: multichannel is a deprecated argument` **and honours it**. That
is precisely the behaviour that became a silent no-op in 0.23 (findings F5), so
the oracle captures the released semantics rather than a modern reinterpretation
of them.

---

## G2 — reproduction with *their* code

**Status: SUPERSEDED BY G3.** Not "skipped" — the evidence G2 was designed to
produce has been obtained by a stronger route.

G2 would have shown that *AdaIR's code reproduces AdaIR's numbers*. G3 shows
that *an independently written pipeline reproduces AdaIR's numbers*, which
**subsumes it**: convention tracing, crop geometry, checkpoint loading,
degradation synthesis, pairing rules and metric implementation are all
constrained simultaneously, and an error in any one would have surfaced in at
least one of five conditions. The metric half is additionally validated on its
own by the golden oracle.

Its remaining purpose — localising a G3 failure — is moot, because G3 passed.

Reconstructing the environment surfaced three obstacles, all recorded in
`findings.md` F5:

1. **`lightning==2.0.1` cannot import.** It eagerly loads `lightning.app` →
   `lightning_cloud`, which reads the Windows certificate store at import time
   and dies with `ssl.SSLError: [ASN1: NOT_ENOUGH_DATA]`. Nothing in the
   restoration path needs it — `test.py` uses Lightning only for
   `LightningModule.load_from_checkpoint`. Resolved with `lightning==2.2.5`,
   which is a **deviation from the pin**: the research code is unmodified, one
   dependency is not.
2. **The pinned transitive versions are not self-consistent** under a current
   resolver (`lightning-utilities==0.8.0` conflicts with the `pytorch-lightning`
   it pulls in).
3. **`test.py` hardcodes `.cuda()`** (lines 61, 85, 136) with no CPU fallback, so
   it additionally needs a CUDA build of torch 1.13.1 (~2.3 GB, cu117).

Remaining work if G2 is later required: install `torch==1.13.1+cu117` in
`adair_legacy`, arrange `data/` into AdaIR's expected tree (already done — our
layout is theirs), and run `python test.py --mode 5 --ckpt_name adair3d.ckpt`.
The environment is otherwise ready.

**Judgement:** completing G2 would confirm a result already established to
±0.01 dB by an independent route, at the cost of a large download and further
dependency archaeology. Recorded as deliberately skipped, with the exact steps
to complete it, rather than quietly dropped.

---

## Specialist vs all-in-one — the Option 2 premise

Measured through the same locked harness, all-in-one `adair3d` against each
single-task specialist on its own test set.

| test set | n | all-in-one | specialist | **ΔPSNR** | ΔSSIM |
|---|---|---|---|---|---|
| denoise BSD68 σ=15 | 68 | 34.12 | 34.36 | **+0.240** | +0.0030 |
| denoise BSD68 σ=25 | 68 | 31.46 | 31.72 | **+0.265** | +0.0051 |
| denoise BSD68 σ=50 | 68 | 28.19 | 28.49 | **+0.306** | +0.0105 |
| derain Rain100L | 100 | 38.64 | 38.89 | **+0.254** | +0.0016 |
| dehaze SOTS-outdoor | 500 | 31.06 | 31.80 | **+0.732** | +0.0005 |

**Mean advantage +0.359 dB** — the *marginal* band (0.2–0.5 dB).

### The mean hides the structure, and the structure is the finding

The advantage is **not uniform**. Dehazing is **+0.732 dB** — comfortably above
the 0.5 dB "real headroom" threshold on its own — while denoising and deraining
cluster tightly at **+0.24 to +0.31 dB**. Dehaze is ~2.7× the others.

So "specialists are marginally better" is the wrong summary. The accurate one is
that **one task has real headroom and two do not**, which suggests a targeted
rather than uniform application: if multi-teacher is pursued, the dehaze branch
is where the surplus actually lives.

A plausible reading is that dehazing is the most globally-structured of the
three degradations — haze is a low-frequency, depth-dependent transformation
requiring scene-level inference — so it competes hardest for capacity in a shared
model. That is a hypothesis, not a result, and this table does not test it.

### Independent corroboration of the harness

The measured single-task denoising values (**34.36 / 31.72 / 28.49**) match
AdaIR's *published single-task* figures to two decimals on all three sigmas.
Those figures were obtained from a secondary source rather than the paper PDF
(which exceeded fetch limits), so the corroboration is suggestive rather than
authoritative — but a three-way exact match is unlikely by chance. On that
reading the harness reproduces **both** published tables, not just the
all-in-one one.

### The confound stands

The specialists were **not trained on a common protocol** — epoch counts, step
counts and steps-per-epoch all differ from each other and from the all-in-one
(`reports/checkpoint_audit.md`). Differing steps-per-epoch implies differing
training-set sizes. Every number above therefore mixes *specialisation* with *a
different training run*, and the artifacts alone cannot separate them.

This matters most for dehaze, the one task with real headroom: `single-dehaze`
ran 9,017 steps/epoch against `adair3d`'s 4,338, so it saw roughly twice the
data per epoch. Part of +0.732 dB is plausibly that, not specialisation.

**Cost note:** multi-teacher adds ~zero GPU cost over single-teacher — each
sample carries exactly one degradation type and routes to exactly one
specialist, so it is one teacher forward per sample either way, and outputs
cache identically. The real costs are storage (~1 GB) and routing complexity.

---

## Locked configuration

```yaml
metrics:                       # src/eval/metrics.py :: ADAIR_DEFAULT
  channel: rgb                 # never Y, for any task
  crop_border: 0
  data_range: 1.0
  clip: true                   # both prediction and ground truth
  round_to_uint8: false        # float comparison
  ssim_win_size: 7             # skimage default, NOT Wang's 11
  ssim_gaussian_weights: false # uniform box window
  ssim_channel_axis: -1        # explicit; never left to a default
image_loading:
  reader: pil_rgb
  crop_to_multiple: 16         # centre crop, degraded and clean alike
  scale: 255.0
degradation:
  noise_space: 255             # sigma on the 0-255 scale
  quantise_to_uint8: true
  seed_mode: filename          # sha256(filename)[:8]
```

**Do not change these.** Every number in the project from here on is comparable
only under this configuration.
