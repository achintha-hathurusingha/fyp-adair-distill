# AdaIR evaluation conventions — traced from source

**Task 2.1 deliverable.** Every entry below is read from AdaIR's code, not from
the paper and not inferred. File and line references are given so each claim can
be checked.

**Source pinned:** `github.com/c-yn/AdaIR` @ **`ccb8b98e49614e07badd0641e5163fa7635c2f02`**
(2025-12-24, "Update README"), shallow clone retrieved 2026-07-30.
**Licence: MIT**, © 2024 Yuning Cui — permissive; redistribution and publication
are unrestricted subject to notice retention.

---

## Headline: there is exactly ONE metric function, used identically by all three tasks

`utils/val_utils.py:50-64`

```python
def compute_psnr_ssim(recoverd, clean):
    assert recoverd.shape == clean.shape
    recoverd = np.clip(recoverd.detach().cpu().numpy(), 0, 1)
    clean    = np.clip(clean.detach().cpu().numpy(), 0, 1)
    recoverd = recoverd.transpose(0, 2, 3, 1)
    clean    = clean.transpose(0, 2, 3, 1)
    for i in range(recoverd.shape[0]):
        psnr += peak_signal_noise_ratio(clean[i], recoverd[i], data_range=1)
        ssim += structural_similarity(clean[i], recoverd[i], data_range=1, multichannel=True)
    return psnr / N, ssim / N, N
```

Called from `test.py:64` (denoise) and `test.py:89` (derain **and** dehaze).

**The hypothesis that deraining uses Y-channel evaluation is FALSE for AdaIR.**
The PReNet/MPRNet lineage does commonly evaluate deraining on Y, but AdaIR does
not: all three tasks go through this one RGB function. This was worth testing —
had we assumed the lineage convention we would have been wrong on deraining by
roughly 1–2 dB — but the code is unambiguous.

## Convention table (identical for denoise / derain / dehaze)

| Question | AdaIR's answer | Source |
|---|---|---|
| Channel basis | **RGB, all 3 channels** — no Y/luma conversion anywhere | `val_utils.py:62` |
| Border crop | **None (0 px)** before metrics | `val_utils.py:50-64` |
| Value range | **`[0, 1]` float**, `data_range=1` | `val_utils.py:61-62` |
| Clipping | `np.clip(·, 0, 1)` on **both** prediction and GT | `val_utils.py:52-53` |
| uint8 rounding | **No.** Compared as float32; never quantised | `val_utils.py:50-64` |
| Save-then-load | **No.** Metrics computed in memory *before* the PNG write | `test.py:64` then `:68`; `:89` then `:93` |
| PSNR impl | `skimage.metrics.peak_signal_noise_ratio` | `val_utils.py:4,61` |
| SSIM impl | `skimage.metrics.structural_similarity`, `multichannel=True` | `val_utils.py:4,62` |
| SSIM window | **skimage default `win_size=7`, uniform (box) filter** — `gaussian_weights` is **not** set | skimage defaults |
| Averaging | Per-image, then mean over the set (`AverageMeter`) | `val_utils.py:8-26` |

> **SSIM caution.** AdaIR uses skimage's *defaults*: a **7×7 uniform window**,
> not the classic Wang et al. 11×11 Gaussian (σ=1.5). Any config defaulting to
> `ssim_window: 11, ssim_sigma: 1.5` would **not** reproduce AdaIR. The
> reproducing setting is `win_size=7, gaussian_weights=False`.

## Image loading (all tasks)

| Step | Detail | Source |
|---|---|---|
| Reader | `PIL.Image.open(...).convert('RGB')` → **RGB order**, not BGR | `dataset_utils.py:252, 348, 351` |
| Geometry | `crop_img(img, base=16)` — **centre-crop to a multiple of 16**, applied to degraded *and* clean | `dataset_utils.py:252, 348, 351`; `image_utils.py:59-64` |
| Tensor | `torchvision.transforms.ToTensor()` → HWC→CHW, divide by 255 → `[0,1]` | `dataset_utils.py:235, 256` |

`crop_img` (`image_utils.py:59-64`) crops, it does **not** pad:

```python
crop_h = h % base; crop_w = w % base
return image[crop_h//2 : h-crop_h+crop_h//2, crop_w//2 : w-crop_w+crop_w//2, :]
```

For BSD68 at 481×321 this yields **480×320** — metrics are computed on the
cropped region, so a harness evaluating at full resolution will not match.

## Denoising specifics

`dataset_utils.py:243-246`

```python
noise = np.random.randn(*clean_patch.shape)
noisy_patch = np.clip(clean_patch + noise * self.sigma, 0, 255).astype(np.uint8)
```

- Noise is added in **`[0, 255]` space**, σ ∈ {15, 25, 50} on that scale, **after**
  the multiple-of-16 crop.
- The noisy input is **cast to uint8** — i.e. quantised. Synthesising noise in
  `[0,1]` float space instead would give a slightly different (and easier) input.
- Seeded once globally: `np.random.seed(0)`, `torch.manual_seed(0)` at
  `test.py:114-115`. Noise therefore depends on **dataset iteration order**, which
  comes from `os.listdir` — not sorted (`dataset_utils.py:238`). This is a
  reproducibility hazard: `os.listdir` order is filesystem-dependent.

## Ground-truth pairing (derain / dehaze)

`dataset_utils.py:325-338`

- **Derain:** `degraded_path.replace("input", "target")`
- **Dehaze:** directory `…/target/`, filename `name.split('_')[0] + '.png'` — i.e.
  SOTS `0001_0.8_0.2.png` → `0001.png`

---

## Discrepancies and risks found

### R1. Urban100 — RESOLVED: not part of the 3-degradation protocol

`test.py:120` sets `denoise_splits = ["bsd68/"]`. Checking this against the
paper's own results table (`figs/adair3d.PNG`) rather than against the script:

| Method | Dehazing SOTS | Deraining Rain100L | Denoising **on BSD68** σ=15 / 25 / 50 | Average |
|---|---|---|---|---|
| **AdaIR (Ours)** | **31.06 / 0.980** | **38.64 / 0.983** | **34.12 / 0.935** · **31.45 / 0.892** · **28.19 / 0.802** | **32.69 / 0.918** |

The table header reads "Denoising on BSD68". **Urban100 does not appear in the
3-degradation table at all**, so the released script and the published claim
agree — there is no partial-script discrepancy. Urban100 appears in `INSTALL.md`
only as a download (it is used in AdaIR's *single-task* denoising experiments,
which are a different setting).

**Resolution: BSD68 is the protocol.** Urban100 is out of scope, and dropped
from our protocol table. This also removes the need for tiled teacher inference
— Urban100's ~1024² images were the only thing that would have forced it.

**These are the G2/G3 target numbers**, to be matched within ±0.10 dB.

### R2. AdaIR's evaluation code cannot run on a modern environment (blocks G2)

Two hard failures, both verified in our `adair` env (scikit-image 0.24.0):

1. `val_utils.py:5` — `from skvideo.measure import niqe`. `scikit-video` is
   unmaintained; **not installed and not installable cleanly on NumPy 2.x**. This
   import is at module scope, so it breaks `test.py` before anything runs, even
   though `niqe` is never used by the three tasks we need.
2. `val_utils.py:62` — `multichannel=True` was deprecated in scikit-image 0.19
   and **removed in 0.23**. On 0.24 the kwarg is swallowed by `**kwargs` and
   silently ignored, leaving `channel_axis=None`, which treats `(H,W,3)` as a 3-D
   volume. In our test it raised (`win_size exceeds image extent`) — but that is
   luck, not safety: on other shapes it would return a **silently different
   number**.

   > Note on what that raise does *not* prove: under `channel_axis=None` a CHW
   > array `(3, H, W)` also has an axis below `win_size=7` and raises
   > identically. The error is therefore consistent with **either** layout and
   > does not discriminate between them. The layout is settled by the transpose
   > below, not by the error.

### Array layout at the metric call site — HWC (settled)

`compute_psnr_ssim` transposes **NCHW → NHWC** at `val_utils.py:55-56`, two
lines before the metric loop:

```python
recoverd = recoverd.transpose(0, 2, 3, 1)
clean    = clean.transpose(0, 2, 3, 1)
```

Verified empirically on a BSD68-shaped batch: `(1,3,320,480) → (1,320,480,3)`,
so the slice passed to `structural_similarity` is `(320, 480, 3)` — **HWC, last
axis = channels**. Under scikit-image 0.19 `multichannel=True` therefore means
per-channel 2-D SSIM averaged over 3 channels, which is exactly what
`channel_axis=-1` does today. Our harness matches; no change required.

AdaIR pins `scikit-image==0.19.3` and `scikit-video==1.1.11` (`env.yaml:221-222`),
Python 3.8.11, PyTorch 1.13.1 (`INSTALL.md`). **G2 therefore requires a separate
legacy environment**, not our project env. This keeps "run their code unmodified"
literally true.

For our harness (G3), the modern equivalent of `multichannel=True` is
**`channel_axis=-1`**; equivalence to 0.19.3 behaviour must be verified
numerically, not assumed.

### R3. `os.listdir` ordering affects denoise noise realisations

Because noise is drawn from a single global seed in file-iteration order, and
`_init_clean_ids` does not sort (`dataset_utils.py:238`), two machines can produce
different noisy inputs from the same seed. Expect small (<0.05 dB) but non-zero
irreproducibility on denoise. Our harness should **sort** file lists and document
the deviation.

### R4. Test-time inputs are uint8-quantised, training-time may not be

Worth confirming against the training path before Task 2.4, so our synthetic
degradation matches on both sides.

---

## Recommended configuration (defaults for `metrics.py`)

Derived from the above; **identical for all three tasks** — no per-task override
is needed, contrary to expectation:

```yaml
metrics:
  channel: rgb          # NOT y, for any task
  crop_border: 0        # no border crop
  round_to_uint8: false # float comparison
  clip: [0.0, 1.0]      # applied to both pred and GT
  data_range: 1.0
  ssim_impl: skimage
  ssim_window: 7          # skimage default, NOT 11
  ssim_gaussian_weights: false   # uniform window, NOT Wang 11x11 Gaussian
  ssim_channel_axis: -1          # modern equivalent of multichannel=True
image_loading:
  reader: pil_rgb
  crop_to_multiple: 16   # centre crop, applied to pred and GT alike
  scale: 255.0           # ToTensor
```

The per-task override mechanism should still be **built** (cheap, and needed if
we later compare against Y-channel-reporting baselines such as MPRNet), but its
default is a single shared convention.

---

## Decisions taken

1. **Urban100 (R1) — RESOLVED empirically.** The paper reports denoising on
   BSD68 only; Urban100 is not in the 3-degradation table. BSD68 is the
   protocol, Urban100 is out of scope, tiled inference is not needed.
2. **G2 legacy environment — BUILD IT.** A pinned Python 3.8.11 / torch 1.13.1 /
   scikit-image 0.19.3 / scikit-video 1.1.11 environment, so AdaIR's artifact
   runs genuinely unmodified. G2 exists to test whether the *released artifact*
   reproduces the *published claim*; patching the artifact would validate
   something else. Single-use, archived after G2, nothing downstream depends on
   it. Timeboxed to two days — the likely failure is CUDA/driver incompatibility
   with torch 1.13.1. On fallback, every patch is diffed and recorded here.
3. **Noise seeding (R3) — filename-derived, not sorted.**
   `seed = int(sha256(filename)[:8], 16)` per image. Sorting fixes only
   cross-machine ordering and breaks the moment a file is added; a filename hash
   is order-independent *and* stable across filesystems and re-downloads.
   Noisy test inputs are then generated **once** and frozen as uint8 PNGs, so
   noise realisation stops being an experimental variable at all.
   **For G2 we accept AdaIR's unsorted global-seed behaviour** — we are
   validating their code, not improving it. G3 reports both numbers and the
   delta; anything under 0.05 dB sits inside the ±0.10 dB tolerance and is
   documented as a known, quantified difference.
