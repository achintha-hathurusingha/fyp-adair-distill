# Student v2: block-wise theory review and two NPU-safe additions

Triggered by: feature-KD (kd_feature_multitask) showed no real benefit over
GT-only training (tied on denoise, behind on derain/dehaze --
[cond_regression.md](../kd_feature_multitask/cond_regression.md)), and the
FiLM-conditioned variant regressed further. Rather than copy the teacher's
own blocks verbatim, this reviews *why* AdaIR's blocks are shaped the way
they are, goes to the older theory those choices rest on (or don't), and
proposes additions grounded in that theory -- filtered through one hard
constraint: it must run on a mobile NPU.

## 1. AdaIR, block by block -- what the paper itself claims

Source: [arXiv:2403.14614](https://arxiv.org/abs/2403.14614) (Cui, Zamir,
Khan, Knoll, Shah, Khan, CVPR 2024) and `third_party/AdaIR/net/model.py`.

| Block | What it does | Paper's own justification |
|---|---|---|
| Patch embed | 3x3 conv, 3->48ch | none needed, standard |
| TransformerBlock (MDTA + GDFN) | channel-attention + gated feedforward, 4-level encoder/decoder | **inherited from Restormer, no justification offered.** Quote: "we adapt the multi-dconv head transposed cross attention ... to mine different feature parts" -- stated, not argued |
| Concat + 1x1-reduce skips | encoder/decoder skip connections | standard U-Net-with-transformers pattern, not discussed |
| FreModule (FMiM + FMoM) | FFT-splits the *input image* into high/low bands, cross-attends each against the decoder feature, gates and re-fuses | **empirically motivated**: Fig. 1 shows noise/rain concentrate energy in high frequencies, haze/low-light in low frequencies. No signal-processing theory cited for *why* a global FFT split is the right tool -- used as a diagnostic, not derived |

**Ablation (Table 7, dehazing, 20 epochs)** -- this is real, load-bearing
evidence, not theory:

| Config | PSNR | Δ |
|---|---|---|
| Baseline (no FreModule) | 28.21 | -- |
| + Frequency Mining (FMiM) | 29.79 | **+1.58 dB** |
| + L→H unit | 30.37 | +0.58 dB |
| + H→L unit | 30.52 | +0.15 dB |
| Full AdaIR | 31.24 | +0.72 dB (cross-attention refinement) |

Frequency-selective processing is worth +1.58dB **by itself**, before any
attention-based refinement -- that number is the actual target to chase, not
AdaIR's specific mechanism for getting it.

**Candid limitation, Table 14**: adding dehaze into multi-task training
degrades performance *more* than adding rain or noise does -- the authors'
own multi-degradation model has the same weak spot our student does. This
matters for calibrating expectations: some of dehaze's gap may be an
intrinsic multi-task tension the teacher only partially resolves, not purely
an architecture gap our student can close.

## 2. Theoretical foundations, oldest / strongest first

None of these are AdaIR's own citations -- they're the underlying theory
AdaIR's empirical design happens to sit on top of, found by going one level
below the paper.

1. **Mallat, "A theory for multiresolution signal decomposition: the
   wavelet representation," IEEE TPAMI 11(7), 1989.**
   [ADS link](https://ui.adsabs.harvard.edu/abs/1989ITPAM..11..674M) ·
   [Wikipedia summary](https://en.wikipedia.org/wiki/Multiresolution_analysis)
   The formal Fourier/wavelet trade-off: a transform can have perfect
   frequency localization (Fourier) or spatial localization (wavelets), not
   both -- wavelet bases are explicitly constructed to be "local in both
   time and frequency." **Relevance**: AdaIR's FreModule computes ONE
   adaptive threshold for the whole image (via global average pooling), so
   it is a global-Fourier design and, by this theorem, structurally cannot
   represent degradation whose intensity varies spatially (patchy haze,
   localized rain streaks) -- a real, provable limitation, not a guess.

2. **Burt & Adelson, "The Laplacian Pyramid as a Compact Image Code," IEEE
   Trans. Communications 31(4), 1983.**
   The classical multiresolution decomposition on the *wavelet* side of
   Mallat's trade-off: blur+decimate to build a Gaussian pyramid, subtract
   consecutive levels for band-pass (Laplacian) residuals. Spatially
   localized by construction -- each band keeps full spatial extent, unlike
   a single global mask. Older and simpler than a full wavelet transform,
   and built entirely from convolution + pooling.

3. **Koschmieder, 1924 (atmospheric optics); McCartney, "Optics of the
   Atmosphere," 1976.** Physical law of light transport through a
   scattering medium: `I(x) = J(x)t(x) + A(1-t(x))`. Haze formation has a
   *known closed form* -- not something that has to be learned from data at
   all.

4. **He, Sun & Tang, "Single Image Haze Removal Using Dark Channel Prior,"
   CVPR 2009 / IEEE TPAMI 33(12), 2011.**
   [PubMed-indexed follow-on work](https://pubmed.ncbi.nlm.nih.gov/39617546/)
   Empirical statistical prior (near-zero minimum color-channel value in
   local patches of haze-free outdoor images) that turns Koschmieder's model
   into a computable, zero-learned-parameter, **per-pixel** (spatially
   varying) transmission estimate `t(x)`. Directly fixes the same
   "global vs. local" gap identified in item 1, specifically for haze.

5. **Chi, Jiang & Mu, "Fast Fourier Convolution," NeurIPS 2020.**
   [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2020/file/2fd5d41ec6cfab47e32164d5624269b1-Paper.pdf)
   States the actual theorem AdaIR never cites: the spectral convolution
   theorem means a point-wise update in the Fourier domain affects the
   *entire* spatial extent of the input -- i.e. global receptive field from
   a 1x1 conv, for free, once you're in frequency space. This is the real
   theoretical justification for "FFT helps," which AdaIR's own paper never
   states (see section 1) -- but it is also exactly the operation
   (`torch.fft`) that turns out to be undeployable (section 5).

6. **Dauphin, Fan, Auli & Grangier, "Language Modeling with Gated
   Convolutional Networks," ICML 2017 / [arXiv:1612.08083](https://arxiv.org/abs/1612.08083).**
   Formal gradient argument for GLU-style gating: a gated linear unit keeps
   an *unscaled* gradient path through the linear branch, unlike stacked
   tanh/sigmoid nonlinearities whose gradients shrink at every layer. This
   is the real theoretical grounding for both NAFNet's SimpleGate *and*
   AdaIR's GDFN -- they're the same theory, different plumbing. **Already
   present in our current student** -- not a gap to close.

7. **Hu, Shen & Sun, "Squeeze-and-Excitation Networks," CVPR 2018.**
   Channel-wise recalibration from a single global-average-pooled
   descriptor. This is the theoretical grounding AdaIR's own MDTA lacks
   (the paper never argues for channel- over spatial-attention) -- and
   NAFNet's existing SCA block (`nn.AdaptiveAvgPool2d(1)` + 1x1 conv) is
   already a minimal instance of exactly this theory. **Already present in
   our current student** -- also not a gap.

Net reading of 6 and 7: two of AdaIR's four real design choices (gating,
channel-over-spatial attention) are *already* well-grounded theoretically,
and the current NAFNet-based student already embodies both. The two
genuinely missing capabilities are frequency-selectivity (item 1/2/5) and a
physical degradation model (item 3/4) -- neither of which the student has
any version of today.

## 3. Is it really the block design? -- a real controlled backbone study

Chen, Chu, Zhang & Sun's NAFNet paper (ECCV 2022) itself was **only ever
evaluated on denoising (SIDD) and deblurring (GoPro)** -- never on
deraining or dehazing. Using it as an all-in-one multi-degradation backbone
was, from the start, an out-of-distribution architectural choice compared
with AdaIR/Restormer/AirNet, which were purpose-built and validated for
exactly that setting.

**"A Comparative Study of Image Restoration Networks for General Backbone
Network Design," [arXiv:2310.11881](https://arxiv.org/abs/2310.11881)**
ran NAFNet, SwinIR and Restormer on identical tasks and found:

| Task | NAFNet | SwinIR | Restormer |
|---|---|---|---|
| Deraining (Test100) | 30.33 | 30.05 | **32.03** |
| Dehazing (SOTS Indoor) | 38.97 | 29.14 | **41.87** |
| Deblurring (GoPro) | **33.08** | 31.66 | 32.92 |

NAFNet loses by 1.7dB (derain) and 2.9dB (dehaze) but *wins* on
deblurring -- and the paper's own explanation is architectural, not just a
training artifact: depthwise convolution has "relatively weak spatial
mapping capability ... compared to spatial self-attention," and
deraining/dehazing specifically need "the ability to handle large-range or
even global information," while denoising is "flexible" (multiple
architecture families do it well). **This matches our own measurement
exactly** -- our student ties the GT-only baseline on denoise and lags on
derain/dehaze (cond_regression.md) -- which upgrades this from "AdaIR-
inspired guess" to "independently reproduced pattern with a published
architectural explanation."

This directly motivated re-opening a question the mobile-NPU section below
had only provisionally deferred (MDTA pending a real probe): global/
large-range context, not just frequency-selectivity or a haze prior, may be
a real, separate gap.

## 4. Follow-up: is attention actually viable, and a mathematically-derived
   alternative if not

**`scripts/probe_mdta.py`** -- ran AdaIR/Restormer's actual MDTA block
(unmodified) through this repo's op-coverage gate. Result: it lowers to
`MatMul`, `Softmax`, and `ReduceL2` (inside the L2-normalize), and **none of
the three appear in ANY of the curated qnn/tflite/tensorrt tables** --
genuinely unverified (real-world Qualcomm NPU deployments of attention
models exist, e.g. on-device LLM inference research, so this is "unproven,"
not "known-bad," unlike FFT in section 5) but a real, uncharacterized risk.

Rather than gamble the deployment target on an unverified op set, two more
additions were built, mathematically matched to what each weak task
actually needs, and NPU-verified the same way as everything above:

**`StripPoolingGate`** (Hu, Zhang, Xie & Yang, "Strip Pooling: Rethinking
Spatial Pooling for Scene Parsing," CVPR 2020) -- pools along one FULL
spatial axis at a time (mean over W keeping H, and vice versa), which is
genuinely global along that axis, at the bottleneck (one instance, cheapest
placement for a global operation). Gets a comparable "large-range
information" capability to attention from AveragePool/mean + Conv only.
First isolated-probe version used `.expand()` on a runtime shape, which
traced to `Equal`/`Where`/`ConstantOfShape`/`Expand` (all UNKNOWN) --
caught and fixed by relying on ONNX `Add`'s native broadcasting instead.
Second bug, caught only once embedded in the FULL model (not the isolated
probe): `F.adaptive_avg_pool2d(y, (x.shape[-2], 1))` traces `x.shape[-2]`
as non-constant once downstream of NAFNet's own dynamic padding, which
ONNX export rejects -- fixed by recognizing that pooling to a target size
equal to one axis' own extent is just `mean(dim=that_axis, keepdim=True)`,
which has no such requirement. Final whole-model op-coverage: identical
UNKNOWN/CAUTION categories to the already-shipped baseline, zero new ones
(`ReduceMean` count goes from 128 to 130 -- more instances of an
*already-present* category, not a new one).

**`OrientedStreakGate`** -- not a named architecture at all, derived
directly from the mathematics of the degradation per the user's direction
to work "from a mathematical perspective." Rain-streak decomposition
literature (Kang, Lin & Lin, "Automatic Single-Image-Based Rain Streak
Removal via Image Decomposition," IEEE TIP 2012; Li, Tan, Guo, Lu & Brown,
"Rain Streak Removal Using Layer Priors," CVPR 2016) models a rainy image
as `I = B + R`, where `R` is a sparse, **directionally-anisotropic**
high-frequency layer -- real streaks have a dominant orientation. Every
conv in NAFBlock is an isotropic (4-fold-symmetric) square kernel --
structurally the wrong shape for a signal whose defining property IS its
orientation. Freeman & Adelson, "The Design and Use of Steerable Filters,"
IEEE TPAMI 13(9), 1991, gives the governing theory: a small basis of
directional filters, linearly combined with learned angle-dependent
weights, synthesizes a response at any orientation. Implemented as 4
fixed-orientation depthwise kernels (0/45/90/135 degrees -- two elongated
rectangular kernels for the axis-aligned angles, two square kernels with
the off-diagonal half masked to zero at init AND held there via a gradient
hook for the diagonal angles), combined via an SE-style channel gate.
Verified: op histogram is exactly `{Conv, GlobalAveragePool, Relu, Sigmoid,
Concat, Mul, Add}` -- **100% SUPPORTED on qnn/tflite/tensorrt, no
exceptions at all** -- the cleanest result of anything built in this
review, since it introduces no op type NAFNet doesn't already use.

Both wired into `NAFNet` as further opt-in flags (`use_strip_pool`,
`use_oriented_streak`), verified together with the section-3 additions in
one whole-model export (`scripts/smoke_nafnet_theory.py`): +206,608 params
total over the 7.37M-param locked student (+2.80%), all 740 parameter
tensors receive finite gradient, zero new UNKNOWN/CAUTION op categories
versus the already-shipped baseline.

## 5. The mobile-NPU constraint (this repo's own methodology)

*(Sections 3-4 above already assumed this section's conclusions -- FFT is
undeployable, MDTA is unverified -- since those findings came first
chronologically. Reading order here follows how the investigation actually
unfolded rather than re-deriving each fact only once, in a purely linear
document.)*

Per `reports/export_smoke_test.md` and `src/export/op_coverage.py` (Gate
G1), the real deployment targets are **QNN** (Qualcomm Hexagon NPU),
**TFLite**, and **TensorRT**, checked via actual ONNX export + a curated
op-support table -- not assumption.

**AdaIR's FreModule cannot be exported to any of them.** It calls
`torch.fft.fft2`/`ifft2` on complex tensors. ONNX has a `DFT` operator
(opset 17+), but web research on real Hexagon NPU deployments turned up no
evidence of DFT/FFT in QNN's, TFLite's, or any mainstream mobile NPU
delegate's shipped op set -- it appears in ongoing FFT-on-NPU *research*
papers, not production support. This is confirmed by the general finding
that "[mobile NPU support is limited ... resulting in a majority of model
inference being executed on the CPU](https://arxiv.org/pdf/2005.05085)"
when an op falls outside the delegate's coverage. So even setting aside
whether copying AdaIR's blocks would help feature-KD alignment, the literal
teacher architecture is not a deployment candidate.

This ruled out the first draft of this work (a from-scratch "MiniAdaIR"
mirroring AdaIR's TransformerBlocks + FreModule 1:1) before it was finished
-- MDTA/GDFN would also need their own real op-coverage probe (MatMul +
Softmax + L2-normalize are not in this repo's curated QNN table at all,
though real-world Qualcomm NPU deployments of attention models do exist per
[on-device LLM inference research](https://arxiv.org/html/2407.05858v2), so
that is "unverified," not "known-bad," unlike FFT). Given the actual
measured problem (feature-KD showing no benefit) was never conclusively an
architecture-mismatch issue to begin with, committing to an unverified,
higher-risk transformer backbone wasn't justified. The two additions below
instead build on top of the current, already NPU-validated NAFNet backbone.

### What was built, and its real (not assumed) op-coverage result

Both live in `src/models/theory_blocks.py`, wired into `NAFNet` as two new
opt-in flags (`use_freq_gate`, `use_dcp_prior`) -- following the same
pattern as `use_degradation_head`, non-invasive to the LOCKED default
architecture (`configs/model/nafnet_locked.yaml`).

**`LaplacianFrequencyGate`** -- a Burt & Adelson-style 3-band pyramid
decomposition of a decoder-stage feature map, each band re-weighted by its
own SE-style channel gate (GAP -> 1x1 conv -> ReLU -> 1x1 conv -> sigmoid),
reconstructed and added back with a zero-initialized final projection
(identity at init, same stabilization idiom as AdaIR's own `para1`/`para2`
and this repo's clamp-engagement philosophy). One instance per decoder
stage. Built entirely from Conv / AvgPool / PixelShuffle-upsample (not
`F.interpolate` -- "Resize" appears in none of the three curated backend
tables, whereas PixelShuffle -> `DepthToSpace` is already characterized
there from NAFNet's own up/downsampling). +129,584 params total (4 stages),
+1.76% over the 7.37M-param locked student.

**`dark_channel_prior`** -- concatenated as an extra input channel before
`intro`. First implementation used `torch.min(dim=1)` + the standard
`-MaxPool(-x)` min-pool trick; **real ONNX export showed this lowers to
`ReduceMin` + `Neg`, and neither appears in any of the three curated backend
tables** -- caught by the same export-gate methodology this project already
uses for everything else, not assumed either way. Rewritten to use only
proven-safe ops: pairwise channel-min via `min(a,b) = a - relu(a-b)` (Split
+ Sub + Relu), and the windowed min via `1 - MaxPool(1-x)` (valid because
the signal is in [0,1] by construction), which avoids `Neg` entirely.
Re-verified: op histogram is exactly `{Split, Sub, Relu, MaxPool, Constant}`
-- all SUPPORTED on qnn/tflite/tensorrt except `Constant`, which is
compile-time constant-folding (already a known non-issue per
`export_smoke_test.md` §5).

**Whole-model check** (`scripts/smoke_nafnet_theory.py`, real run on
devon): exporting the full NAFNet with both additions enabled produces
*exactly* the same UNKNOWN/CAUTION op categories the plain locked model
already has in `export_smoke_test.md` (Cast/Constant/ConstantOfShape/
Gather/Identity/Mod/Shape/Unsqueeze as UNKNOWN on qnn; DepthToSpace/Div/Pow/
ReduceMean/Slice/Sqrt as CAUTION) -- zero new risk categories. The two new
op types introduced (`MaxPool`, `Split`) are SUPPORTED on all three
backends. Forward/backward pass verified: all 704 parameter tensors receive
finite gradient; `LaplacianFrequencyGate` confirmed byte-identical to the
un-modified model when loaded onto the same weights (max diff `0.00e+00`).

## 6. What this does NOT establish

- **No training run yet.** All four additions (`LaplacianFrequencyGate`,
  `dark_channel_prior`, `StripPoolingGate`, `OrientedStreakGate`) are
  architecture- and export-verified, not accuracy-verified -- whether any
  of them actually close the denoise/derain/dehaze gap is untested. The
  natural next step is single-task ceiling probes (train on denoise-only /
  derain-only / dehaze-only, no multi-task mixing, short matched budget) to
  separate "does the current architecture have a real per-task capacity/
  receptive-field ceiling" from "is the gap multi-task interference" --
  before spending a full 90k-iteration multi-task run on any one variant.
- MDTA/GDFN were probed (not just theorized about) and found genuinely
  unverified rather than known-bad -- `MatMul`/`Softmax`/`ReduceL2` are
  simply absent from this repo's curated tables for all three backends,
  which plausibly reflects the table having never needed to characterize
  attention ops before (this codebase had none until this probe), not a
  real Qualcomm/TFLite limitation. Updating the curated table itself from
  primary QNN/TFLite documentation, or an actual on-device convert test,
  would resolve this properly and is worth doing if the strip-pool/oriented
  additions don't close enough of the gap on their own.
- Table 14 of the AdaIR paper (section 1) shows dehaze hurts even the
  teacher's own multi-task training more than other degradations -- some of
  the current dehaze gap may not be closeable by architecture changes
  alone, regardless of which of the above turns out to help.

## Full reference list

1. Cui, Zamir, Khan, Knoll, Shah, Khan. "AdaIR: Adaptive All-in-One Image
   Restoration via Frequency Mining and Modulation." CVPR 2024.
   https://arxiv.org/abs/2403.14614
2. Mallat. "A Theory for Multiresolution Signal Decomposition: The Wavelet
   Representation." IEEE TPAMI 11(7), 1989.
3. Burt & Adelson. "The Laplacian Pyramid as a Compact Image Code." IEEE
   Trans. Communications 31(4), 1983.
4. Koschmieder. "Theorie der horizontalen Sichtweite." 1924 (classical
   atmospheric optics, as cited by the dehazing literature below).
5. McCartney. "Optics of the Atmosphere: Scattering by Molecules and
   Particles." Wiley, 1976.
6. He, Sun & Tang. "Single Image Haze Removal Using Dark Channel Prior."
   CVPR 2009; IEEE TPAMI 33(12), 2011.
7. Chi, Jiang & Mu. "Fast Fourier Convolution." NeurIPS 2020.
   https://proceedings.neurips.cc/paper_files/paper/2020/file/2fd5d41ec6cfab47e32164d5624269b1-Paper.pdf
8. Dauphin, Fan, Auli & Grangier. "Language Modeling with Gated
   Convolutional Networks." ICML 2017. https://arxiv.org/abs/1612.08083
9. Hu, Shen & Sun. "Squeeze-and-Excitation Networks." CVPR 2018.
10. Zamir et al. "Restormer: Efficient Transformer for High-Resolution
    Image Restoration." CVPR 2022 (origin of MDTA/GDFN, which AdaIR
    inherits without independent justification -- see section 1).
11. Chen, Liu, Chen & Fu. "Simple Baselines for Image Restoration." ECCV
    2022 (NAFNet -- the current student's backbone, this project's own
    `src/models/nafnet.py`).
12. "A Comparative Study of Image Restoration Networks for General Backbone
    Network Design." https://arxiv.org/abs/2310.11881 (independent
    controlled comparison of NAFNet/SwinIR/Restormer on identical tasks --
    section 3's central evidence).
13. Kang, Lin & Lin. "Automatic Single-Image-Based Rain Streak Removal via
    Image Decomposition." IEEE Trans. Image Processing, 2012.
14. Li, Tan, Guo, Lu & Brown. "Rain Streak Removal Using Layer Priors."
    CVPR 2016.
15. Freeman & Adelson. "The Design and Use of Steerable Filters." IEEE
    TPAMI 13(9), 1991.
16. Hu, Zhang, Xie & Yang. "Strip Pooling: Rethinking Spatial Pooling for
    Scene Parsing." CVPR 2020.

Op-coverage methodology and curated support tables:
`reports/export_smoke_test.md`, `src/export/op_coverage.py` (this repo,
Gate G1). Real op-coverage probes run this session: `scripts/
probe_mdta.py`, `scripts/probe_strip_pool.py`, `scripts/
probe_oriented_filter.py`, `scripts/smoke_theory_blocks.py`, `scripts/
smoke_nafnet_theory.py`.
