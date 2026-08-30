# Frequency-level restoration using spatial mathematics — literature review

**Question.** How do you obtain frequency-domain restoration *behaviour* using
only spatial operators? Not "is frequency useful" (we tested that separately)
but "what is the mathematics that lets a convolution do a spectral job."

**Why this review exists.** Four architecturally-motivated interventions in this
project produced four nulls (feature-KD −0.79/−0.64 dB; FiLM conditioning
regressed; frequency-mask repair ~0.00 dB; degradation-matched operators
−0.08 dB and widening). Before proposing a fifth, this establishes what the
theory actually says is achievable, and connects it to our own measurements.

Every claim below is tagged with whether it is **[theory]**, **[literature]**,
or **[measured here]**.

---

## 1. The exact bridge — and its cost

**[theory] Convolution theorem.** Multiplication in frequency is convolution in
space: `F⁻¹{H·F{x}} = h * x` where `h = F⁻¹{H}`. So *any* linear
frequency-domain filter has an exact spatial equivalent. There is no
approximation at this step.

**The catch:** the equivalent kernel `h` has *full support* (image-sized),
because frequency multiplication is circular convolution. It collapses to a
practical `k×k` conv only if `H` is smooth.

**[measured here]** (`scripts/freq_to_spatial.py`) We computed the optimal
linear restoration filter per degradation (cross-spectral/Wiener estimator,
300 paired crops), inverse-transformed it, and measured truncation cost on 80
held-out crops:

| task | k=3 | k=7 | k=11 | full-filter gain |
|---|---|---|---|---|
| denoise | −0.12 | −0.05 | −0.02 | +5.01 dB |
| dehaze | −0.01 | +0.02 | +0.00 | +0.68 dB |
| derain | −1.62 | −0.24 | −0.13 | +0.87 dB |

**A 7×7–11×11 convolution carries essentially the entire optimal frequency
filter.** The bridge is real and cheap.

**But the linear ceiling is low for our weak tasks**: the *full* filter buys
only +0.68 dB (dehaze) and +0.87 dB (derain). A purely linear
frequency-domain block — however implemented — cannot close our measured
−0.55/−0.76 dB gaps. Any real gain must come from nonlinearity or
content-adaptivity, which the convolution theorem does not cover.

---

## 2. The fundamental limit — why "frequency vs spatial" is a false dichotomy

**[theory] Gabor's uncertainty principle.** Joint space/frequency resolution is
bounded: `Δx·Δu ≥ 1/4π`. Gabor functions achieve equality — they are the
optimal joint-localisation family
([Daugman, JOSA A 2(7), 1985](https://opg.optica.org/josaa/abstract.cfm?uri=josaa-2-7-1160)).

This reframes the whole question:

| representation | frequency localisation | spatial localisation |
|---|---|---|
| raw pixels | none | perfect |
| global FFT | perfect | **none** |
| wavelet / Gabor / pyramid | partial | partial |
| **k×k convolution** | **partial** | **partial** |

**A convolution is already a joint space-frequency operator.** It sits on the
same uncertainty curve as a wavelet — it simply has a *learned* rather than
prescribed frequency response.

**This is the theoretical reason our AdaIR findings came out as they did.** A
global FFT branch is not "adding frequency understanding" to a CNN that lacks
it; it is moving to the *degenerate corner* of the uncertainty curve where
spatial localisation is zero. For spatially-varying degradations (patchy haze,
localised streaks) that is strictly worse. It also predicts our measurement
that a plain CNN beat every spectral descriptor at degradation ID
(97.0% vs 90.7%, `scripts/spatial_converter.py`) — the radial spectrum
discards exactly the spatial and orientation information the conv retains.

---

## 3. Classical spatial realisations (the toolbox)

| method | what it buys | ops |
|---|---|---|
| **Laplacian pyramid** — Burt & Adelson, *IEEE Trans. Comm.* 31(4), 1983 | band-pass decomposition, spatially localised | blur, decimate, subtract |
| **Multiresolution analysis / wavelets** — Mallat, *IEEE TPAMI* 11(7), 1989 | orthonormal scale decomposition, fast pyramidal algorithm | QMF conv + decimate |
| **Steerable filters** — Freeman & Adelson, *IEEE TPAMI* 13(9), 1991 | response at *any* orientation from a small fixed basis | conv + learned angular weights |
| **Steerable pyramid** — Simoncelli & Freeman, *ICIP* 1995 | scale **×** orientation, self-inverting, translation- and rotation-equivariant | conv + decimate |
| **Gabor filters** — Daugman 1985 | optimal joint space-frequency localisation | conv |

**The steerable pyramid is the most complete answer to the question asked.** It
delivers scale *and* orientation, is self-inverting (no information lost), and
is "shiftable" — free of the translation/rotation instability that plagues
orthogonal wavelets. It is built entirely from convolution and decimation.

**[measured here]** This is not merely aesthetic. Our own optimal-filter
analysis found the derain kernel is **visibly anisotropic** (central peak with
vertical structure, derived from data, not assumed) — and separately, that a
*radially averaged* spectrum is structurally blind to orientation, which is why
derain's spectral curve sat on top of the clean curve. Orientation is the axis
that matters for our worst task, and the steerable pyramid is the classical
construction that provides it. Our `OrientedStreakGate` (4 fixed orientations,
learned gate) is a crude special case of it.

---

## 4. Modern deep-learning realisations

**MWCNN** — [Liu et al., CVPRW 2018](https://arxiv.org/abs/1805.07071).
Replaces pooling with **DWT**, and upsampling with **IWT**. Because the DWT is
invertible, *no information is lost in downsampling* — unlike strided conv or
max-pool. Captures frequency *and* location.

> **Actionable for us:** our U-Net downsamples with strided conv, which is
> lossy. A Haar DWT is a fixed 2×2 orthogonal transform — implementable as
> `PixelUnshuffle` + a fixed 1×1 mixing conv, both of which are already
> **NPU-verified** in our export gate (`DepthToSpace` is in our shipped graph).
> This is a rare case of a literature idea that is both principled and free
> under our deployment constraint.

**Octave convolution** — [Chen et al., ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/papers/Chen_Drop_an_Octave_Reducing_Spatial_Redundancy_in_Convolutional_Neural_Networks_ICCV_2019_paper.pdf).
Splits feature maps into high- and low-frequency groups, storing the
low-frequency group at half resolution, with information exchange between
them. Reduces memory/compute *and* enlarges the effective receptive field —
frequency-aware, entirely spatial.

**Dilated convolution — a warning, not a tool.**
[Chen et al., CVPR 2024](https://arxiv.org/abs/2403.05369) analyse its
frequency response: by the Fourier scaling property, dilation `D` scales the
kernel by `D`, which **attenuates high-frequency response**, and "gridding"
artifacts are aliasing — they appear when feature-map frequency content exceeds
the dilated sampling rate.

> **Actionable for us:** dilation is the *wrong* tool for the rain path. Rain is
> the high-frequency, oriented task; dilated convs specifically degrade
> high-frequency response. Rules out an otherwise-tempting cheap way to buy
> receptive field.

**SFNet / FSNet** — [Cui et al., ICLR 2023 / TPAMI](https://github.com/c-yn/SFNet).
**[measured here]** We read the source: it performs "frequency selection" with
**no `torch.fft` anywhere**. A learned content-adaptive low-pass filter, then
`high = input − low`, then a channel gate. From the *same lab as AdaIR* —
i.e. that group's own later work abandoned forward-pass FFT.

**Fast Fourier Convolution** — [Chi et al., NeurIPS 2020](https://proceedings.neurips.cc/paper_files/paper/2020/file/2fd5d41ec6cfab47e32164d5624269b1-Paper.pdf).
The converse direction: uses FFT to get a global receptive field from a 1×1
conv, via the spectral convolution theorem. Theoretically clean, but exactly
the NPU-hostile direction we cannot ship.

---

## 5. Why our hard tasks are hard — spectral bias

**[theory/literature] The F-principle / spectral bias** —
[Rahaman et al., ICML 2019](https://arxiv.org/abs/1806.08734); Xu et al.
(concurrent). Networks fit **low frequencies first**; high-frequency components
are learned more slowly. This is a *learning-dynamics* bias, independent of
architecture.

This gives a mechanistic account of our own per-task pattern:

| task | spectral character | prediction | **[measured here]** |
|---|---|---|---|
| dehaze | low-frequency, smooth | learned readily | improves fast, but capped |
| denoise | broadband, unstructured | net learns to smooth — aligned with the bias | we **tie** the baseline |
| derain | high-frequency, sparse, **oriented** | fights the bias | our **worst** gap (−0.76 dB) |

It also explains the over-smoothing that Charbonnier/L1 is known for: the loss
under-weights high-frequency error, and spectral bias already disfavours it, so
the two compound. **Rain loses twice.**

> **Implication:** the derain gap may be a *training-dynamics* problem, not an
> architecture problem — which would explain why adding a rain-specific
> operator (`OrientedStreakGate`) did not close it (−0.046 dB at 66 k).

---

## 6. Synthesis — what the theory says to do

Ranked by evidential support, not novelty:

1. **Stop adding frequency *architecture*.** A conv already occupies the right
   point on the uncertainty curve (§2); global FFT moves to the degenerate
   corner; AdaIR's own lab dropped it (§4); our measurements agree at every
   level. **This line is closed.**

2. **The remaining frequency lever is the *loss*, not the module.** Spectral
   bias (§5) says high-frequency content is systematically under-learned, and
   L1 under-penalises it. A frequency-domain *loss* costs nothing at inference
   and directly targets that. Note our `kd_freq` track matched the **teacher's**
   spectrum (dead path, plus KD damage); matching **ground truth** is a
   different and untested proposition.

3. **Invertible downsampling (Haar DWT) is free and principled** (§4) — fixes a
   real information loss, uses ops already NPU-verified.

4. **If orientation is pursued, use the steerable pyramid properly** (§3) —
   scale × orientation, self-inverting, rotation-equivariant — rather than our
   ad-hoc 4-orientation bank. But note (1): the linear ceiling is +0.87 dB on
   derain (§1), and our oriented operator already failed to move it.

5. **Avoid dilated convolution on the high-frequency path** (§4).

**The honest headline:** §1 shows the *linear* frequency-spatial bridge is real
but caps at +0.68/+0.87 dB, and §5 suggests our worst gap is a learning-dynamics
problem. Together these argue the next experiment should be about **the loss and
the training protocol**, not another block.

---

## References

1. Burt & Adelson. *The Laplacian Pyramid as a Compact Image Code.* IEEE Trans. Communications 31(4), 1983.
2. Daugman. *Uncertainty relation for resolution in space, spatial frequency, and orientation optimized by two-dimensional visual cortical filters.* JOSA A 2(7), 1985.
3. Mallat. *A Theory for Multiresolution Signal Decomposition: The Wavelet Representation.* IEEE TPAMI 11(7), 1989.
4. Freeman & Adelson. *The Design and Use of Steerable Filters.* IEEE TPAMI 13(9), 1991.
5. Simoncelli & Freeman. *The Steerable Pyramid: A Flexible Architecture for Multi-Scale Derivative Computation.* ICIP 1995. https://www.cns.nyu.edu/~eero/steerpyr/
6. Liu, Zhang, Zhang, Lin & Zuo. *Multi-level Wavelet-CNN for Image Restoration.* CVPRW 2018. https://arxiv.org/abs/1805.07071
7. Chen et al. *Drop an Octave: Reducing Spatial Redundancy in CNNs with Octave Convolution.* ICCV 2019.
8. Rahaman et al. *On the Spectral Bias of Neural Networks.* ICML 2019. https://arxiv.org/abs/1806.08734
9. Chi, Jiang & Mu. *Fast Fourier Convolution.* NeurIPS 2020.
10. Cui et al. *Selective Frequency Network for Image Restoration* (SFNet), ICLR 2023; *Image Restoration via Frequency Selection* (FSNet), IEEE TPAMI.
11. Chen et al. *Frequency-Adaptive Dilated Convolution for Semantic Segmentation.* CVPR 2024. https://arxiv.org/abs/2403.05369
12. Cui et al. *AdaIR: Adaptive All-in-One Image Restoration via Frequency Mining and Modulation.* ICLR 2025. https://arxiv.org/abs/2403.14614

**Our own measurements referenced:** `scripts/freq_to_spatial.py`,
`scripts/spectral_samescene.py`, `scripts/spectral_spatial_proxy.py`,
`scripts/spatial_converter.py`, `reports/student_v3/design.md`,
`reports/kd_feature_multitask/cond_regression.md`.
