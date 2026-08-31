# Why distillation keeps failing here — a literature review

**Question.** Not "does KD work" in general, but: *why does KD help one of our
tasks and actively harm the other two?* Every KD variant we have tried has
either regressed or come out null, while the literature reports KD as broadly
beneficial. Something specific to our setting explains the difference, and it
should be identifiable from prior work rather than treated as a local quirk.

Tagged **[theory]**, **[literature]**, **[measured here]** throughout.

---

## 0. The pattern that needs explaining

**[measured here]** KD's effect is not uniform — it is cleanly ordered by task:

| task | KD effect (TEST07-B, 50ep x 3 seeds) | KD effect (90k arms) |
|---|---:|---:|
| denoise | **+1.238 dB** | +0.007 dB |
| dehaze | -1.027 dB | -0.545 dB |
| derain | **-2.591 dB** | -0.757 dB |

Two independent experiments, same ordering. That is a structured effect, not
noise.

---

## 1. The capacity gap — the explanation that fits our data

**[literature]** Cho & Hariharan (*On the Efficacy of Knowledge Distillation*,
ICCV 2019) first showed systematically that **an overly powerful teacher harms
distillation**: the student cannot match a function too far from its own
capacity, and student accuracy *degrades* as the teacher grows. Mirzadeh et al.
(*Improved Knowledge Distillation via Teacher Assistant*, AAAI 2020) confirmed
it and proposed TAKD — insert an intermediate-capacity assistant to bridge the
gap.

**[measured here] Our per-task data fits this almost exactly.** Using the
teacher-minus-GT-only-student gap as the per-task capacity gap:

| task | teacher gap | KD (TEST07-B) | KD (90k arms) |
|---|---:|---:|---:|
| denoise | **0.567 dB** | +1.238 | +0.007 |
| dehaze | 2.283 dB | -1.027 | -0.545 |
| derain | **2.897 dB** | -2.591 | -0.757 |

- Pearson r (gap vs KD effect): **-0.987** (TEST07-B), **-0.9999** (90k arms)
- Rank ordering by teacher gap is **identical** to rank ordering by KD benefit,
  in both experiments independently.

**The single task where our student is nearly at teacher level (denoise, 0.57 dB)
is the single task where KD helps.** Where the teacher is far ahead, KD hurts,
and it hurts *in proportion to how far ahead it is*.

> **Honest caveat: n = 3.** A high |r| on three points is weak on its own —
> three points nearly always fit a line. The load-bearing evidence is that the
> *ordering* reproduces across two independent experiments with different
> schedules, seeds and checkpoints, not the correlation coefficient.

---

## 2. Regression is not classification — the dark knowledge is missing

**[theory]** Hinton et al.'s original argument is that soft labels carry *dark
knowledge*: the relative probabilities across wrong classes encode similarity
structure the hard label discards. **In image restoration there is no such
structure.** The teacher's output is just another image, and the KD target is a
regression target of the same type as the ground truth — but strictly worse,
because the teacher is imperfect.

**[literature]** Reviews of KD for low-level vision make the same point:
feature maps in classification are "abstract and compact", which is what makes
distillation effective there, and this *does not hold* for restoration/SR, where
distillation is correspondingly difficult.

**[theory] The consequence for us is sharp.** When ground truth is available,
the teacher output is a strictly noisier version of the same supervision signal.
Matching it means matching the teacher's *errors* too. On denoise our teacher is
0.57 dB from our student — the errors it adds are small. On derain it is 2.90 dB
ahead, but its output is still 3+ dB from ground truth, so pulling the student
toward it injects substantial error the GT loss was already correcting.

---

## 3. Gradient conflict between the two objectives

**[literature]** The two losses are not automatically compatible. Work on
multi-task KD optimisation (*MoKD*, 2025) identifies **gradient conflicts** —
task-objective and distillation gradients misaligned — as a core failure mode,
and notes feature-based KD in particular struggles with "inconsistency between
the optimization objectives of ground-truth supervision and the distillation
targets."

*Do Not Blindly Imitate the Teacher* (Perturbed Loss KD, 2023) makes the
complementary point: the discrepancy between the teacher's output distribution
and the true distribution means **blind imitation of an unreliable teacher
produces inferior performance**, and the fix is to stop treating the teacher's
output as ground truth.

**[measured here] We have already observed this failure mode directly.** Our v1
FiLM conditioning regressed on every task and the gap *widened* over training.
The diagnosis at the time was structural — the FiLM modulated the exact tensor
the feature-KD loss reads, so two objectives fought over one representation.
That is a gradient conflict, and the literature names it.

---

## 4. Multi-task negative transfer

**[literature]** Negative transfer is the documented phenomenon where improving
one task degrades others, caused by task dissimilarity (optimal representations
conflict), data imbalance, and gradient interference. Multi-task distillation
work proposes collecting **task-specific Pareto-optimal teachers** and using
multi-teacher KD, rather than one teacher for all tasks.

**[measured here]** This is our exact configuration: one all-in-one teacher,
three degradations with very different structure (isotropic noise, oriented
sparse rain, smooth global haze), distilled through one shared representation.
Our measured result — one task up, two down — is the textbook signature.

---

## 5. Restoration-specific KD

**[literature]** *Knowledge Distillation for Image Restoration: Simultaneous
Learning from Degraded and Clean Images* (Zhang & Yan, 2025) states directly
that KD's potential "in image-to-image translation, particularly image
restoration, remains underexplored", and that restoration differs from SR
because it must **remove degradation before reconstructing**. Their SLKD
framework uses a **dual-teacher, single-student** design splitting the two
sub-problems: Degradation Removal Learning on the encoder, Image Reconstruction
Learning on the decoder — >80% FLOP/parameter reduction with quality retained.

**[literature]** SR-KD work adds a warning that maps onto our own frequency
results: because student output, teacher output and the input are all highly
correlated, **unscreened knowledge transfer concentrates on redundant
low-frequency information while failing to transfer the high-frequency detail
guidance the student actually needs.**

**[measured here]** Consistent with our worst task being derain — sparse,
oriented and high-frequency — and our best being denoise.

---

## 6. What this predicts, and what to do

Ranked by how well the evidence supports them:

1. **Task-selective KD.** The capacity-gap result (§1) predicts KD should be
   applied *only* where the teacher-student gap is small — i.e. denoise
   (+1.238 dB measured) — and disabled for derain and dehaze. This is the
   cheapest test we have and it follows directly from our own strongest
   correlation. **Already proposed in this project and still never run.**

2. **Per-task specialist teachers.** §1 and §4 both point here: a specialist
   teacher has a smaller gap on its own task, which should convert KD from
   harmful to helpful. **We already hold `adair-single-dehaze.ckpt` and
   `adair-single-denoise.ckpt`** — and measured the single-task dehaze
   specialist at 31.80 dB vs the all-in-one's 31.06, so the specialist really is
   the stronger per-task teacher. No new teacher training required.

3. **Bound the imitation.** §3 says don't treat teacher output as ground truth.
   Concretely: weight KD by per-task teacher reliability, or clamp the KD loss
   where the teacher itself is far from GT.

4. **TAKD-style assistant.** §1's canonical remedy. Real, but expensive — it
   needs an intermediate model trained and adds a stage.

5. **Split the objective SLKD-style** (§5) — degradation removal on the encoder,
   reconstruction on the decoder — rather than one KD loss on one tensor.

**The one result that complicates this story.** **[measured here]** The running
`B0V3-KD-FEAT` arm (v3 architecture + the same KD) is currently **ahead of both
parents** at every checkpoint from 9k to 24k, and its gain is concentrated in
**dehaze (+0.562 dB vs the NAFNet KD arm)** — the task where §1 predicts KD
should hurt. If that survives to 90k it implies an architecture x distillation
interaction the capacity-gap account alone does not cover. It is early (the
previous v3 arm reversed after 50k), but it is the strongest counter-evidence we
have and should not be written out of the review.

---

## References

1. Hinton, Vinyals & Dean. *Distilling the Knowledge in a Neural Network.* NIPS Deep Learning Workshop, 2015. arXiv:1503.02531
2. Romero et al. *FitNets: Hints for Thin Deep Nets.* ICLR 2015. arXiv:1412.6550
3. Cho & Hariharan. *On the Efficacy of Knowledge Distillation.* ICCV 2019. arXiv:1910.01348
4. Mirzadeh et al. *Improved Knowledge Distillation via Teacher Assistant.* AAAI 2020. arXiv:1902.03393
5. Son et al. *Densely Guided Knowledge Distillation using Multiple Teacher Assistants.* ICCV 2021. arXiv:2009.08825
6. Li et al. *Multi-Task Distillation: Towards Mitigating the Negative Transfer in Multi-Task Learning.* ICIP 2021.
7. Jacob et al. *Online Knowledge Distillation for Multi-Task Learning.* WACV 2023.
8. *Do Not Blindly Imitate the Teacher: Using Perturbed Loss for Knowledge Distillation.* 2023. arXiv:2305.05010
9. *MoKD: Multi-Task Optimization for Knowledge Distillation.* 2025. arXiv:2505.08170
10. Zhang & Yan. *Knowledge Distillation for Image Restoration: Simultaneous Learning from Degraded and Clean Images.* 2025. arXiv:2501.09268
11. *Learning Knowledge Representation with Meta Knowledge Distillation for Single Image Super-Resolution.* 2022. arXiv:2207.08356
12. *A Comprehensive Review of Knowledge Distillation in Computer Vision.* 2024. arXiv:2404.00936

**Our own measurements referenced:** `reports/kd_feature_multitask/`,
`teacher-experiments/test07_b/`, `reports/student_v3/`,
`runs/b0v2_kd_feat/`, `runs/b0v3/`, `runs/b0v3_kd_feat/`.
