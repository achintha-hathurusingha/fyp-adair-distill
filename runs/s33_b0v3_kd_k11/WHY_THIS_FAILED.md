# B0V3-KD-K11, first attempt — FAILED, kept as evidence

Stopped at 15k on 2026-09-01. **Do not treat these numbers as a result.**
This run is preserved because it is what found the bug.

At 15k it was **1.335 dB behind its matched control and widening**, with the
damage almost entirely on dehaze (**-3.026**) while denoise was untouched
(**-0.037**).

Ruled out first: not a config problem. `diff configs/train/b0v3_kd_feat.yaml
configs/train/b0v3_kd_k11.yaml` showed only the four block flags, and the model
was an exact identity at init (max |control - K11| = 0.000e+00, 4 extra tensors
/ 7,088 values).

## The defect

`PlainLargeKernelBlock.forward` was `x + fuse(conv(x))` with
`nn.init.zeros_(self.fuse.weight)`. Measured at step 0:

    reparam_blocks.2.conv.weight   |w|max=0.0909   |grad|max=0.000e+00   <- frozen
    reparam_blocks.2.fuse.weight   |w|max=0.0000   |grad|max=1.182e-03   <- moving

dL/d(conv) is proportional to fuse.weight = 0, so the 11x11 depthwise kernel got
no gradient and stayed at its random Kaiming init, while fuse grew because
dL/d(fuse) is proportional to conv(x) != 0. The block spent early training
**scaling up a frozen random large kernel** and injecting it into the decoder —
a random blur/high-pass. That is exactly why dehaze (smooth global structure)
collapsed and denoise did not. The zero-init was meant to be a warm start and
did the opposite.

`ReparamOrientedBlock` had the identical structure, so B0V3-KD-ORI would have
failed the same way and we would have wrongly concluded that orientation does
not help. Its partial run is in `runs/s33_b0v3_kd_ori/` (killed before its first
checkpoint).

## Fix

Delta-initialise the depthwise kernel (centre tap 1, rest 0) so conv(x) = x at
init; fuse stays zero, so the block is still an exact identity, and what fuse
scales up early is a copy of the signal rather than noise. RepLKNet does the
same for large kernels. Applied to both classes in
`src/models/reparam_oriented.py`; verified by `scripts/verify_delta_init.py`.

## Why this was caught at all

The plain-k11 arm existed only as a **receptive-field control**, to separate
orientation from kernel size. It instead caught a defect that would have
silently invalidated the whole of Phase 3 — the oriented arm would have looked
like a clean negative result.
