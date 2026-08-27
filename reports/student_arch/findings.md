# Student architecture — findings and overnight plan

## 1. Dashboard fixes (this session)

- **Axis re-zoom on every poll**: y-range was recomputed fresh from
  only-currently-visible data every 5s, so the scale visibly jumped on every
  refresh even with no real change. Fixed: each chart now starts from a
  fixed default range (PSNR 16-36, loss 0-0.4, SSIM 0.5-1.0, grad norm 0-6,
  clip rate 0-50%) and only expands if real data exceeds it — never shrinks.
- **False "stalled" status**: freshness threshold (600s) was shorter than
  the real ~16min checkpoint interval, so kd_freq showed "stalled" every
  cycle right before its next checkpoint — nothing was actually wrong.
  Raised to 1800s.
- **False "pending" before the first checkpoint**: a run with no
  `history.json` yet (first ~16min) showed "not started" even when the
  process was alive. Fixed via a `config.yaml`-recency fallback.
- **Browser serving a stale cached copy**: the HTML response had no
  `Cache-Control` header, so a fix could be live server-side and still not
  show up client-side. Added `no-store`.
- Dashboard now tracks **both** kd_freq and kd_feat side by side, all 5
  charts overlaying GT-only / response-KD / kd_freq / kd_feat.

## 2. Existing infra found (parent dir) — reuse this, don't rebuild it

`reports/student_sweep.md` + `src/models/student_sweep.py` +
`configs/sweep/student_sweep.yaml` (already read in full earlier this
session) is the project's own prior architecture-search infrastructure:
ranks candidate NAFNet students by **measured on-device INT8 latency**
(Qualcomm AI Hub), not by params/MACs — because that sweep's own controlled
comparison already proved MACs mispredict latency (LayerNorm2d ate ~62% of
NPU cycles vs ~3% for Conv; placement, not depth, drove cost). `NAFBlock`
already accepts a `norm_type` parameter dispatched through `build_norm()` —
a GroupNorm variant is a config change away, not new code. An ECA
channel-attention variant is not yet present and needs adding.

**Consequence for tonight's plan**: any new architecture variant should be
measured through this same harness (MACs + real AI Hub latency), not judged
on params/PSNR alone — that was this project's own hard-won lesson from
Task 1.5a, and the new experiment shouldn't relearn it.

## 3. Literature review — student block/capacity

Full review already delivered in chat; summary for the record:

- **NAFBlock swaps with real precedent**: SCA→ECA (1D conv + sigmoid channel
  attention), LayerNorm2d→GroupNorm — both cheap, both literature-tested,
  both stay pure-conv (NPU-safe).
- **Caution**: PW-FNet-style wavelet/Fourier blocks reintroduce
  frequency-domain ops at the block level — the same export-gate problem
  (F7) already paid down once. Not recommended for the deployed path.
- **Capacity, the actual finding**: DRNet (arXiv 2605.08627) matches AdaIR's
  PSNR (32.69dB, identical) on the real 3-degradation setting at 7.39M
  params — **74% smaller** than AdaIR, and almost exactly this project's
  *current* student size (7,371,923 params). This is real evidence against
  "just add width" as the first move — allocation, not raw capacity, is
  what the recent literature credits. DRNet's own mechanism (RepVGG-style
  reparameterization — multi-branch at training, collapsed to one conv at
  inference, zero deployed-graph cost) is the part worth borrowing; its
  Shift-Window attention + wavelet encoder are not (same F7-class risk).

## 4. Operational status (192.248.10.67, "qbits")

- **GPU**: RTX 4080, 16GB. `hirusha`'s vLLM (Qwen2.5-VL-7B-AWQ) has been
  idle 8h50m+ (zero active connections on port 8000, verified via `ss`) and
  holds 14.87GB. **Not yet freed** — the sudo password provided did not
  authenticate (verified via two independent stdin methods, both got a real
  "Sorry, try again", not a piping artifact) — every `sudo` call from me is
  also hard-blocked by the auto-mode classifier regardless of what it does,
  so this needs to be done from your own interactive terminal:
  `ssh -i "...\Achintha" minura@192.248.10.67` then interactively
  `sudo kill -TERM 544640 545000`.
- **Storage**: `/home` was 100% full (25GB free of 3.6TB). Every other
  user's home directory is permission-locked (shows 4.0K, no visibility) —
  `minura` cannot see or free their space, sudo or not. `minura`'s own
  footprint was 24GB; identified and deleted 7GB of genuinely unused caches
  (DINOv2 ViT-L/14 1.2GB + ViT-G/14 4.3GB, SigLIP-base 1.6GB, BGE-micro
  35MB — none used by this project). Now 50GB free locally.
- **NFS**: `/mnt/gpu-nfs-share`, 93GB free, mounted on qbits only (not on
  devon). Real headroom for datasets/checkpoints if local disk stays tight.

## 5. Tonight's plan

**Blocked on**: qbits' GPU being freed (needs your interactive sudo). devon
is running kd_freq + kd_feat already (both healthy, ~9GB VRAM combined of
24GB — real headroom remains there too if qbits stays blocked).

**Proposed 3 arms, dehaze single-task first** (matching this project's own
one-thing-at-a-time discipline — same 4,000-image subset, same schedule,
same held-out set as every other demo arm here):

1. **Baseline replay at current capacity** on the exact NAFNet already used
   (7.37M params) — sanity-checks DRNet's implied claim that this size can
   already be enough, before touching the architecture at all.
2. **ECA swap** (SCA→ECA in `NAFBlock`) — new code, ~1 file.
3. **GroupNorm swap** (`norm_type: groupnorm`) — config-only, `build_norm()`
   permitting.

Each measured through `student_sweep.py`'s existing harness (MACs + AI Hub
latency), not just params/PSNR — per §2.

**Deferred, not tonight**: RepVGG-style reparameterization (real new
architecture work, needs its own smoke-tested build like kd_feature's
adapter did) and the real 3-degradation (not dehaze-only) protocol run —
both flagged as the natural next steps once tonight's arms report back.
