# Overnight autonomous run log

Started 2026-07-30 21:23 UTC


## Task 1 — trainer built, smoke test passed

- 200-iter smoke: loss decreasing, validation via locked harness, checkpoint
  save/resume, **peak VRAM 2.14 GB logged (instrumentation verified non-zero)**.

### Throughput measurement and two deviations from the spec

Measured on the RTX 3050 6GB:

| batch | patch | ms/iter | peak VRAM | 30k iters |
|---|---|---|---|---|
| 32 | 256 | **6214** | **8.38 GB** | 51.8 h |
| 16 | 256 | 473 | 4.21 GB | 3.9 h |
| 32 | **128** | **244** | **2.13 GB** | **2.0 h** |
| 64 | 128 | 473 | 4.21 GB | 3.9 h |

**Deviation 1 — patch 128 instead of 256.** At batch 32 / patch 256 the run
needs 8.38 GB on a 6 GB card, spills to host memory, and collapses to 6.2 s/iter
(25x slower than the same batch at patch 128). Patch **128 is also AdaIR's own
training patch size** (`options.py:15`), so this moves *closer* to the reference
protocol rather than away from it.

**Deviation 2 — 30k iterations instead of 50k.** 50k x 3 arms = 10.2 h of pure
compute before validation, which does not leave room for the escalation ladder
or Task 5. 30k gives ~2.3 h/arm including validation.

Both deviations are applied **identically to every arm**, so the controlled
comparison — the entire point of the ablation — is preserved. Only absolute
quality is affected, not the ranking.

**Caveat to carry into the report:** at 30k iterations the arms may not have
fully separated, and a ranking at 30k is not guaranteed to hold at convergence.
Flagged for review.

=== 22:01 UTC starting Q-A ===

### JUDGEMENT CALL — restarted the arms on BSD400+WED (2026-07-31 ~07:25 local)

Q-A had reached iteration 2000 (PSNR 25.61) on **BSD400 alone** when the WED
download completed. Killed and restarted all arms on the full set.

Reason: 30k iters x batch 32 = 960k samples. Over BSD400's 400 images that is
**2400 epochs** — a severe overfit regime that risks compressing the very
differences between norm variants the ablation exists to measure. With WED
(4744 images, verified count) the set is 5144 images and **187 epochs**, which
is a normal training regime.

Cost: ~15 minutes of Q-A progress. Judged clearly worth it — an ablation run in
a heavy-overfit regime could have produced a null result that reflected the data
budget rather than the architecture.

### Also noted: create_run_dir stalled ~3.5 h

config.yaml written 03:31, env.txt 07:07. The `pip freeze` subprocess in
`create_run_dir` blocked, most likely contending with the concurrent AI Hub
submission and WED download. Training did not begin until 07:07. Not fatal, but
it cost 3.5 h of the night and is worth making non-blocking (or cached) before
the next unattended run.

=== 01:53 UTC starting Q-A (BSD400+WED, 5144 imgs) ===
