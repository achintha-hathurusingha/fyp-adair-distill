# AdaIR checkpoint audit

Every released checkpoint, hashed and verified to load onto the architecture with **zero missing and zero unexpected keys**. A partial load that silently succeeds would produce plausible-but-wrong numbers and poison everything downstream.

Reference architecture: `AdaIR(decoder=True)` from `third_party/AdaIR` @ `ccb8b98`, **28,784,824 parameters**.

| checkpoint | MB | clean load | entries | params | epoch | step | prefix |
|---|---|---|---|---|---|---|---|
| `adair-single-dehaze.ckpt` | 346 | **yes** | 587 | 28,784,824 | 49 | 450850 | `net.` |
| `adair-single-denoise.ckpt` | 346 | **yes** | 587 | 28,784,824 | 149 | 800550 | `net.` |
| `adair-single-derain.ckpt` | 346 | **yes** | 587 | 28,784,824 | 49 | 150000 | `net.` |
| `adair3d.ckpt` | 346 | **yes** | 587 | 28,784,824 | 149 | 650700 | `net.` |
| `adair5d.ckpt` | 346 | **yes** | 587 | 28,784,824 | 149 | 745500 | `net.` |

## SHA256

| checkpoint | sha256 |
|---|---|
| `adair-single-dehaze.ckpt` | `33195f1362c71ad31f27765b29e510aae7be046ef72aeeee109bccee5ad00881` |
| `adair-single-denoise.ckpt` | `17acf9d598a18b5b86e26718377827a98fd09d075c6547b76c188ce9e3bd3f77` |
| `adair-single-derain.ckpt` | `7a9558ae5a1e096da0d9633f5d38c34270f8aa9f99f680ef1d4a991f0f312a81` |
| `adair3d.ckpt` | `f3822d9c2eaf4a812f4122c5ec0082bc8eaf2bee9cb2b3a961d4984ed05937fb` |
| `adair5d.ckpt` | `e5b8ea892b68057f7d20265ac92585286362b0512c755fddd46681364d17dcf9` |

All checkpoints load cleanly onto the reference architecture.

## Availability of single-task specialists (strategic)

Verified clean-loading single-task checkpoints: `adair-single-dehaze.ckpt`, `adair-single-denoise.ckpt`, `adair-single-derain.ckpt` — covering **dehaze, denoise, derain**.

**All three specialists for the 3-degradation protocol are available.** This makes the specialist→generalist (multi-teacher) direction viable in Phase 02 with no third-party model sourcing.

Architecturally they are identical to the all-in-one teacher — every checkpoint loads onto the same `AdaIR(decoder=True)` with the same parameter count — so one codebase, one loading path, one licence.

### Confound: the specialists were NOT trained on a common protocol

| checkpoint | epoch | global_step | steps/epoch |
|---|---|---|---|
| `adair-single-dehaze.ckpt` | 49 | 450,850 | 9,017 |
| `adair-single-denoise.ckpt` | 149 | 800,550 | 5,337 |
| `adair-single-derain.ckpt` | 49 | 150,000 | 3,000 |
| `adair3d.ckpt` | 149 | 650,700 | 4,338 |
| `adair5d.ckpt` | 149 | 745,500 | 4,970 |

Epoch counts, step counts and steps-per-epoch all differ across the specialists and against the all-in-one. Differing steps-per-epoch implies **differing training-set sizes**, i.e. task-specific training protocols rather than one shared regime.

**Consequence for the specialist→generalist premise:** any measured specialist-over-all-in-one advantage is *confounded* — part of it is specialisation, part is simply a different (often longer) training protocol on a different data mix. A student inheriting that surplus would be inheriting both, and the claim "specialist knowledge transfers" would be weaker than it appears. This does not kill the option, but it must be stated whenever the gap is quoted.

> Recorded as available. **Not** in scope for Phase 01 — no multi-teacher infrastructure is built in this task. The gap itself is measured under our locked conventions at G3.

## Notes

- These are **full Lightning training checkpoints** (~346 MB ≈ 3× the 28.78M parameters: weights plus two Adam moments), not inference-only exports. Weights live under `state_dict` with a uniform `net.` prefix from the `AdaIRModel` wrapper (`test.py:21`); it is stripped before loading.
- Loading is verified with `strict=False` **and then asserted** to have produced no missing and no unexpected keys, which is stricter than `strict=True` alone because it also reports what would have been silently ignored.
