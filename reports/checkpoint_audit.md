# AdaIR checkpoint audit

Every released checkpoint, hashed and verified to load onto the architecture with **zero missing and zero unexpected keys**. A partial load that silently succeeds would produce plausible-but-wrong numbers and poison everything downstream.

Reference architecture: `AdaIR(decoder=True)` from `third_party/AdaIR` @ `ccb8b98`, **28,784,824 parameters**.

| checkpoint | MB | clean load | entries | params | epoch | step | prefix |
|---|---|---|---|---|---|---|---|
| `adair-single-dehaze.ckpt` | 346 | **yes** | 587 | 28,784,824 | 49 | 450850 | `net.` |
| `adair-single-denoise.ckpt` | 346 | **yes** | 587 | 28,784,824 | 149 | 800550 | `net.` |
| `adair-single-derain.ckpt` | 346 | **yes** | 587 | 28,784,824 | 49 | 150000 | `net.` |

## SHA256

| checkpoint | sha256 |
|---|---|
| `adair-single-dehaze.ckpt` | `33195f1362c71ad31f27765b29e510aae7be046ef72aeeee109bccee5ad00881` |
| `adair-single-denoise.ckpt` | `17acf9d598a18b5b86e26718377827a98fd09d075c6547b76c188ce9e3bd3f77` |
| `adair-single-derain.ckpt` | `7a9558ae5a1e096da0d9633f5d38c34270f8aa9f99f680ef1d4a991f0f312a81` |

All checkpoints load cleanly onto the reference architecture.

## Availability of single-task specialists (strategic)

Verified clean-loading single-task checkpoints: `adair-single-dehaze.ckpt`, `adair-single-denoise.ckpt`, `adair-single-derain.ckpt` — covering **dehaze, denoise, derain**.

**All three specialists for the 3-degradation protocol are available.** This makes the specialist→generalist (multi-teacher) direction viable in Phase 02 with no third-party model sourcing.

It is also a *cleaner* experiment than externally-sourced specialists would have been: these share the **same architecture** as the all-in-one teacher, so any student improvement is attributable to specialist knowledge rather than to architectural diversity among teachers. One codebase, one loading path, one licence.

> Recorded as available. **Not** in scope for Phase 01 — no multi-teacher infrastructure is built in this task.

## Notes

- These are **full Lightning training checkpoints** (~346 MB ≈ 3× the 28.78M parameters: weights plus two Adam moments), not inference-only exports. Weights live under `state_dict` with a uniform `net.` prefix from the `AdaIRModel` wrapper (`test.py:21`); it is stripped before loading.
- Loading is verified with `strict=False` **and then asserted** to have produced no missing and no unexpected keys, which is stricter than `strict=True` alone because it also reports what would have been silently ignored.
