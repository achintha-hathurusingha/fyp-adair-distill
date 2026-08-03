# devon CPU fault — how the healthy cores were identified and pinned

Reference for how training is kept off devon's two degraded CPU cores. Commands
are exactly what was run, so they can be re-executed to re-verify.

Machine: devon, `192.248.10.68`, Intel Core i9-14900K + RTX 4090, Ubuntu.
Symptom: random segfaults and `INTERNALERROR` during training and `pytest`, at a
**different point each run** — the signature of memory corruption rather than a
code bug, which fails in the same place every time.

## 1. Map logical CPUs to physical cores

```bash
lscpu -e                       # CPU -> CORE mapping, 32 logical / 24 physical
```

The i9-14900K has 8 performance cores (2 logical CPUs each, CPUs 0-15) and
16 efficiency cores (1 logical CPU each, CPUs 16-31).

## 2. Test each physical core in isolation

The whole diagnosis is this loop: pin the test suite to one physical core's pair
of logical CPUs and run it four times.

```bash
taskset -c 0,1   python -m pytest tests/ -q     # physical core 0
taskset -c 2,3   python -m pytest tests/ -q     # physical core 1
taskset -c 8,9   python -m pytest tests/ -q     # physical core 4
taskset -c 10,11 python -m pytest tests/ -q     # physical core 5
```

| physical core | logical CPUs | max boost | result |
|---|---|---|---|
| 0, 1, 2, 3, 6, 7 | 0-7, 12-15 | 5.7 GHz | **4/4 pass each** |
| **4** | **8, 9** | **6.0 GHz** | **3/4 segfault** |
| **5** | **10, 11** | **6.0 GHz** | **2/4 segfault / abort** |

## 3. Confirm the two bad cores are the highest-voltage ones

```bash
cat /sys/devices/system/cpu/cpu*/cpufreq/cpuinfo_max_freq
```

Cores 4 and 5 are the only two that boost to 6.0 GHz — Intel's Thermal Velocity
Boost "favoured" cores. Every other core caps at 5.7 GHz.

This is what makes it a diagnosis rather than a correlation: the failing cores
are precisely the two that ran at the highest sustained voltage, which is the
mechanism of the documented Raptor Lake degradation defect.

Microcode was checked first, since that is the usual remedy:

```bash
grep microcode /proc/cpuinfo | head -1        # 0x133
```

`0x133` is **above** Intel's `0x12B` mitigation, so the voltage fix was already
applied. Microcode prevents further degradation; it cannot reverse damage
already done. A machine still failing on current microcode is one whose silicon
is already degraded — so no software update would help, and the real remedy is
an RMA.

## 4. Verify the split in both directions

Excluding cores is only meaningful if the excluded set actually fails and the
kept set actually passes.

```bash
taskset -c 0-7,12-31 python -m pytest tests/ -q    # healthy set  -> 10/10 pass
taskset -c 8-11      python -m pytest tests/ -q    # bad cores    -> 3/6 fail
```

## 5. Apply it — `scripts/run_b0_devon.sh`

```bash
exec taskset -c 0-7,12-31 python -m src.train.train "$@"
```

Called in place of `python -m src.train.train`:

```bash
./scripts/run_b0_devon.sh --arm B0 --seed 1 --out-root runs/b0_final --num-workers 8
```

**No changes were needed in `train.py`.** `taskset` sets affinity on the process
before `exec`, and dataloader workers inherit it when they fork — so pinning the
launcher covers the entire process tree. Putting it inside `train.py` would also
hardcode one machine's hardware fault into the training code.

28 of 32 logical CPUs remain, so the RTX 4090 is not dataloader-bound. An earlier
E-cores-only workaround (`taskset -c 16-31`) was abandoned as unnecessarily
conservative: it gave up all six healthy P-cores and left the GPU at 23%
utilisation.

## 6. Check a running job

```bash
taskset -cp 6908
# pid 6908's current affinity list: 0-7,12-31
```

Necessary because `taskset` **execs into python and replaces itself**, so `ps`
shows a bare `python -m src.train.train` with no trace of the pin. The command
line cannot tell you whether a run is pinned; only the live process can.

## 7. Evidence the workaround actually works

Two B0 runs resumed from the same iteration-20000 checkpoint produced
**bit-identical** metrics at iteration 25000 — including a gradient norm of
`65240022.433` matching to every printed digit.

A machine with active memory corruption cannot reproduce an eight-digit value
exactly. This is the difference between "it crashes less now" and a verified
fix. (That run still diverged — but from a real numerical bug, finding F9, not
from hardware. Separating the two is exactly what this test established.)

Throughput on the healthy set, 12 workers: **10.9 steps/s**, steady across three
consecutive 5000-step intervals, GPU 72-76% utilised — vs **1.66 steps/s** on the
laptop RTX 3050, a **6.5x** speedup (~7.7 h/seed vs ~50 h). devon also matched
the laptop's B0 trajectory to **0.004 dB** at iteration 5000.

## Limitations

- This **routes around** damaged silicon; it does not repair it. The correct fix
  is an RMA of the CPU (Intel extended the warranty to 5 years for this defect).
- Degradation is **progressive**. The healthy cores may follow. Re-run step 2
  periodically rather than assuming the 2026-08-01 result still holds.
- A passing short test is not evidence of health. On 2026-07-31 a synthetic loop
  passed 118/118 and a real `pytest` segfaulted minutes later. Only real
  workloads surface the fault — always re-verify with the actual test suite.
- `train.py` does not record CPU affinity in the run directory, so provenance for
  a finished run rests on this document plus a live `taskset -cp` check. Worth
  closing before B0-v2 by writing `os.sched_getaffinity(0)` into `env.txt`.
