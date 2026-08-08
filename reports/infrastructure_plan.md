# Infrastructure plan — experiment tracking, scheduling, backup, monitoring

Written for the fyp_visionx team. Scope: everything after the Phase 01 demo,
covering the four things currently missing — a job queue, durable experiment
tracking, off-machine backup, and alerting.

**Access model, which shapes every decision below.** Sudo exists on **devon
only**, and requires a password, so nothing privileged there can be automated —
it needs a person at a terminal. Everything else runs in LXC containers on a
**separate physical host**, provisioned through a different channel. The plan is
therefore split into three parts: what someone else provisions, what you run by
hand on devon, and what can be configured over SSH afterwards.

---

## 1. Measured starting point

Surveyed 2026-08-08, not assumed:

| | |
|---|---|
| devon | RTX 4090 24 GB, driver 580.173.02, 32 logical CPUs, 1.8 TB disk, **783 GB free** |
| devon IP | `192.248.10.68/24` — **publicly routable** |
| accounts | `anujaya`, `dhawala`, `hasitha`, `minura`, `ranga` — **four are empty (4 KB)**; only `minura` has 113 GB and a conda install |
| storage in use | `runs/` 5.9 GB, `data/` 15 GB, other FYP work 42 GB |
| schedulers | none — no SLURM, no Ray. Jobs launch with `nohup` |
| sudo | present, **password-gated**, shared with others |
| services host | separate physical machine, LXC/VM, **200 GB** available (expandable) |
| git | gitlab.com (public) for the team repo; GitHub for this sub-project |

Two of these correct earlier assumptions in this project's notes: `runs/` is
**5.9 GB, not 100+ GB**, and **nobody else is actually using devon yet** —
contention is anticipated, not current. That is the good case: the queue can be
built before it hurts rather than retrofitted under pressure.

---

## 2. Architecture

```
  services host (separate machine, LXC)          devon (GPU)
  ┌────────────────────────────────────┐        ┌──────────────────────┐
  │ slurm-ctl   slurmctld + slurmdbd   │◄──────►│ slurmd               │
  │             MariaDB (accounting)   │        │  CpuSpecList=8,9,10,11│
  │ mlflow      MLflow + PostgreSQL    │◄───────│  Gres=gpu:1          │
  │ minio       S3 artifact store      │◄───────│                      │
  │ monitor     Prometheus + Grafana   │◄───────│ node_exporter        │
  │             + Alertmanager → email │        │ dcgm-exporter        │
  │ backup      restic repository      │◄───────│ restic client (cron) │
  │ ci-runner   GitHub Actions runner  │        │                      │
  └────────────────────────────────────┘        └──────────────────────┘
           ▲                                     future: gpu-node-2
      SSH tunnel only — nothing bound to a public interface
```

**The controller lives on the services host, not devon.** devon has degraded
silicon with a pending RMA and dropped off the network twice in 48 hours. A
queue that dies with the compute node is not a queue.

### Tool choices, and the ones deliberately rejected

| need | choice | why |
|---|---|---|
| scheduling | **SLURM** | Enforces the degraded-core exclusion at scheduler level (below). Standard in academia. Second GPU node is one `slurmd`. |
| tracking | **MLflow** | Runs already write resolved config, git hash, pip freeze and metrics — this renders and compares them. Model registry included. |
| artifacts | **MinIO** | S3 API, so MLflow's artifact store is standard rather than a bespoke path. Survives devon entirely. |
| backup | **restic** | Deduplicating, encrypted, incremental. Internal network, so uni-network speed is irrelevant. |
| metrics | **Prometheus** + node/DCGM exporters | GPU temperature, utilisation, and the CPU package temperature that hit 84 °C under four concurrent runs. |
| alerting | **Alertmanager** → email | Must distinguish "devon unreachable" from "job failed", or every network blip pages about healthy jobs. |
| CI | **self-hosted GitHub Actions runner** | Tests run internally; no minutes cost; can reach devon if needed. |
| sweeps | **Ray Tune inside an sbatch allocation** | Phase 02's KD grid. Ray as the *outer* scheduler was considered and rejected — see below. |
| data versioning | **checksummed manifests in git** — *not DVC* | See below. |

**Why not Ray as the scheduler.** It was the initial recommendation and is a
reasonable choice, but SLURM wins on two specifics here. First, `CpuSpecList`
enforces the damaged-core exclusion for every user, whereas Ray would require
every launcher to remember `taskset` — a convention invisible to `ps`, and one
that a teammate running `python -m src.train.train` directly would bypass
silently. Second, Ray's job-submission API is remote code execution by design;
on a publicly routable address that is an actively exploited vector, with GPU
hosts specifically targeted. SLURM's attack surface is smaller and its ports are
easier to reason about. Ray Tune still gets used, one layer down.

**Why not DVC.** The datasets are immutable public benchmarks — BSD400, WED,
Rain100L, RESIDE-OTS. What actually varies is *subsets and splits*, and those
are already committed as seeded text manifests (`reports/dehaze_train_list.txt`,
`derain_train_list.txt`, `reside_required_files.txt`) with the generating seed in
the header and a verifier that refuses partial sets. DVC would add a second
source of truth for a problem already solved. What is missing is only a
**checksum registry** so a corrupted or silently-changed dataset is detectable —
a small script, not a framework.

### The scheduler enforces the hardware mitigation

This is the strongest single argument for SLURM here. `reports/devon_cpu_mitigation.md`
records that physical cores 4 and 5 (logical CPUs 8–11) are degraded and must be
excluded. Today that lives in one launcher script; `taskset` execs into python so
`ps` shows no trace, and any teammate launching directly lands on damaged
silicon with nothing to catch it.

```
NodeName=devon CPUs=32 CpuSpecList=8,9,10,11 Gres=gpu:1 RealMemory=64000 State=UNKNOWN
```

Cores 8–11 are then reserved away from every job, for every user, by the
scheduler. The mitigation stops being a convention.

---

## 3. What gets provisioned (hand to the other session)

Six unprivileged LXC containers. Full copy-paste specification in
`reports/infra_provisioning_prompt.md`.

| container | vCPU | RAM | disk | ports (internal only) |
|---|---|---|---|---|
| `slurm-ctl` | 2 | 2 GB | 20 GB | 6817, 6819 |
| `mlflow` | 2 | 4 GB | 20 GB | 5000 |
| `minio` | 2 | 4 GB | **100 GB** | 9000, 9001 |
| `monitor` | 2 | 4 GB | 30 GB | 9090, 3000, 9093 |
| `backup` | 1 | 2 GB | **80 GB** | 22 (restic over SSH) |
| `ci-runner` | 4 | 8 GB | 40 GB | — (outbound only) |

Storage totals 190 GB of the 200 GB available, leaving headroom. Sizing rationale
is in §6.

---

## 4. What you run on devon (sudo, password, interactive)

Ordered, with what each step is for. Nothing here can be automated over SSH
because sudo is password-gated — that is a feature, not an obstacle, on a shared
machine.

1. **Pin the NVIDIA driver.** `unattended-upgrades` already broke it mid-run
   once (`memory/devon-unattended-upgrades`). `apt-mark hold` the driver and
   CUDA packages so it cannot happen again unattended.
2. **Install munge**, then copy `/etc/munge/munge.key` from `slurm-ctl` and set
   ownership 0400 munge:munge. SLURM authenticates with this; a mismatched key
   produces confusing "invalid credential" failures.
3. **Install `slurmd`**, drop in `slurm.conf` (identical on every node), enable
   the service.
4. **Install `node_exporter` and `dcgm-exporter`** as systemd units bound to the
   internal interface only.
5. **Firewall.** devon is publicly routable. Allow SSH; restrict 6818, 9100,
   9400 to the services host address. This is the single highest-value item on
   the list — a public GPU box is a target.
6. **Create matching UIDs** for the four teammates whose home directories are
   empty, and record them, so the second GPU server can be provisioned with the
   same IDs. Mismatched UIDs across nodes cause permission failures that are
   tedious to diagnose.

---

## 5. What I configure over SSH afterwards

No privilege needed for any of it:

- `sacct`/`sbatch` wrapper scripts in `scripts/`, replacing `run_b0_devon.sh`,
  so submission is uniform and the resource request is explicit.
- MLflow logging in `src/train/trainer.py`, alongside the existing run-directory
  writes rather than replacing them — the files stay the source of truth.
- **Backfill**: import every existing run into MLflow, including B0-denoise's
  three seeds, B0-v2, the 1.5b normalisation ablation, and the dehaze/derain
  demos, so history is present from day one rather than starting empty.
- `restic` backup script and its retention policy.
- Grafana dashboards and Alertmanager rules.
- A dataset checksum registry extending `scripts/reside_manifest.py`.

---

## 6. Backup and disaster recovery

**What is worth backing up is much smaller than the disk suggests**, and being
explicit about it is what makes 200 GB comfortable:

| | size | backed up? | why |
|---|---|---|---|
| `runs/` | 5.9 GB | **yes** | irreplaceable — days of GPU time |
| code, configs, reports | — | already in git | two remotes |
| `data/` public datasets | 15 GB | **no** — checksums only | re-downloadable; a manifest and hash detects corruption |
| AdaIR teacher checkpoints | ~0.6 GB | **yes** | released, but slow to re-fetch |
| other FYP work | 42 GB | selectively | depends what is reproducible |

Realistic first snapshot: **10–20 GB**, growing a few GB per experiment. The
80 GB restic repository holds years at that rate.

**Retention**, matching the earlier decision: keep `best.pth`, `last.pth`,
`history.json`, `metrics.json`, `config.yaml`, `git_commit.txt` and `train.log`
for every run; discard the ~30 intermediate checkpoints once a run completes.
About 150 MB per run rather than 2.5 GB.

**Schedule**: nightly `restic backup` from devon to the `backup` container over
the internal network, `restic forget --keep-daily 7 --keep-weekly 8 --keep-monthly 12`,
and a **weekly restore test** — an untested backup is a hypothesis, and this
project's own history says instruments that are never made to disagree tend not
to work.

**What this does and does not protect against.** It survives devon's disk
failing, the CPU being RMA'd, or the OS being rebuilt. It does **not** survive
the site — both machines are in the same building on the same power. If that
matters, the mitigation is a periodic `restic copy` to external storage; the
datasets are public so there is no data-policy obstacle, only the slow uni link.

---

## 7. Monitoring and alerting

**Dashboards** (Grafana, via SSH tunnel): GPU utilisation, temperature and
memory; CPU package temperature — which reached 84 °C against an 80 °C threshold
under four concurrent runs, while the GPU sat at 58 °C and 55 % of its power
limit, so the dataloader is the thermal constraint, not the GPU; disk free;
SLURM queue depth and job states.

**Email alerts**, deliberately few so they stay meaningful:

| alert | condition | why |
|---|---|---|
| job failed | SLURM state `FAILED`/`TIMEOUT`/`OOM` | a 20-hour run dying at 3 am |
| node down | `slurmd` unreachable > 10 min | distinguishes this from job failure — the 18-hour outage would have fired this once, not once per job |
| disk low | < 100 GB free on devon | training writes checkpoints; a full disk corrupts a run |
| GPU hot | > 85 °C sustained 10 min | |
| CPU hot | package > 90 °C sustained 10 min | the degraded silicon is voltage/heat sensitive |
| backup failed | no successful snapshot in 48 h | silent backup failure is the classic disaster-recovery hole |

Explicitly **not** alerted: job started, job finished normally, routine
temperature variation. An alert stream people learn to ignore is worse than none.

---

## 8. Risks

1. **devon is a single point of compute.** The RMA will take it offline. Nothing
   here changes that; it only ensures the *data and history* survive. The second
   GPU server is the real mitigation.
2. **Publicly routable GPU host.** The firewall step is the highest-value item in
   §4. MLflow and MinIO ship with weak or no auth by default and must never bind
   to the public interface.
3. **Password-gated sudo shared with others.** Every privileged change needs
   coordination. This makes the setup slower but is correct for a shared box.
4. **`unattended-upgrades` has already broken this machine once.** Pinning the
   driver is step 1 for a reason.
5. **Same-site backup.** Honest limitation, stated in §6.

---

## 9. Phasing

Each phase is independently useful; nothing later depends on finishing earlier
work perfectly.

| phase | delivers | needs |
|---|---|---|
| **1** | `backup` container, restic nightly + restore test | 1 container |
| **2** | `mlflow` + `minio`, backfill of all existing runs | 2 containers, no devon sudo |
| **3** | `slurm-ctl` + `slurmd`, submission wrappers | devon sudo session |
| **4** | `monitor`, dashboards, email alerts | 1 container + devon sudo |
| **5** | `ci-runner`, tests on push | 1 container |

**Phase 1 first, deliberately.** Right now every training run exists in exactly
one place on a machine with known-degraded hardware. That is the largest
unmitigated risk, and it is also the cheapest to close — one container and a
cron entry, no devon sudo at all.
