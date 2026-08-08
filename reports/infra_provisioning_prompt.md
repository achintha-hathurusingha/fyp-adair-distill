# Provisioning request — paste into the infrastructure session

Everything below is a request for **container creation only**. Application
configuration is done afterwards over SSH and is not part of this request.

---

## Context

A research group needs a small services host for ML experiment infrastructure.
The GPU machine (`devon`, `192.248.10.68`, Ubuntu 24.04, RTX 4090) already
exists and is **not** part of this request — it stays as-is and only gains
client packages later.

Please create **six unprivileged LXC containers** on the separate physical host.
Ubuntu 24.04 for all of them.

## Containers

| name | vCPU | RAM | disk | purpose |
|---|---|---|---|---|
| `slurm-ctl` | 2 | 2 GB | 20 GB | SLURM controller + accounting DB |
| `mlflow` | 2 | 4 GB | 20 GB | MLflow tracking server + PostgreSQL |
| `minio` | 2 | 4 GB | **100 GB** | S3-compatible artifact store |
| `monitor` | 2 | 4 GB | 30 GB | Prometheus + Grafana + Alertmanager |
| `backup` | 1 | 2 GB | **80 GB** | restic backup repository |
| `ci-runner` | 4 | 8 GB | 40 GB | self-hosted GitHub Actions runner |

Total: 13 vCPU, 24 GB RAM, 290 GB disk. **If 290 GB is not available**, reduce
`minio` to 60 GB and `backup` to 50 GB (190 GB total) — those two are the only
ones that scale with usage, and both can be grown later.

## Networking

- All containers on the **same internal bridge**, able to reach each other and
  `devon` at `192.248.10.68`.
- **Static internal IPs**, and please report which address each container got.
- **No port forwarding from any public interface.** Access is via SSH tunnel
  only. This is a hard requirement: `devon` is publicly routable, and MLflow,
  MinIO, Grafana and the SLURM controller all ship with weak or absent
  authentication by default. An exposed MLflow or Ray-style endpoint on a host
  with a GPU behind it is an actively exploited target.
- `ci-runner` needs **outbound** internet (to reach GitHub); the others need
  outbound only for package installation.

## Access

- SSH into each container as a non-root user with sudo, or provide root — either
  is fine, please say which.
- Please add this public key for access:
  *(paste `~/.ssh/id_ed25519.pub` from the laptop here)*

## Storage notes

- `minio` and `backup` hold the only off-machine copies of multi-day training
  results, so please put them on the most reliable storage available, and say
  whether the underlying pool has redundancy (RAID/ZFS mirror) or is a single
  disk. That determines whether a further copy is needed.
- The other four containers hold configuration and metrics that are cheap to
  rebuild.

## Please report back

1. Internal IP of each container
2. SSH access details
3. Whether the storage pool is redundant
4. Any deviation from the requested sizes

## Not part of this request

Application installation and configuration — SLURM, MLflow, MinIO, Prometheus,
Grafana, restic, the GitHub runner — are all done afterwards over SSH.
