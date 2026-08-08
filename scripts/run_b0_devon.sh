#!/usr/bin/env bash
# B0 on devon, pinned to the HEALTHY CPUs.
#
# devon is an i9-14900K with the Raptor Lake elevated-voltage defect. The damage
# is confined to physical cores 4 and 5 (logical CPUs 8-11) -- the two 6.0 GHz
# Thermal Velocity Boost favoured cores, i.e. the ones that ran at the highest
# voltage. Per-core pytest: cores 0,1,2,3,6,7 pass 4/4 each; cores 4,5 fail 3/4
# and 2/4. The healthy set below passes 10/10; CPUs 8-11 alone fail 3/6.
#
# taskset covers the whole process tree, dataloader workers included. Removing
# it reintroduces segfaults. See memory: devon-hardware-instability.
set -euo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
conda activate adair-distill
cd ~/fyp-adair-distill

# Optional MLflow/MinIO credentials, sourced from OUTSIDE the repo and never
# committed. src.utils.tracking.RunTracker is best-effort either way -- a run
# launched without this file trains identically, just with tracking disabled.
# See ~/.mlflow_credentials.env.example for the expected shape.
if [ -f "$HOME/.mlflow_credentials.env" ]; then
  source "$HOME/.mlflow_credentials.env"
fi

exec taskset -c 0-7,12-31 python -m src.train.train "$@"
