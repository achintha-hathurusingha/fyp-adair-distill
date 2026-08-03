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
exec taskset -c 0-7,12-31 python -m src.train.train "$@"
