#!/usr/bin/env bash
# The reservoir comparison on one task: every readout source and MEA size, in parallel. Logs in outputs/logs/.
set -u
task=${1:-stand_push}
seed=${2:-0}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
mkdir -p outputs/logs
run() { uv run scripts/train_reservoir.py --task "$task" --seed "$seed" --threads 2 "$@" \
          > "outputs/logs/${task}_$(echo "$@" | tr -d ' -')_s${seed}.log" 2>&1 & }
run --source none
run --source mea --mea mcs60
run --source dead --mea mcs60
run --source mea --mea grid256
run --source mea --mea hd1024
run --source oracle
wait
