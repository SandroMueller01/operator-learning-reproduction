#!/usr/bin/env bash
# Phase 7: generate the remaining 3 NSB cases, sequentially, each using a
# 28-way process pool internally (generate_nsb_case.py --workers) -- the
# original fully-serial design (one solve at a time) was ~89x slower than
# this in practice on the nsb_affine_d4 case, so cases run one at a time
# but each case's own solves run in parallel across WSL2's cores.
#
# All 4 diffusion cases and nsb_affine_d4 are already generated; this
# script only covers the remaining 3 NSB cases.
#
# Usage (inside WSL, fenics2019 env active, from repo root):
#   bash scripts/wsl/run_remaining_cases.sh > /tmp/run_remaining_cases.log 2>&1 &
# Deliberately no "set -e"/"set -u": one case's failure shouldn't abort the
# rest, and conda's own activation scripts are not nounset-safe (activating
# fenics2019 trips "ADDR2LINE: unbound variable" under "set -u").

REPO_ROOT="/mnt/c/Users/sandr/Documents/JKU_Master_Artificial_Intelligence/operator-learning-reproduction"
cd "$REPO_ROOT"

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate fenics2019

run_case() {
    local script=$1
    local case_name=$2
    local coefficient=$3
    local dimension=$4

    echo "=== Starting $case_name at $(date) ==="
    python "$script" --case-name "$case_name" --coefficient "$coefficient" --dimension "$dimension" --workers 28
    echo "=== Finished $case_name at $(date), exit code $? ==="
}

run_case scripts/wsl/generate_nsb_case.py nsb_affine_d8 affine 8
run_case scripts/wsl/generate_nsb_case.py nsb_log_d4 log 4
run_case scripts/wsl/generate_nsb_case.py nsb_log_d8 log 8

echo "=== All remaining cases complete at $(date) ==="
