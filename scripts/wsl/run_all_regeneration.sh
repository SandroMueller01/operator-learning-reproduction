#!/usr/bin/env bash
# Regenerate diffusion_affine_d4 (the one case interrupted by a WSL VM
# restart) and then all 4 NSB cases, sequentially. No "set -e": one case's
# failure shouldn't abort the rest.
#
# Run this via a single foreground `wsl -d Ubuntu-22.04 -- bash -lc` call
# wrapped in the Bash tool's run_in_background:true -- NOT via a manually
# nohup/disowned inner background job. A detached inner job only survives
# the *shell* exiting; it does not survive the WSL2 VM itself being torn
# down, which happens once no wsl.exe process stays attached. Keeping this
# script itself as the thing run_in_background tracks keeps the VM alive
# for the whole duration.

REPO_ROOT="/mnt/c/Users/sandr/Documents/JKU_Master_Artificial_Intelligence/operator-learning-reproduction"
cd "$REPO_ROOT"

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate fenics2019

run_diffusion_case() {
    local case_name=$1
    local coefficient=$2
    local dimension=$3
    echo "=== Starting $case_name at $(date) ==="
    python scripts/wsl/generate_diffusion_case.py --case-name "$case_name" --coefficient "$coefficient" --dimension "$dimension"
    echo "=== Finished $case_name at $(date), exit code $? ==="
}

run_nsb_case() {
    local case_name=$1
    local coefficient=$2
    local dimension=$3
    echo "=== Starting $case_name at $(date) ==="
    python scripts/wsl/generate_nsb_case.py --case-name "$case_name" --coefficient "$coefficient" --dimension "$dimension" --workers 28
    echo "=== Finished $case_name at $(date), exit code $? ==="
}

run_diffusion_case diffusion_affine_d4 affine 4

run_nsb_case nsb_affine_d4 affine 4
run_nsb_case nsb_affine_d8 affine 8
run_nsb_case nsb_log_d4 log 4
run_nsb_case nsb_log_d8 log 8

echo "=== ALL REGENERATION COMPLETE at $(date) ==="
