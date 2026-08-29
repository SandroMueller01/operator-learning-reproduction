#!/usr/bin/env bash
# Regenerate all 4 NSB cases with the corrected mesh/elements
# (data/original_mesh/poisson.xml, AFW BDM2/DG1vec/DG1/DG2vec3), the
# corrected constant inlet velocity, the corrected DG0 (no-mean-shift)
# pressure, the corrected inlet-term sign, and the corrected d=8
# sparse-grid level (4, 3937 points).
#
# Usage (inside WSL, from repo root):
#   bash scripts/wsl/run_all_nsb_cases.sh > /tmp/run_all_nsb_cases.log 2>&1 &

REPO_ROOT="/mnt/c/Users/sandr/Documents/JKU_Master_Artificial_Intelligence/operator-learning-reproduction"
cd "$REPO_ROOT"

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate fenics2019

run_case() {
    local case_name=$1
    local coefficient=$2
    local dimension=$3

    echo "=== Starting $case_name at $(date) ==="
    python scripts/wsl/generate_nsb_case.py --case-name "$case_name" --coefficient "$coefficient" --dimension "$dimension" --workers 28
    echo "=== Finished $case_name at $(date), exit code $? ==="
}

run_case nsb_affine_d4 affine 4
run_case nsb_affine_d8 affine 8
run_case nsb_log_d4 log 4
run_case nsb_log_d8 log 8

echo "=== All NSB cases complete at $(date) ==="
