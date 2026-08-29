"""Phase 7: measure per-solve timing for the NSB mixed FEM solver, and
report Newton non-convergence rate, to inform the full sweep resource
estimate and retry-policy design."""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ol_reproduction.data.sampling import sample_uniform_parameters
from ol_reproduction.pde.diffusion.fenics_mixed_solver import load_original_mesh
from ol_reproduction.pde.navier_stokes_brinkman.fenics_solver import (
    solve_nsb_mixed_fenics,
)

N_WARMUP = 2
N_TIMED = 20

mesh = load_original_mesh()
parameters = sample_uniform_parameters(num_samples=N_WARMUP + N_TIMED, dimension=4, seed=123)

for i in range(N_WARMUP):
    solve_nsb_mixed_fenics(mesh=mesh, coefficient_name="affine", parameters=parameters[i])

times = []
non_converged = 0
for i in range(N_WARMUP, N_WARMUP + N_TIMED):
    start = time.perf_counter()
    result = solve_nsb_mixed_fenics(mesh=mesh, coefficient_name="affine", parameters=parameters[i])
    times.append(time.perf_counter() - start)
    if not result.converged:
        non_converged += 1

times = np.array(times)
print(f"n={len(times)} mean={times.mean():.4f}s median={np.median(times):.4f}s "
      f"min={times.min():.4f}s max={times.max():.4f}s non_converged={non_converged}/{N_TIMED}")

train_solves = 500 * 12  # each seed solves max(TRAIN_SIZES)=500 once, sliced for smaller m
test_solves = 1105
total_solves = train_solves + test_solves
est_seconds = total_solves * np.median(times)
print(f"Estimated time for one full nsb_affine_d4 case (u+p together per solve): "
      f"{total_solves} solves, {est_seconds:.1f}s ({est_seconds/3600:.2f} h)")
