"""Phase 6: measure per-solve timing for the mixed FEM diffusion solver,
to inform the full 8-case x 12-trial x 14-m sweep resource estimate."""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ol_reproduction.data.sampling import sample_uniform_parameters
from ol_reproduction.pde.diffusion.fenics_mixed_solver import (
    build_diffusion_mesh,
    solve_diffusion_mixed_fenics,
)

RESOLUTION = 23
N_WARMUP = 2
N_TIMED = 30

mesh = build_diffusion_mesh(RESOLUTION)
parameters = sample_uniform_parameters(num_samples=N_WARMUP + N_TIMED, dimension=4, seed=0)

for i in range(N_WARMUP):
    solve_diffusion_mixed_fenics(
        mesh=mesh, coefficient_name="affine", parameters=parameters[i],
    )

times = []
for i in range(N_WARMUP, N_WARMUP + N_TIMED):
    start = time.perf_counter()
    solve_diffusion_mixed_fenics(
        mesh=mesh, coefficient_name="affine", parameters=parameters[i],
    )
    times.append(time.perf_counter() - start)

times = np.array(times)
print(f"n={len(times)} mean={times.mean():.4f}s median={np.median(times):.4f}s "
      f"min={times.min():.4f}s max={times.max():.4f}s")

# Estimate full sweep: each seed solves max(TRAIN_SIZES)=500 points once
# (sliced into the 14 m-value files), x 12 seeds, + 1105 shared test points
# (d=4, level=5).
train_solves = 500 * 12
test_solves = 1105
total_solves = train_solves + test_solves
est_seconds = total_solves * np.median(times)
print(f"Estimated time for one full diffusion_affine_d4 case: "
      f"{total_solves} solves, {est_seconds:.1f}s ({est_seconds/60:.1f} min)")
