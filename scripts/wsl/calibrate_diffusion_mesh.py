"""Phase 3: search mshr resolutions for a mesh close to the paper's Fig. 4
(K=2622 total DOF, h_min=0.0844, h_max=0.1146)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ol_reproduction.pde.diffusion.fenics_mixed_solver import (
    calibrate_mesh_resolution,
)

TARGET_DOFS = 2622
TARGET_HMIN = 0.0844
TARGET_HMAX = 0.1146

results = calibrate_mesh_resolution(resolutions=tuple(range(14, 30)))

print(f"{'res':>4} {'cells':>6} {'hmin':>8} {'hmax':>8} {'dof_sigma':>10} {'dof_u':>7} {'total':>7} {'|dof-target|':>13}")
best = None
for info in results:
    diff = abs(info.num_dofs_total - TARGET_DOFS)
    if best is None or diff < best[0]:
        best = (diff, info)
    print(
        f"{info.resolution:>4} {info.num_cells:>6} {info.h_min:>8.4f} "
        f"{info.h_max:>8.4f} {info.num_dofs_sigma:>10} {info.num_dofs_u:>7} "
        f"{info.num_dofs_total:>7} {diff:>13}"
    )

print()
print("Closest to target K=2622:", best[1])
