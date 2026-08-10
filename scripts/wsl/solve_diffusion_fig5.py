"""Phase 3 milestone: reproduce the paper's Fig. 5 setup with the new mixed
FEM diffusion solver -- affine coefficient a_{1,4}, x=(1,0,0,0), and compare
the resulting u field visually against the paper figure."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import dolfin as df

from ol_reproduction.pde.diffusion.fenics_mixed_solver import (
    build_diffusion_mesh,
    solve_diffusion_mixed_fenics,
)

RESOLUTION = 23
PARAMETERS = np.array([1.0, 0.0, 0.0, 0.0])

mesh = build_diffusion_mesh(RESOLUTION)
result = solve_diffusion_mixed_fenics(
    mesh=mesh,
    coefficient_name="affine",
    parameters=PARAMETERS,
    forcing=10.0,
    base_value=2.62,
    resolution_for_metadata=RESOLUTION,
)

print("mesh_info:", result.mesh_info)
print("u_dofs shape:", result.u_dofs.shape)
print("u range: [%.6f, %.6f]" % (result.u_dofs.min(), result.u_dofs.max()))

# DG0 solution: one constant value per cell. Build a per-cell-centroid
# scatter/tripcolor plot (u is piecewise constant, so this renders it
# faithfully without needing to project to a nodal space).
coords = mesh.coordinates()
cells = mesh.cells()
centroids = coords[cells].mean(axis=1)

fig, ax = plt.subplots(figsize=(5, 5))
triangulation = mtri.Triangulation(coords[:, 0], coords[:, 1], cells)
tpc = ax.tripcolor(
    triangulation,
    facecolors=result.u_dofs,
    cmap="turbo",
    shading="flat",
)
fig.colorbar(tpc, ax=ax, label="u(x)")
ax.set_title(
    "Mixed FEM diffusion, affine a_1,4, x=(1,0,0,0)\n"
    f"K={result.mesh_info.num_dofs_total} DOF "
    f"(u:{result.mesh_info.num_dofs_u}, sigma:{result.mesh_info.num_dofs_sigma})"
)
ax.set_xlabel("z1")
ax.set_ylabel("z2")
fig.tight_layout()

output_path = Path(__file__).resolve().parents[2] / "results" / "figures" / "phase3_diffusion_fig5_check.png"
output_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output_path, dpi=150)
print("Saved:", output_path)
