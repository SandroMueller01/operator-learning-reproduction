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
    load_original_mesh,
    solve_diffusion_mixed_fenics,
)

FE_DEGREE = 1
PARAMETERS = np.array([1.0, 0.0, 0.0, 0.0])

mesh = load_original_mesh()
result = solve_diffusion_mixed_fenics(
    mesh=mesh,
    coefficient_name="affine",
    parameters=PARAMETERS,
    forcing=10.0,
    base_value=2.62,
    fe_degree=FE_DEGREE,
)

print("mesh_info:", result.mesh_info)
print("u_dofs shape:", result.u_dofs.shape)
print("u range: [%.6f, %.6f]" % (result.u_dofs.min(), result.u_dofs.max()))

# u is DG1 (piecewise linear, discontinuous across cells), so reconstruct
# the Function and project onto a CG1 space purely for a smooth-looking
# visualization -- the projection has no effect on result.u_dofs itself.
u_element = df.FiniteElement("DG", mesh.ufl_cell(), FE_DEGREE)
u_space = df.FunctionSpace(mesh, u_element)
u_func = df.Function(u_space)
u_func.vector()[:] = result.u_dofs
u_cg = df.project(u_func, df.FunctionSpace(mesh, "CG", 1))

coords = mesh.coordinates()
cells = mesh.cells()
vertex_values = u_cg.compute_vertex_values(mesh)

fig, ax = plt.subplots(figsize=(5, 5))
triangulation = mtri.Triangulation(coords[:, 0], coords[:, 1], cells)
tpc = ax.tripcolor(
    triangulation,
    vertex_values,
    cmap="turbo",
    shading="gouraud",
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
