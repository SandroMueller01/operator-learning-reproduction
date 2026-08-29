"""Phase 4 milestone: reproduce the paper's Fig. 6 setup with the new
nonlinear mixed FEM NSB solver -- affine coefficient a_{1,4}, x=(1,0,0,0),
same mesh as the diffusion case, and compare u/p visually against the
paper figure."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import dolfin as df

from ol_reproduction.pde.diffusion.fenics_mixed_solver import load_original_mesh
from ol_reproduction.pde.navier_stokes_brinkman.fenics_solver import (
    build_nsb_function_space,
    solve_nsb_mixed_fenics,
)

FE_DEGREE = 1
PARAMETERS = np.array([1.0, 0.0, 0.0, 0.0])

mesh = load_original_mesh()
result = solve_nsb_mixed_fenics(
    mesh=mesh,
    coefficient_name="affine",
    parameters=PARAMETERS,
    base_value=2.62,
    fe_degree=FE_DEGREE,
)

print("converged:", result.converged, "newton_iterations:", result.newton_iterations)
print("mesh_info:", result.mesh_info)
print("u_dofs shape:", result.u_dofs.shape)
print("p_dofs shape:", result.p_dofs.shape)

# Reconstruct Functions on the exact same spaces build_nsb_function_space
# used (u is field index 0, per the authors' [Hu,Ht,Hsig,Hsig,Hgam] order)
# so the raw dof arrays plot correctly rather than assuming an ordering.
coords = mesh.coordinates()
cells = mesh.cells()
centroids = coords[cells].mean(axis=1)

W, _ = build_nsb_function_space(mesh, fe_degree=FE_DEGREE)
u_space = W.sub(0).collapse()
u_func = df.Function(u_space)
u_func.vector()[:] = result.u_dofs

p_space = df.FunctionSpace(mesh, "DG", 0)
p_func = df.Function(p_space)
p_func.vector()[:] = result.p_dofs

integral_p = df.assemble(p_func * df.dx)
print("integral of p over Omega:", integral_p)

u_at_centroids = np.array([u_func(pt) for pt in centroids])
p_at_centroids = np.array([p_func(pt) for pt in centroids])

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
triangulation = mtri.Triangulation(coords[:, 0], coords[:, 1], cells)

axes[0].quiver(
    centroids[:, 0], centroids[:, 1],
    u_at_centroids[:, 0], u_at_centroids[:, 1],
    np.linalg.norm(u_at_centroids, axis=1),
    cmap="turbo", scale=20,
)
axes[0].set_title("Velocity field u")
axes[0].set_xlabel("z1")
axes[0].set_ylabel("z2")

tpc = axes[1].tripcolor(triangulation, facecolors=p_at_centroids, cmap="turbo", shading="flat")
fig.colorbar(tpc, ax=axes[1], label="p(x)")
axes[1].set_title("Pressure p")
axes[1].set_xlabel("z1")

fig.suptitle(
    f"Mixed FEM NSB, affine a_1,4, x=(1,0,0,0), converged={result.converged}, "
    f"integral(p)={integral_p:.2e}"
)
fig.tight_layout()

output_path = Path(__file__).resolve().parents[2] / "results" / "figures" / "phase4_nsb_fig6_check.png"
output_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output_path, dpi=150)
print("Saved:", output_path)
