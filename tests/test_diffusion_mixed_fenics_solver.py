"""Tests for the mixed FEM diffusion solver (paper eq. B.9-B.12).

These tests require legacy FEniCS (dolfin), which only runs inside the WSL
'fenics2019' conda environment (see scripts/wsl/) -- they are skipped
automatically wherever dolfin is not importable (e.g. the Windows-side
venv used for the rest of the test suite).

Uses the paper authors' own mesh (``load_original_mesh``, 244 cells) with
their BDM2 (sigma) x DG1 (u) element pair -- not a synthetic
structured mesh -- so the DOF counts below are exact, verified numbers
(2622 total: sigma 1890, u 732), not formula-derived estimates.
"""

from __future__ import annotations

import numpy as np
import pytest

df = pytest.importorskip("dolfin")

from ol_reproduction.pde.diffusion.fenics_mixed_solver import (  # noqa: E402
    build_diffusion_mesh,
    calibrate_mesh_resolution,
    describe_mesh,
    load_original_mesh,
    solve_diffusion_mixed_fenics,
)


def test_mesh_dof_counts_match_the_paper_authors_figures() -> None:
    """BDM2 x DG1 on the real mesh: 1890 (sigma) + 732 (u) = 2622 total."""
    mesh = load_original_mesh()
    info = describe_mesh(mesh, resolution=0, fe_degree=1)

    assert mesh.num_cells() == 244
    assert info.num_dofs_u == 732
    assert info.num_dofs_sigma == 1890
    assert info.num_dofs_total == 2622
    assert info.num_dofs_total == info.num_dofs_sigma + info.num_dofs_u


def test_solve_produces_finite_solution_in_expected_range() -> None:
    """u should stay close to [0, 0.5], the range of the Dirichlet data."""
    mesh = load_original_mesh()
    parameters = np.array([1.0, 0.0, 0.0, 0.0])

    result = solve_diffusion_mixed_fenics(
        mesh=mesh,
        coefficient_name="affine",
        parameters=parameters,
        forcing=10.0,
        base_value=2.62,
    )

    assert np.all(np.isfinite(result.u_dofs))
    assert np.all(np.isfinite(result.sigma_dofs))
    # A piecewise-linear (DG1) solution can slightly overshoot the boundary
    # data near the boundary itself, so allow a small tolerance rather than
    # a hard [0, 0.5].
    assert result.u_dofs.min() > -0.2
    assert result.u_dofs.max() < 0.7


def test_solution_is_largest_near_bottom_boundary() -> None:
    """u=0.5 on the bottom edge, u=0 elsewhere: solution should peak low z2."""
    mesh = load_original_mesh()
    parameters = np.array([1.0, 0.0, 0.0, 0.0])

    result = solve_diffusion_mixed_fenics(
        mesh=mesh,
        coefficient_name="affine",
        parameters=parameters,
    )

    u_element = df.FiniteElement("DG", mesh.ufl_cell(), 1)
    u_space = df.FunctionSpace(mesh, u_element)
    u_func = df.Function(u_space)
    u_func.vector()[:] = result.u_dofs
    u_cg = df.project(u_func, df.FunctionSpace(mesh, "CG", 1))

    coords = mesh.coordinates()
    vertex_values = u_cg.compute_vertex_values(mesh)
    low_z2_mean = vertex_values[coords[:, 1] < 0.2].mean()
    high_z2_mean = vertex_values[coords[:, 1] > 0.8].mean()

    assert low_z2_mean > high_z2_mean


def test_log_coefficient_solve_also_succeeds() -> None:
    mesh = load_original_mesh()
    parameters = np.array([0.3, -0.2, 0.5, 0.1])

    result = solve_diffusion_mixed_fenics(
        mesh=mesh,
        coefficient_name="log",
        parameters=parameters,
    )

    assert np.all(np.isfinite(result.u_dofs))


def test_calibrate_mesh_resolution_returns_info_per_resolution() -> None:
    """Diagnostic-only fallback for the structured-mesh smoke test; not
    used for paper-scale work (which uses load_original_mesh instead)."""
    results = calibrate_mesh_resolution(resolutions=(8, 10))

    assert [info.resolution for info in results] == [8, 10]
    for info in results:
        assert info.num_dofs_total > 0


def test_build_diffusion_mesh_smoke_test_fallback_still_works() -> None:
    """The structured-mesh fallback (mshr-free) should still build and
    solve, even though paper-scale generation no longer uses it."""
    mesh = build_diffusion_mesh(10)
    parameters = np.array([1.0, 0.0, 0.0, 0.0])

    result = solve_diffusion_mixed_fenics(
        mesh=mesh,
        coefficient_name="affine",
        parameters=parameters,
    )

    assert np.all(np.isfinite(result.u_dofs))
