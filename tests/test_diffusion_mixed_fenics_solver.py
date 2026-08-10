"""Tests for the mixed FEM diffusion solver (paper eq. B.9-B.12).

These tests require legacy FEniCS (dolfin), which only runs inside the WSL
'fenics2019' conda environment (see scripts/wsl/) -- they are skipped
automatically wherever dolfin is not importable (e.g. the Windows-side
venv used for the rest of the test suite).
"""

from __future__ import annotations

import numpy as np
import pytest

df = pytest.importorskip("dolfin")

from ol_reproduction.pde.diffusion.fenics_mixed_solver import (  # noqa: E402
    build_diffusion_mesh,
    calibrate_mesh_resolution,
    describe_mesh,
    solve_diffusion_mixed_fenics,
)

RESOLUTION = 12  # small/fast resolution for unit tests


def test_mesh_dof_counts_match_element_theory() -> None:
    """RT1 has one DOF per edge, DG0 has one DOF per cell."""
    mesh = build_diffusion_mesh(RESOLUTION)
    info = describe_mesh(mesh, resolution=RESOLUTION)

    assert info.num_dofs_u == mesh.num_cells()
    assert info.num_dofs_sigma == mesh.num_edges()
    assert info.num_dofs_total == info.num_dofs_sigma + info.num_dofs_u


def test_solve_produces_finite_solution_in_expected_range() -> None:
    """u should stay close to [0, 0.5], the range of the Dirichlet data."""
    mesh = build_diffusion_mesh(RESOLUTION)
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
    # DG0 solution can slightly overshoot the boundary data near the
    # boundary itself, so allow a small tolerance rather than a hard [0, 0.5].
    assert result.u_dofs.min() > -0.05
    assert result.u_dofs.max() < 0.55


def test_solution_is_largest_near_bottom_boundary() -> None:
    """u=0.5 on the bottom edge, u=0 elsewhere: solution should peak low z2."""
    mesh = build_diffusion_mesh(RESOLUTION)
    parameters = np.array([1.0, 0.0, 0.0, 0.0])

    result = solve_diffusion_mixed_fenics(
        mesh=mesh,
        coefficient_name="affine",
        parameters=parameters,
    )

    cells = mesh.cells()
    coords = mesh.coordinates()
    centroid_z2 = coords[cells].mean(axis=1)[:, 1]

    low_z2_mean = result.u_dofs[centroid_z2 < 0.2].mean()
    high_z2_mean = result.u_dofs[centroid_z2 > 0.8].mean()

    assert low_z2_mean > high_z2_mean


def test_log_coefficient_solve_also_succeeds() -> None:
    mesh = build_diffusion_mesh(RESOLUTION)
    parameters = np.array([0.3, -0.2, 0.5, 0.1])

    result = solve_diffusion_mixed_fenics(
        mesh=mesh,
        coefficient_name="log",
        parameters=parameters,
    )

    assert np.all(np.isfinite(result.u_dofs))


def test_calibrate_mesh_resolution_returns_info_per_resolution() -> None:
    results = calibrate_mesh_resolution(resolutions=(8, 10))

    assert [info.resolution for info in results] == [8, 10]
    for info in results:
        assert info.num_dofs_total > 0
