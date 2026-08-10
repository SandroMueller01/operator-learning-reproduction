"""Tests for the nonlinear mixed FEM NSB solver (paper eq. B.14-B.15).

Requires legacy FEniCS (dolfin); skipped automatically wherever dolfin is
not importable (see tests/test_diffusion_mixed_fenics_solver.py for the
same pattern).
"""

from __future__ import annotations

import numpy as np
import pytest

df = pytest.importorskip("dolfin")

from ol_reproduction.pde.diffusion.fenics_mixed_solver import (  # noqa: E402
    build_diffusion_mesh,
)
from ol_reproduction.pde.navier_stokes_brinkman.fenics_solver import (  # noqa: E402
    build_nsb_function_space,
    describe_nsb_mesh,
    solve_nsb_mixed_fenics,
)

RESOLUTION = 10  # small/fast resolution for unit tests


def test_function_space_dof_counts() -> None:
    mesh = build_diffusion_mesh(RESOLUTION)
    function_space, boundary_markers = build_nsb_function_space(mesh)
    info = describe_nsb_mesh(mesh, function_space, resolution=RESOLUTION)

    # u is DG0 vector: 2 DOF per cell.
    assert info.num_dofs_u == 2 * mesh.num_cells()
    assert info.num_dofs_total == function_space.dim()
    assert boundary_markers is not None


def test_solve_converges_and_has_zero_mean_pressure() -> None:
    mesh = build_diffusion_mesh(RESOLUTION)
    parameters = np.array([1.0, 0.0, 0.0, 0.0])

    result = solve_nsb_mixed_fenics(
        mesh=mesh,
        coefficient_name="affine",
        parameters=parameters,
    )

    assert result.converged
    assert result.newton_iterations > 0
    assert np.all(np.isfinite(result.u_dofs))
    assert np.all(np.isfinite(result.p_dofs))

    # Zero-mean pressure is enforced as an exact post-hoc shift.
    assert abs(result.p_dofs.mean()) < 1.0e-8


def test_log_coefficient_solve_also_converges() -> None:
    mesh = build_diffusion_mesh(RESOLUTION)
    parameters = np.array([0.3, -0.2, 0.5, 0.1])

    result = solve_nsb_mixed_fenics(
        mesh=mesh,
        coefficient_name="log",
        parameters=parameters,
    )

    assert result.converged
    assert np.all(np.isfinite(result.u_dofs))


def test_velocity_is_nonzero_away_from_walls() -> None:
    """Sanity check that the solve produces actual flow, not a trivial
    all-zero solution (e.g. from a silently-degenerate inlet BC)."""
    mesh = build_diffusion_mesh(RESOLUTION)
    parameters = np.array([1.0, 0.0, 0.0, 0.0])

    result = solve_nsb_mixed_fenics(
        mesh=mesh,
        coefficient_name="affine",
        parameters=parameters,
    )

    assert np.abs(result.u_dofs).max() > 0.1
