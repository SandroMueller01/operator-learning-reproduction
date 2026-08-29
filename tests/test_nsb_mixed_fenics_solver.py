"""Tests for the nonlinear mixed FEM NSB solver (paper eq. B.14-B.15).

Requires legacy FEniCS (dolfin); skipped automatically wherever dolfin is
not importable (see tests/test_diffusion_mixed_fenics_solver.py for the
same pattern).

Uses the paper authors' own mesh (``load_original_mesh``, 244 cells) and
AFW elements (BDM2 sigma-rows, DG1 vector u, DG1 gamma, DG2 vector-dim-3
t) -- not a synthetic structured mesh -- so the DOF counts below are
exact, verified numbers matching the paper's own documented 1464 (u) /
244 (p) figures, not formula-derived estimates.
"""

from __future__ import annotations

import numpy as np
import pytest

df = pytest.importorskip("dolfin")

from ol_reproduction.pde.diffusion.fenics_mixed_solver import (  # noqa: E402
    load_original_mesh,
)
from ol_reproduction.pde.navier_stokes_brinkman.fenics_solver import (  # noqa: E402
    build_nsb_function_space,
    describe_nsb_mesh,
    solve_nsb_mixed_fenics,
)


def test_function_space_dof_counts_match_the_paper() -> None:
    mesh = load_original_mesh()
    function_space, boundary_markers = build_nsb_function_space(mesh)
    info = describe_nsb_mesh(mesh, function_space, resolution=0)

    assert mesh.num_cells() == 244
    assert info.num_dofs_u == 1464  # matches the paper's documented figure
    assert info.num_dofs_sigma == 3780  # 1890 per BDM2 row x 2 rows
    assert info.num_dofs_total == function_space.dim() == 10368
    assert boundary_markers is not None


def test_pressure_space_dof_count_matches_the_paper() -> None:
    """Pressure is DG0 (one dof per cell), matching the paper's documented
    244 pressure DOFs exactly -- not DG1 as previously assumed."""
    mesh = load_original_mesh()
    parameters = np.array([1.0, 0.0, 0.0, 0.0])

    result = solve_nsb_mixed_fenics(
        mesh=mesh,
        coefficient_name="affine",
        parameters=parameters,
    )

    assert result.p_dofs.shape == (244,)


def test_solve_converges() -> None:
    mesh = load_original_mesh()
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
    # No post-hoc mean-shift is applied (matches the authors' own code,
    # which never renormalizes p) -- so the mean is not expected to be zero.


def test_log_coefficient_solve_also_converges() -> None:
    mesh = load_original_mesh()
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
    mesh = load_original_mesh()
    parameters = np.array([1.0, 0.0, 0.0, 0.0])

    result = solve_nsb_mixed_fenics(
        mesh=mesh,
        coefficient_name="affine",
        parameters=parameters,
    )

    assert np.abs(result.u_dofs).max() > 0.01
