"""Mixed FEM solver for the parametric diffusion equation (paper eq. B.9-B.12).

Unlike ``fenics_solver.py`` (a standard primal CG1 weak form for the same
PDE), this module implements the paper's actual *mixed* variational
formulation: find ``(sigma, u) in H(div; Omega) x L^2(Omega)`` such that

    d_a(sigma, tau) + b(tau, u) = G(tau)   for all tau in H(div; Omega)
    b(sigma, v)                 = F(v)     for all v in L^2(Omega)

where ``d_a(sigma, tau) = int (sigma . tau) / a(x)``,
``b(tau, v) = int div(tau) v``, ``F(v) = -int f v``, and
``G(tau) = int_{dOmega} g (tau . n) ds`` lifts the Dirichlet data ``g`` for
``u`` into a *natural* boundary term (paper Remark: BC (2.7)-(B.9) has no
Neumann portion, so the entire boundary contributes to ``G`` and no strong
``DirichletBC`` is imposed on either subspace).

The element pair is the standard lowest-order mixed-Poisson choice: RT1 for
``sigma`` (H(div)-conforming) and DG0 for ``u`` (L^2-conforming). The paper
does not state its element family explicitly, only the resulting mesh
statistics (K=2622 total DOF, h_min=0.0844, h_max=0.1146 on a visibly
non-uniform triangulation, see Fig. 4) -- RT1 x DG0 is the standard starting
hypothesis, calibrated against those numbers via ``build_diffusion_mesh``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ol_reproduction.coefficients.affine import affine_coefficient
from ol_reproduction.coefficients.log_transformed import (
    log_transformed_coefficient,
)

try:
    import dolfin as df
except ImportError as error:  # pragma: no cover - exercised only without FEniCS
    df = None
    _FENICS_IMPORT_ERROR = error
else:
    _FENICS_IMPORT_ERROR = None

try:
    import mshr
except ImportError:  # pragma: no cover - mshr has known conda-forge ABI issues
    mshr = None


def _require_fenics() -> None:
    """Raise a clear error if legacy FEniCS (dolfin) is unavailable."""
    if df is None:
        raise ImportError(
            "dolfin (legacy FEniCS 2019.1.0) is required for the mixed "
            "diffusion solver. Run this module inside the WSL 'fenics2019' "
            f"conda environment (see scripts/wsl/). Original import error: "
            f"{_FENICS_IMPORT_ERROR}"
        )


@dataclass(frozen=True)
class DiffusionMeshInfo:
    """Mesh statistics recorded for comparison against the paper's Fig. 4."""

    resolution: int
    num_cells: int
    h_min: float
    h_max: float
    num_dofs_sigma: int
    num_dofs_u: int
    num_dofs_total: int


@dataclass(frozen=True)
class MixedDiffusionResult:
    """Result of one mixed FEM diffusion solve."""

    u_dofs: np.ndarray
    sigma_dofs: np.ndarray
    mesh_info: DiffusionMeshInfo


def build_diffusion_mesh(resolution: int):
    """Build a triangular mesh of (0, 1)^2.

    The paper's mesh (Fig. 4) is visibly non-uniform (h_min != h_max),
    suggesting an unstructured (e.g. mshr-generated) triangulation. This
    repo's conda-forge ``mshr`` 2019.1.0 build has a known pybind11 ABI
    incompatibility with the paired ``fenics`` 2019.1.0 build in this WSL
    environment (``ImportError: generic_type: type "CSGGeometry"
    referenced unknown base type "dolfin::Variable"`` -- reproducible via
    ``scripts/wsl/test_mshr_import.py``, not fixed by reinstalling via
    conda or mamba). If ``mshr`` is importable, it is used; otherwise this
    falls back to a structured ``UnitSquareMesh``, which is uniform
    (h_min == h_max) but whose ``resolution`` can still be calibrated to
    match the paper's total DOF count K=2622 -- see
    ``calibrate_mesh_resolution``. The paper does not state its exact mesh
    generation method, so this is a documented, deliberate approximation
    (see Phase 3 open risks), not a silent one.
    """
    _require_fenics()

    if mshr is not None:
        domain = mshr.Rectangle(df.Point(0.0, 0.0), df.Point(1.0, 1.0))
        return mshr.generate_mesh(domain, resolution)

    return df.UnitSquareMesh(resolution, resolution)


def describe_mesh(mesh, resolution: int) -> DiffusionMeshInfo:
    """Compute mesh/DOF statistics for a given mixed function space."""
    _require_fenics()

    sigma_element = df.FiniteElement("RT", mesh.ufl_cell(), 1)
    u_element = df.FiniteElement("DG", mesh.ufl_cell(), 0)
    mixed_space = df.FunctionSpace(mesh, df.MixedElement([sigma_element, u_element]))

    num_dofs_sigma = mixed_space.sub(0).dim()
    num_dofs_u = mixed_space.sub(1).dim()

    return DiffusionMeshInfo(
        resolution=resolution,
        num_cells=mesh.num_cells(),
        h_min=mesh.hmin(),
        h_max=mesh.hmax(),
        num_dofs_sigma=num_dofs_sigma,
        num_dofs_u=num_dofs_u,
        num_dofs_total=num_dofs_sigma + num_dofs_u,
    )


def _make_coefficient_expression(
    coefficient_name: str,
    parameters: np.ndarray,
    base_value: float,
):
    """Build a FEniCS UserExpression evaluating a(z, x) at each mesh point.

    Reuses the exact tested NumPy coefficient functions
    (``affine_coefficient`` / ``log_transformed_coefficient``) pointwise, so
    the FEniCS coefficient can never drift from the NumPy one used by the
    finite-difference pipeline and its unit tests.
    """
    _require_fenics()

    parameters = np.asarray(parameters, dtype=float)
    name = coefficient_name.lower()

    if name == "affine":

        class _AffineExpression(df.UserExpression):
            def eval(self, value, point) -> None:
                z1 = np.array([point[0]])
                value[0] = float(
                    affine_coefficient(
                        z1=z1,
                        parameters=parameters,
                        base_value=base_value,
                    )[0]
                )

            def value_shape(self):
                return ()

        return _AffineExpression(degree=5)

    if name == "log":

        class _LogExpression(df.UserExpression):
            def eval(self, value, point) -> None:
                z1 = np.array([point[0]])
                value[0] = float(
                    log_transformed_coefficient(
                        z1=z1,
                        parameters=parameters,
                    )[0]
                )

            def value_shape(self):
                return ()

        return _LogExpression(degree=5)

    raise ValueError(f"Unsupported coefficient: {coefficient_name}")


def _make_boundary_expression(bottom_value: float, top: float, left: float, right: float):
    """Build the Dirichlet data g(z), applied as a natural boundary term.

    g = bottom_value on (0,1) x {0}, and ``top``/``left``/``right`` on the
    remaining three edges (paper: top=left=right=0).
    """
    _require_fenics()

    tol = 1.0e-10

    class _BoundaryExpression(df.UserExpression):
        def eval(self, value, point) -> None:
            if point[1] < tol:
                value[0] = bottom_value
            elif point[1] > 1.0 - tol:
                value[0] = top
            elif point[0] < tol:
                value[0] = left
            elif point[0] > 1.0 - tol:
                value[0] = right
            else:
                value[0] = 0.0

        def value_shape(self):
            return ()

    return _BoundaryExpression(degree=2)


def solve_diffusion_mixed_fenics(
    mesh,
    coefficient_name: str,
    parameters: np.ndarray,
    forcing: float = 10.0,
    base_value: float = 2.62,
    boundary_values: dict[str, float] | None = None,
    resolution_for_metadata: int = 0,
) -> MixedDiffusionResult:
    """Solve the mixed FEM diffusion problem (paper eq. B.9-B.12) once.

    Parameters
    ----------
    mesh:
        A mesh built by ``build_diffusion_mesh`` (or any dolfin mesh of
        ``(0, 1)^2``). Passed in (rather than built internally) so callers
        solving many samples for the same case reuse one fixed mesh, as the
        paper's protocol requires (same discretization for every m/trial).
    coefficient_name:
        ``"affine"`` or ``"log"``.
    parameters:
        Parameter vector ``x`` of shape ``(d,)``.
    forcing:
        Constant forcing term f (paper: f=10).
    base_value:
        Affine coefficient base value (paper: 2.62). Ignored for ``"log"``.
    boundary_values:
        Optional override of ``{"bottom", "top", "left", "right"}``
        (paper defaults: bottom=0.5, others=0).
    resolution_for_metadata:
        mshr resolution used to build ``mesh``, recorded in the returned
        mesh info for provenance only.

    Returns
    -------
    MixedDiffusionResult
        The u and sigma DOF vectors plus mesh/DOF statistics.
    """
    _require_fenics()

    boundary_values = boundary_values or {}
    bottom_value = float(boundary_values.get("bottom", 0.5))
    top = float(boundary_values.get("top", 0.0))
    left = float(boundary_values.get("left", 0.0))
    right = float(boundary_values.get("right", 0.0))

    sigma_element = df.FiniteElement("RT", mesh.ufl_cell(), 1)
    u_element = df.FiniteElement("DG", mesh.ufl_cell(), 0)
    mixed_space = df.FunctionSpace(mesh, df.MixedElement([sigma_element, u_element]))

    sigma, u = df.TrialFunctions(mixed_space)
    tau, v = df.TestFunctions(mixed_space)
    normal = df.FacetNormal(mesh)

    a_coefficient = _make_coefficient_expression(
        coefficient_name=coefficient_name,
        parameters=parameters,
        base_value=base_value,
    )
    g_boundary = _make_boundary_expression(bottom_value, top, left, right)
    f_forcing = df.Constant(forcing)

    bilinear_form = (
        df.dot(sigma, tau) / a_coefficient * df.dx
        + df.div(tau) * u * df.dx
        + df.div(sigma) * v * df.dx
    )
    linear_form = (
        g_boundary * df.dot(tau, normal) * df.ds
        - f_forcing * v * df.dx
    )

    solution = df.Function(mixed_space)
    df.solve(
        bilinear_form == linear_form,
        solution,
        solver_parameters={"linear_solver": "lu"},
    )

    sigma_h, u_h = solution.split(deepcopy=True)

    mesh_info = describe_mesh(mesh, resolution=resolution_for_metadata)

    return MixedDiffusionResult(
        u_dofs=np.asarray(u_h.vector().get_local(), dtype=np.float64),
        sigma_dofs=np.asarray(sigma_h.vector().get_local(), dtype=np.float64),
        mesh_info=mesh_info,
    )


def calibrate_mesh_resolution(
    target_dofs: int = 2622,
    resolutions: tuple[int, ...] = tuple(range(10, 40)),
) -> list[DiffusionMeshInfo]:
    """Report mesh statistics across a range of mshr resolutions.

    Used offline to pick a resolution whose (h_min, h_max, K) are close to
    the paper's reported (0.0844, 0.1146, 2622) -- see module docstring.
    Does not modify any state; purely diagnostic.
    """
    _require_fenics()

    results = []
    for resolution in resolutions:
        mesh = build_diffusion_mesh(resolution)
        results.append(describe_mesh(mesh, resolution=resolution))

    return results
