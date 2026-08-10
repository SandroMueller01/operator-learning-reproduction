"""Mixed FEM nonlinear solver for the Navier-Stokes-Brinkman equations
(paper eq. B.14-B.15).

Implements the non-augmented Banach-spaces mixed formulation of Gatica,
Nunez & Ruiz-Baier (2022, J. Numer. Math. 31(4):343-373) -- the paper's own
cited source for eq. (B.15) -- using AFW (Arnold-Falk-Winther) finite
elements at lowest order (ell=0), plus the arXiv paper's Nitsche-method
addition (eq. B.15 discussion) for the outlet's zero-Cauchy-stress
condition, which the base reference (all-Dirichlet) does not need.

Unknowns: pseudostress sigma in H(div_4/3), velocity u in L^4, strain rate
t in L^2_tr (trace-free), vorticity gamma in L^2_skew. Pressure is
recovered afterwards via p = -1/2 tr(sigma + u tensor u).

AFW elements at ell=0 (Section 4.4.3 of the reference paper):
    sigma_h in P_1(Omega) ^ H(div;Omega)   (i.e. BDM_1, per tensor row)
    u_h     in P_0(Omega)                  (DG0 vector)
    gamma_h in L^2_skew(Omega) ^ P_0(Omega)
    t_h     in P_1(Omega) ^ L^2_tr(Omega)

In 2D, a skew-symmetric tensor has exactly one independent scalar (the
off-diagonal entry), and a trace-free tensor has exactly three
independent scalars -- both are represented here via plain scalar DG
FunctionSpaces combined with ``as_matrix`` in the UFL forms, rather than
custom constrained finite elements, which is an exact (not approximate)
representation of L^2_skew(Omega) and L^2_tr(Omega) in two dimensions.

Open risk (documented, not silently resolved): the paper's own inlet
profile formula, u_D = (0.0625)^-1 * ((z2-0.5)(1-z2), 0) on the inlet
edge (0,1) x {1}, is written in terms of z2 -- but z2=1 is *constant* on
that edge, which would make the literal formula evaluate to zero
everywhere along the inlet (a degenerate, physically meaningless BC).
This is almost certainly a z1/z2 transcription slip in the paper (the
repo's own configs/navier_stokes_brinkman.yaml flags the same ambiguity).
This module substitutes z1 (the coordinate that actually varies along
the inlet edge) for the paper's printed z2, which is the only
substitution that makes the formula non-degenerate.
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


INLET_MARKER = 1
OUTLET_MARKER = 2
WALL_MARKER = 3

LAMBDA = 0.1  # paper: lambda = Re^-1 = 0.1
KAPPA = 1.0e4  # paper: Nitsche penalty for the outlet condition


def _require_fenics() -> None:
    """Raise a clear error if legacy FEniCS (dolfin) is unavailable."""
    if df is None:
        raise ImportError(
            "dolfin (legacy FEniCS 2019.1.0) is required for the NSB mixed "
            "solver. Run this module inside the WSL 'fenics2019' conda "
            f"environment (see scripts/wsl/). Original import error: "
            f"{_FENICS_IMPORT_ERROR}"
        )


def eta_field_expression():
    """Return eta(z) = 10 + z1^2 + z2^2 (paper: scaled inverse permeability)."""
    _require_fenics()
    return df.Expression("10.0 + x[0]*x[0] + x[1]*x[1]", degree=2)


def forcing_vector():
    """Return paper forcing f = (0, -1)."""
    _require_fenics()
    return df.Constant((0.0, -1.0))


def inlet_velocity_expression():
    """Return the paper's inlet velocity profile on the top edge z2=1.

    See module docstring: z1 is substituted for the paper's printed z2,
    which is constant (=1) on the inlet edge and would otherwise make the
    formula degenerate. The substitution is additionally reflected
    (z1 -> 1-z1 inside the paper's (.-0.5)(1-.) formula, which simplifies
    to z1*(0.5-z1) below) so the profile vanishes at z1=0 -- the corner
    shared with the wall boundary (u=0 there), where a strong-Dirichlet
    mismatch between the inlet and wall data otherwise produces a severe,
    spurious pressure spike (confirmed empirically: the non-reflected
    substitution gives u_D=-8 at that corner and a large corner artifact
    in the Fig. 6 comparison plot). The profile is nonzero at the other
    end (z1=1, the inlet/outlet corner), which is fine since the outlet
    has no Dirichlet velocity constraint to conflict with.
    """
    _require_fenics()
    return df.Expression(
        (
            "pow(0.0625, -1.0) * x[0] * (0.5 - x[0])",
            "0.0",
        ),
        degree=2,
    )


class _BoundaryMarkers:
    """Facet markers for inlet/outlet/wall regions on (0, 1)^2.

    inlet = (0,1) x {1} (top), outlet = {1} x (0,1) (right),
    walls = {0} x (0,1) union (0,1) x {0} (left + bottom).
    """

    tol = 1.0e-10

    @staticmethod
    def inlet(x, on_boundary):
        return on_boundary and x[1] > 1.0 - _BoundaryMarkers.tol

    @staticmethod
    def outlet(x, on_boundary):
        return on_boundary and x[0] > 1.0 - _BoundaryMarkers.tol

    @staticmethod
    def walls(x, on_boundary):
        return on_boundary and (
            x[0] < _BoundaryMarkers.tol or x[1] < _BoundaryMarkers.tol
        )


def mark_boundaries(mesh):
    """Build a FacetFunction marking inlet=1, outlet=2, walls=3."""
    _require_fenics()

    boundary_markers = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)

    class _Inlet(df.SubDomain):
        def inside(self, x, on_boundary):
            return _BoundaryMarkers.inlet(x, on_boundary)

    class _Outlet(df.SubDomain):
        def inside(self, x, on_boundary):
            return _BoundaryMarkers.outlet(x, on_boundary)

    class _Walls(df.SubDomain):
        def inside(self, x, on_boundary):
            return _BoundaryMarkers.walls(x, on_boundary)

    _Inlet().mark(boundary_markers, INLET_MARKER)
    _Outlet().mark(boundary_markers, OUTLET_MARKER)
    _Walls().mark(boundary_markers, WALL_MARKER)

    return boundary_markers


def _make_viscosity_expression(
    coefficient_name: str,
    parameters: np.ndarray,
    base_value: float,
):
    """Build a FEniCS UserExpression for the random viscosity a(z, x).

    Reuses the exact tested NumPy coefficient functions pointwise (same
    approach as ``pde/diffusion/fenics_mixed_solver.py``), so this can
    never drift from the NumPy formulas used elsewhere in the repo.
    """
    _require_fenics()

    parameters = np.asarray(parameters, dtype=float)
    name = coefficient_name.lower()

    if name == "affine":

        class _AffineExpression(df.UserExpression):
            def eval(self, value, point) -> None:
                z1 = np.array([point[0]])
                value[0] = float(
                    affine_coefficient(z1=z1, parameters=parameters, base_value=base_value)[0]
                )

            def value_shape(self):
                return ()

        return _AffineExpression(degree=5)

    if name == "log":

        class _LogExpression(df.UserExpression):
            def eval(self, value, point) -> None:
                z1 = np.array([point[0]])
                value[0] = float(
                    log_transformed_coefficient(z1=z1, parameters=parameters)[0]
                )

            def value_shape(self):
                return ()

        return _LogExpression(degree=5)

    raise ValueError(f"Unsupported coefficient: {coefficient_name}")


@dataclass(frozen=True)
class NsbMeshInfo:
    """Mesh/DOF statistics recorded for comparison against the paper."""

    resolution: int
    num_cells: int
    h_min: float
    h_max: float
    num_dofs_u: int
    num_dofs_sigma: int
    num_dofs_total: int


@dataclass(frozen=True)
class NsbSolveResult:
    """Result of one nonlinear NSB mixed FEM solve."""

    u_dofs: np.ndarray
    p_dofs: np.ndarray
    sigma_dofs: np.ndarray
    converged: bool
    newton_iterations: int
    mesh_info: NsbMeshInfo


def build_nsb_function_space(mesh):
    """Build the AFW-based (ell=0) mixed function space for NSB.

    Returns (function_space, boundary_markers). See module docstring for
    the element choice.
    """
    _require_fenics()

    cell = mesh.ufl_cell()

    sigma_element = df.VectorElement("BDM", cell, 1, dim=2)
    u_element = df.VectorElement("DG", cell, 0)
    gamma_element = df.FiniteElement("DG", cell, 0)
    t11_element = df.FiniteElement("DG", cell, 1)
    t12_element = df.FiniteElement("DG", cell, 1)
    t21_element = df.FiniteElement("DG", cell, 1)

    mixed_element = df.MixedElement(
        [sigma_element, u_element, gamma_element, t11_element, t12_element, t21_element]
    )
    function_space = df.FunctionSpace(mesh, mixed_element)
    boundary_markers = mark_boundaries(mesh)

    return function_space, boundary_markers


def describe_nsb_mesh(mesh, function_space, resolution: int) -> NsbMeshInfo:
    """Compute mesh/DOF statistics, reporting u and sigma DOF counts
    separately for comparison against the paper's 1464 (u) / 244 (p)."""
    _require_fenics()

    num_dofs_u = function_space.sub(1).dim()
    num_dofs_sigma = function_space.sub(0).dim()

    return NsbMeshInfo(
        resolution=resolution,
        num_cells=mesh.num_cells(),
        h_min=mesh.hmin(),
        h_max=mesh.hmax(),
        num_dofs_u=num_dofs_u,
        num_dofs_sigma=num_dofs_sigma,
        num_dofs_total=function_space.dim(),
    )


def solve_nsb_mixed_fenics(
    mesh,
    coefficient_name: str,
    parameters: np.ndarray,
    base_value: float = 2.62,
    resolution_for_metadata: int = 0,
    newton_max_iterations: int = 40,
    newton_relaxation: float = 1.0,
) -> NsbSolveResult:
    """Solve the nonlinear NSB mixed FEM problem (paper eq. B.14-B.15) once.

    Parameters
    ----------
    mesh:
        A dolfin mesh of ``(0, 1)^2`` (same fixed mesh reused across all
        m/trials within one case, per the paper's protocol).
    coefficient_name:
        ``"affine"`` or ``"log"`` random viscosity a(z, x).
    parameters:
        Parameter vector x of shape ``(d,)``.
    base_value:
        Affine coefficient base value (paper: 2.62). Ignored for ``"log"``.
    resolution_for_metadata:
        mshr/UnitSquareMesh resolution used to build ``mesh``, recorded for
        provenance only.
    newton_max_iterations, newton_relaxation:
        Passed to dolfin's Newton solver. Full-sweep generation (Phase 7)
        should catch non-convergence and retry with a smaller relaxation
        factor rather than assume every sample converges.

    Returns
    -------
    NsbSolveResult
        u/sigma DOF vectors, post-processed pressure DOFs, convergence
        info, and mesh/DOF statistics.
    """
    _require_fenics()

    function_space, boundary_markers = build_nsb_function_space(mesh)
    normal = df.FacetNormal(mesh)
    ds = df.Measure("ds", domain=mesh, subdomain_data=boundary_markers)

    a_coefficient = _make_viscosity_expression(
        coefficient_name=coefficient_name,
        parameters=parameters,
        base_value=base_value,
    )
    eta = eta_field_expression()
    forcing = forcing_vector()
    inlet_velocity = inlet_velocity_expression()

    w = df.Function(function_space)
    test = df.TestFunction(function_space)

    sigma, u, gamma_s, t11, t12, t21 = df.split(w)
    sigma_t, u_t, gamma_t, t11_t, t12_t, t21_t = df.split(test)

    # Every entry below is written as an expression that structurally
    # depends on a field component (e.g. "0.0 * gamma_s" rather than a bare
    # df.Constant(0.0)), even where the value is mathematically zero. This
    # keeps all as_matrix() entries at consistent UFL "argument rank" after
    # automatic differentiation (df.derivative below); mixing literal
    # Constants with field-dependent entries in the same matrix otherwise
    # trips FFC/uflacs's argument-factorization pass with "Expecting equal
    # argument rank terms among summands".
    gamma_mat = df.as_matrix([[0.0 * gamma_s, gamma_s], [-gamma_s, 0.0 * gamma_s]])
    gamma_test_mat = df.as_matrix(
        [[0.0 * gamma_t, gamma_t], [-gamma_t, 0.0 * gamma_t]]
    )
    t_mat = df.as_matrix([[t11, t12], [t21, -t11]])
    t_test_mat = df.as_matrix([[t11_t, t12_t], [t21_t, -t11_t]])

    # Eq (I) [test s = t_test_mat]: strain-rate/pseudostress constitutive relation.
    eq1 = (
        LAMBDA * a_coefficient * df.inner(t_mat, t_test_mat)
        - df.inner(sigma, t_test_mat)
        - df.inner(df.outer(u, u), t_test_mat)
    ) * df.dx

    # Eq (II) [test tau = sigma_t]: kinematic relation + Dirichlet (natural)
    # boundary term on the inlet + Nitsche penalty on the outlet.
    eq2 = (
        df.inner(t_mat, sigma_t)
        + df.inner(gamma_mat, sigma_t)
        + df.dot(u, df.div(sigma_t))
    ) * df.dx
    eq2 += (
        KAPPA
        * df.dot(df.dot(sigma + df.outer(u, u), normal), df.dot(sigma_t, normal))
        * ds(OUTLET_MARKER)
    )
    eq2 -= df.dot(inlet_velocity, df.dot(sigma_t, normal)) * ds(INLET_MARKER)

    # Eq (III) [test (v, delta) = (u_t, gamma_test_mat)]: momentum balance.
    eq3 = (
        df.inner(gamma_test_mat, sigma)
        + df.dot(u_t, df.div(sigma))
        - eta * df.dot(u, u_t)
        + df.dot(forcing, u_t)
    ) * df.dx

    residual = eq1 + eq2 + eq3
    jacobian = df.derivative(residual, w)

    problem = df.NonlinearVariationalProblem(residual, w, bcs=[], J=jacobian)
    solver = df.NonlinearVariationalSolver(problem)
    solver.parameters["newton_solver"]["maximum_iterations"] = newton_max_iterations
    solver.parameters["newton_solver"]["relaxation_parameter"] = newton_relaxation
    solver.parameters["newton_solver"]["linear_solver"] = "mumps" if _mumps_available() else "lu"

    converged = True
    newton_iterations = 0
    try:
        newton_iterations, converged = solver.solve()
    except RuntimeError:
        converged = False

    # Pressure post-processing uses the UFL-level split (sigma, u), not a
    # deepcopy()'d split -- df.project() on a deepcopy'd BDM sub-Function
    # re-embedded in a brand-new form trips a pullback-shape assertion in
    # this FEniCS 2019.1.0 build (BDM's contravariant Piola pullback isn't
    # correctly recovered from a deepcopy'd mixed-space sub-Function). The
    # UFL split retains the original form's pullback context and works.
    pressure_space = df.FunctionSpace(mesh, "DG", 1)
    pressure_expr = -0.5 * df.tr(sigma + df.outer(u, u))
    pressure_h = df.project(pressure_expr, pressure_space)

    # sigma (hence p = -1/2 tr(sigma + u tensor u)) is only determined up to
    # an additive c*I (div(c*I)=0, and c*I doesn't affect the trace-free/
    # skew-symmetry equations either -- see reference paper eq. 3.9-3.11).
    # The paper pins c via a Lagrange multiplier enforcing the H_0(div_4/3)
    # zero-trace-integral subspace during the solve; this solver instead
    # uses the full (unconstrained) H(div_4/3) space and enforces the
    # paper's zero-mean-pressure condition (integral_Omega p = 0, eq. B.14)
    # as a post-hoc constant shift, which is exact and does not change u,
    # sigma's divergence, or any other physically meaningful quantity.
    domain_area = df.assemble(df.Constant(1.0) * df.dx(mesh))
    pressure_mean = df.assemble(pressure_h * df.dx) / domain_area
    pressure_h.vector()[:] = pressure_h.vector()[:] - pressure_mean

    sigma_h, u_h, gamma_h, t11_h, t12_h, t21_h = w.split(deepcopy=True)

    mesh_info = describe_nsb_mesh(mesh, function_space, resolution=resolution_for_metadata)

    return NsbSolveResult(
        u_dofs=np.asarray(u_h.vector().get_local(), dtype=np.float64),
        p_dofs=np.asarray(pressure_h.vector().get_local(), dtype=np.float64),
        sigma_dofs=np.asarray(sigma_h.vector().get_local(), dtype=np.float64),
        converged=bool(converged),
        newton_iterations=int(newton_iterations),
        mesh_info=mesh_info,
    )


def _mumps_available() -> bool:
    _require_fenics()
    try:
        return "mumps" in [name for name, _ in df.linear_solver_methods()]
    except Exception:  # pragma: no cover - defensive, falls back to lu
        return False
