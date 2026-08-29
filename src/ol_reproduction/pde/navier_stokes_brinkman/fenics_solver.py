"""Mixed FEM nonlinear solver for the Navier-Stokes-Brinkman equations
(paper eq. B.14-B.15).

Implements the non-augmented Banach-spaces mixed formulation of Gatica,
Nunez & Ruiz-Baier (2022, J. Numer. Math. 31(4):343-373) -- the paper's own
cited source for eq. (B.15) -- using AFW (Arnold-Falk-Winther) finite
elements, plus the arXiv paper's Nitsche-method addition (eq. B.15
discussion) for the outlet's zero-Cauchy-stress condition, which the base
reference (all-Dirichlet) does not need.

Unknowns: pseudostress sigma in H(div_4/3), velocity u in L^4, strain rate
t in L^2_tr (trace-free), vorticity gamma in L^2_skew. Pressure is
recovered afterwards via p = -1/2 tr(sigma + u tensor u).

The mesh and formulation now come directly from the paper authors' own
released code (``PDE_DATA/CODE_NSB``, obtained from the supervisor after an
earlier calibrated-approximation attempt turned out not to match the
authors' own script in several ways -- see
``practical_work_report/91-appendix.tex`` for the full history):

* Mesh: the authors' exact ``poisson.xml`` (same file as the diffusion
  experiment, loaded via ``load_original_mesh``), not an independently
  built mesh.
* AFW elements, ``FE_degree=1`` (``deg`` below), per tensor row:
      sigma_h in P_{deg+1}(Omega) ^ H(div;Omega)  (BDM2, one copy per row)
      u_h     in P_deg(Omega)                     (DG1 vector)
      gamma_h in L^2_skew(Omega) ^ P_deg(Omega)    (DG1)
      t_h     in P_{deg+1}(Omega) ^ L^2_tr(Omega)  (DG2, 3 components)
  (previously BDM1/DG0/DG0/DG1 -- one degree too low on every field).
* Inlet velocity: the constant ``(0.1, 0.0)`` (``uinlet`` in the authors'
  code), not the position-dependent parabolic profile previously
  reconstructed from a paper formula that turned out to be ambiguous.
* Pressure: projected into **DG0** (``Ph`` in the authors' code, matching
  the paper's documented 244 pressure DOFs on this mesh -- one DG0 dof per
  cell), not DG1, and with **no** post-hoc mean-shift: the authors never
  renormalize ``p`` to zero mean after projecting.
* Inlet boundary term sign: the authors' residual is
  ``AA - FF + nitsche_term`` with
  ``FF = dot(tau*n, uinlet) * u_D * ds - dot(f, v) * dx`` and the inlet
  indicator ``u_D = -1`` on the inlet edge, which combines to a **plus**
  sign on the inlet term in the final residual
  (``-(-1) = +1``). This module previously had that boundary term
  subtracted, a genuine sign bug independent of the element/mesh/inlet
  corrections above.

In 2D, a skew-symmetric tensor has exactly one independent scalar (the
off-diagonal entry), and a trace-free tensor has exactly three
independent scalars -- both are represented here via plain scalar DG
FunctionSpaces combined with ``as_matrix`` in the UFL forms, rather than
custom constrained finite elements, which is an exact (not approximate)
representation of L^2_skew(Omega) and L^2_tr(Omega) in two dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ol_reproduction.coefficients.affine import affine_coefficient
from ol_reproduction.coefficients.log_transformed import (
    log_transformed_coefficient,
)
from ol_reproduction.pde.diffusion.fenics_mixed_solver import load_original_mesh

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
    """Return the paper authors' inlet velocity, a constant ``(0.1, 0.0)``
    (``uinlet`` in ``PDE_DATA/CODE_NSB/PDE_data_NSB.py``) -- not a function
    of position, and not the parabolic profile previously reconstructed
    from an ambiguous paper formula."""
    _require_fenics()
    return df.Expression(("0.1", "0.0"), degree=2)


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


def build_nsb_function_space(mesh, fe_degree: int = 1):
    """Build the AFW mixed function space for NSB.

    Field order matches the authors' own ``Hh = FunctionSpace(mesh,
    MixedElement([Hu,Ht,Hsig,Hsig,Hgam]))``: velocity, strain-rate (3
    independent components), pseudostress row 1, pseudostress row 2,
    vorticity. ``fe_degree`` matches their ``--FE_degree`` (default 1):
    sigma rows use ``BDM(fe_degree+1)``, u uses ``DG(fe_degree)``, gamma
    uses ``DG(fe_degree)``, t uses ``DG(fe_degree+1)`` (3 components).

    Returns (function_space, boundary_markers). See module docstring for
    the element choice.
    """
    _require_fenics()

    cell = mesh.ufl_cell()

    u_element = df.VectorElement("DG", cell, fe_degree)
    t11_element = df.FiniteElement("DG", cell, fe_degree + 1)
    t12_element = df.FiniteElement("DG", cell, fe_degree + 1)
    t21_element = df.FiniteElement("DG", cell, fe_degree + 1)
    sigma1_element = df.FiniteElement("BDM", cell, fe_degree + 1)
    sigma2_element = df.FiniteElement("BDM", cell, fe_degree + 1)
    gamma_element = df.FiniteElement("DG", cell, fe_degree)

    mixed_element = df.MixedElement(
        [
            u_element,
            t11_element,
            t12_element,
            t21_element,
            sigma1_element,
            sigma2_element,
            gamma_element,
        ]
    )
    function_space = df.FunctionSpace(mesh, mixed_element)
    boundary_markers = mark_boundaries(mesh)

    return function_space, boundary_markers


def describe_nsb_mesh(mesh, function_space, resolution: int) -> NsbMeshInfo:
    """Compute mesh/DOF statistics, reporting u and sigma DOF counts
    separately for comparison against the paper's 1464 (u) / 244 (p)."""
    _require_fenics()

    num_dofs_u = function_space.sub(0).dim()
    num_dofs_sigma = function_space.sub(4).dim() + function_space.sub(5).dim()

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
    fe_degree: int = 1,
    newton_max_iterations: int = 40,
    newton_relaxation: float = 1.0,
) -> NsbSolveResult:
    """Solve the nonlinear NSB mixed FEM problem (paper eq. B.14-B.15) once.

    Residual assembled term-for-term from the authors'
    ``PDE_DATA/CODE_NSB/PDE_data_NSB.py::gen_dirichlet_data_NSB`` (variable
    names ``a, b1, b, b2, bbt, bb, cc`` there map onto ``eq1``/``eq2``/
    ``eq3`` below); see module docstring for the corrections this made
    versus the previous version (elements, inlet velocity, pressure space,
    inlet-term sign).

    Parameters
    ----------
    mesh:
        A dolfin mesh of ``(0, 1)^2`` (same fixed mesh reused across all
        m/trials within one case, per the paper's protocol) -- use
        ``load_original_mesh()`` for paper-scale work.
    coefficient_name:
        ``"affine"`` or ``"log"`` random viscosity a(z, x).
    parameters:
        Parameter vector x of shape ``(d,)``.
    base_value:
        Affine coefficient base value (paper: 2.62). Ignored for ``"log"``.
    resolution_for_metadata:
        mshr/UnitSquareMesh resolution used to build ``mesh``, recorded for
        provenance only.
    fe_degree:
        Matches the authors' ``--FE_degree`` (default 1). See
        ``build_nsb_function_space``.
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

    function_space, boundary_markers = build_nsb_function_space(mesh, fe_degree=fe_degree)
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

    u, t11, t12, t21, sigma1, sigma2, gamma_s = df.split(w)
    u_t, t11_t, t12_t, t21_t, sigma1_t, sigma2_t, gamma_t = df.split(test)

    # as_tensor((row0, row1)) stacks two vector-valued expressions as tensor
    # rows -- matches the authors' ``sigma = as_tensor((sig1,sig2))`` exactly
    # (as_matrix expects nested scalars, not vector-valued rows).
    sigma = df.as_tensor((sigma1, sigma2))
    sigma_t = df.as_tensor((sigma1_t, sigma2_t))

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
    eq2 += df.dot(inlet_velocity, df.dot(sigma_t, normal)) * ds(INLET_MARKER)

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
    # Space and formula match the authors' ``Ph = FunctionSpace(mesh, 'DG',
    # 0)`` / ``ph = project(-1/ndim*tr(sigmah+outer(uh,uh)), Ph)`` exactly
    # -- including no post-hoc mean-shift (the authors never renormalize p).
    pressure_space = df.FunctionSpace(mesh, "DG", 0)
    pressure_expr = -0.5 * df.tr(sigma + df.outer(u, u))
    pressure_h = df.project(pressure_expr, pressure_space)

    u_h, t11_h, t12_h, t21_h, sigma1_h, sigma2_h, gamma_h = w.split(deepcopy=True)
    sigma_dofs = np.concatenate(
        [
            np.asarray(sigma1_h.vector().get_local(), dtype=np.float64),
            np.asarray(sigma2_h.vector().get_local(), dtype=np.float64),
        ]
    )

    mesh_info = describe_nsb_mesh(mesh, function_space, resolution=resolution_for_metadata)

    return NsbSolveResult(
        u_dofs=np.asarray(u_h.vector().get_local(), dtype=np.float64),
        p_dofs=np.asarray(pressure_h.vector().get_local(), dtype=np.float64),
        sigma_dofs=sigma_dofs,
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
