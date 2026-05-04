"""FEniCS solver for the paper-aligned diffusion benchmark.

This module solves

    -div(a(z, x) grad u(z, x)) = f

on the unit square with Dirichlet boundary conditions:
- u = 0.5 on the bottom boundary,
- u = 0 elsewhere.

The implementation uses legacy FEniCS/DOLFIN. It expects the package
``dolfin`` to be installed, for example through a FEniCS 2019.1.0 environment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


try:
    import dolfin as df
except ImportError as error:  # pragma: no cover
    df = None
    _FENICS_IMPORT_ERROR = error
else:
    _FENICS_IMPORT_ERROR = None


@dataclass(frozen=True)
class FenicsDiffusionConfig:
    """Configuration for the FEniCS diffusion solve."""

    mesh_resolution: int = 32
    forcing: float = 10.0
    bottom_value: float = 0.5
    coefficient_name: str = "affine"
    affine_base_value: float = 2.62
    log_base_shift: float = 1.0
    dtype: str = "float32"


class AffineCoefficientExpression:
    """Paper affine coefficient expression.

    a_1(z, x) = 2.62 + sum_j x_j sin(pi z_1 j) / j^(3/2)
    """

    def __init__(
        self,
        parameters: np.ndarray,
        base_value: float = 2.62,
    ) -> None:
        """Create affine coefficient expression."""
        _require_fenics()
        self.parameters = np.asarray(parameters, dtype=float)
        self.base_value = float(base_value)

    def to_fenics_expression(self):
        """Return a FEniCS user expression."""
        parameters = self.parameters
        base_value = self.base_value

        class _Expression(df.UserExpression):
            def eval(self, value, point) -> None:
                coefficient = base_value
                z1 = point[0]

                for index, parameter in enumerate(parameters, start=1):
                    coefficient += (
                        parameter
                        * np.sin(np.pi * z1 * index)
                        / (index**1.5)
                    )

                value[0] = coefficient

            def value_shape(self):
                return ()

        return _Expression(degree=5)


class LogCoefficientExpression:
    """Paper-inspired log coefficient expression.

    This is still a practical approximation of the paper's a_{2,d}; the exact
    expansion from the paper can be inserted here later.
    """

    def __init__(
        self,
        parameters: np.ndarray,
        base_shift: float = 1.0,
    ) -> None:
        """Create log coefficient expression."""
        _require_fenics()
        self.parameters = np.asarray(parameters, dtype=float)
        self.base_shift = float(base_shift)

    def to_fenics_expression(self):
        """Return a FEniCS user expression."""
        parameters = self.parameters
        base_shift = self.base_shift

        class _Expression(df.UserExpression):
            def eval(self, value, point) -> None:
                log_coefficient = base_shift
                z1 = point[0]

                for index, parameter in enumerate(parameters, start=1):
                    log_coefficient += (
                        parameter
                        * np.sin(np.pi * z1 * index)
                        / (index**1.5)
                    )

                value[0] = np.exp(log_coefficient)

            def value_shape(self):
                return ()

        return _Expression(degree=5)


def solve_diffusion_fenics(
    parameters: np.ndarray,
    config: FenicsDiffusionConfig | None = None,
) -> np.ndarray:
    """Solve one FEniCS diffusion problem.

    Parameters
    ----------
    parameters:
        Parameter vector with shape ``(d,)``.
    config:
        FEniCS diffusion solver configuration.

    Returns
    -------
    np.ndarray
        Solution degrees of freedom as a one-dimensional array.
    """
    _require_fenics()

    if config is None:
        config = FenicsDiffusionConfig()

    mesh = df.UnitSquareMesh(config.mesh_resolution, config.mesh_resolution)
    function_space = df.FunctionSpace(mesh, "CG", 1)

    trial = df.TrialFunction(function_space)
    test = df.TestFunction(function_space)

    coefficient = _build_coefficient_expression(
        parameters=parameters,
        config=config,
    )

    forcing = df.Constant(config.forcing)

    bilinear_form = coefficient * df.dot(df.grad(trial), df.grad(test)) * df.dx
    linear_form = forcing * test * df.dx

    boundary_condition = df.DirichletBC(
        function_space,
        _BoundaryValue(bottom_value=config.bottom_value),
        "on_boundary",
    )

    solution = df.Function(function_space)
    df.solve(
        bilinear_form == linear_form,
        solution,
        boundary_condition,
        solver_parameters={
            "linear_solver": "lu",
        },
    )

    array = solution.vector().get_local()

    if config.dtype == "float32":
        return np.asarray(array, dtype=np.float32)

    if config.dtype == "float64":
        return np.asarray(array, dtype=np.float64)

    raise ValueError(f"Unsupported dtype: {config.dtype}")


def _build_coefficient_expression(
    parameters: np.ndarray,
    config: FenicsDiffusionConfig,
):
    """Build configured FEniCS coefficient expression."""
    coefficient_name = config.coefficient_name.lower()

    if coefficient_name == "affine":
        return AffineCoefficientExpression(
            parameters=parameters,
            base_value=config.affine_base_value,
        ).to_fenics_expression()

    if coefficient_name == "log":
        return LogCoefficientExpression(
            parameters=parameters,
            base_shift=config.log_base_shift,
        ).to_fenics_expression()

    raise ValueError(f"Unsupported coefficient: {config.coefficient_name}")


class _BoundaryValue:
    """Dirichlet boundary value for the diffusion benchmark."""

    def __init__(self, bottom_value: float) -> None:
        """Create boundary value expression."""
        _require_fenics()
        self.bottom_value = float(bottom_value)

    def __call__(self, point, on_boundary) -> float:
        """Evaluate boundary value."""
        if not on_boundary:
            return 0.0

        if abs(point[1]) < 1.0e-14:
            return self.bottom_value

        return 0.0


def _require_fenics() -> None:
    """Raise a clear error if FEniCS is unavailable."""
    if df is None:
        raise ImportError(
            "FEniCS/DOLFIN is not installed in this environment. "
            "Install legacy FEniCS, then retry. Original import error: "
            f"{_FENICS_IMPORT_ERROR}"
        )