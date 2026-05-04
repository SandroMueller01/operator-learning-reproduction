"""FEniCS solver scaffold for the Navier--Stokes--Brinkman benchmark.

The paper uses a nonlinear mixed finite element formulation for the stationary
Navier--Stokes--Brinkman system. Implementing it exactly requires the full weak
form, nonlinear solve, pressure post-processing, and boundary decomposition.

This module intentionally raises ``NotImplementedError`` until the exact mixed
FEM solver is implemented. It exists so the repository has the correct
extension point for exact paper reproduction.
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
class FenicsNsbConfig:
    """Configuration for the paper NSB FEniCS benchmark."""

    mesh_resolution: int = 32
    lambda_value: float = 0.1
    pressure_stabilization: float = 1.0e-8
    coefficient_name: str = "affine"
    dtype: str = "float32"


def solve_nsb_fenics(
    parameters: np.ndarray,
    config: FenicsNsbConfig | None = None,
) -> dict[str, np.ndarray]:
    """Solve one paper-style NSB problem with FEniCS.

    Parameters
    ----------
    parameters:
        Parameter vector with shape ``(d,)``.
    config:
        Solver configuration.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary containing ``y_u`` and ``y_p``.

    Raises
    ------
    NotImplementedError
        Always raised until the exact mixed nonlinear solver is implemented.
    """
    _require_fenics()

    if config is None:
        config = FenicsNsbConfig()

    raise NotImplementedError(
        "Exact FEniCS Navier--Stokes--Brinkman solver is not implemented yet. "
        "Required paper components: nonlinear convective term, symmetric "
        "gradient, inlet profile, wall no-slip condition, outlet zero normal "
        "Cauchy stress, zero-mean pressure, and pressure post-processing."
    )


def eta_field_expression():
    """Return eta(z) = 10 + z1^2 + z2^2 as a FEniCS expression."""
    _require_fenics()
    return df.Expression(
        "10.0 + x[0]*x[0] + x[1]*x[1]",
        degree=2,
    )


def forcing_vector():
    """Return paper forcing f = (0, -1)."""
    _require_fenics()
    return df.Constant((0.0, -1.0))


def inlet_velocity_expression():
    """Return paper inlet velocity expression."""
    _require_fenics()
    return df.Expression(
        (
            "pow(0.0625, -1.0) * ((x[1] - 0.5) * (1.0 - x[1]))",
            "0.0",
        ),
        degree=2,
    )


def _require_fenics() -> None:
    """Raise a clear error if FEniCS is unavailable."""
    if df is None:
        raise ImportError(
            "FEniCS/DOLFIN is not installed in this environment. "
            "Install legacy FEniCS, then retry. Original import error: "
            f"{_FENICS_IMPORT_ERROR}"
        )