"""FEniCS solver scaffold for the Boussinesq benchmark.

The paper uses a 3D coupled nonlinear mixed finite element formulation for
stationary Boussinesq flow. Implementing it exactly requires a full 3D mixed FEM
solver with velocity, pressure and temperature spaces.

This module intentionally raises ``NotImplementedError`` until the exact solver
is implemented. It provides the correct extension point and paper parameter
helpers.
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
class FenicsBoussinesqConfig:
    """Configuration for the paper Boussinesq FEniCS benchmark."""

    mesh_resolution: int = 8
    coefficient_name: str = "affine"
    dtype: str = "float32"


def solve_boussinesq_fenics(
    parameters: np.ndarray,
    config: FenicsBoussinesqConfig | None = None,
) -> dict[str, np.ndarray]:
    """Solve one paper-style Boussinesq problem with FEniCS.

    Parameters
    ----------
    parameters:
        Parameter vector with shape ``(d,)``.
    config:
        Solver configuration.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary containing ``y_u``, ``y_phi`` and ``y_p``.

    Raises
    ------
    NotImplementedError
        Always raised until the exact 3D mixed nonlinear solver is implemented.
    """
    _require_fenics()

    if config is None:
        config = FenicsBoussinesqConfig()

    raise NotImplementedError(
        "Exact FEniCS Boussinesq solver is not implemented yet. Required "
        "paper components: 3D unit-cube mesh, velocity-pressure-temperature "
        "mixed spaces, gravity coupling, temperature-dependent viscosity "
        "varpi(phi)=0.1+exp(-phi), parametric thermal conductivity tensor, "
        "and coupled nonlinear solve."
    )


def gravity_vector():
    """Return paper gravity vector g = (0, 0, -1)."""
    _require_fenics()
    return df.Constant((0.0, 0.0, -1.0))


def viscosity_temperature_law(phi):
    """Return varpi(phi) = 0.1 + exp(-phi)."""
    _require_fenics()
    return 0.1 + df.exp(-phi)


def bottom_temperature_expression():
    """Return paper bottom temperature boundary expression."""
    _require_fenics()
    return df.Expression(
        "exp(4.0 * (-(x[0] - 0.5)*(x[0] - 0.5) "
        "- (x[1] - 0.5)*(x[1] - 0.5)))",
        degree=4,
    )


def bottom_velocity_expression():
    """Return paper bottom velocity boundary value u_D = (1, 1, 0)."""
    _require_fenics()
    return df.Constant((1.0, 1.0, 0.0))


def _require_fenics() -> None:
    """Raise a clear error if FEniCS is unavailable."""
    if df is None:
        raise ImportError(
            "FEniCS/DOLFIN is not installed in this environment. "
            "Install legacy FEniCS, then retry. Original import error: "
            f"{_FENICS_IMPORT_ERROR}"
        )