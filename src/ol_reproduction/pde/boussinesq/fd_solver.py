"""Simplified finite-difference solver for a Boussinesq-type problem.

This module implements a practical, simplified Boussinesq-style data generator.
It is not an exact implementation of the mixed finite element Boussinesq solver
used in the target paper.

The simplified model is generated in two stages:

1. Solve a scalar heat equation for the temperature field ``phi``.
2. Use the mean temperature as a buoyancy force in a simplified Brinkman flow
   solve to obtain velocity ``u`` and pressure ``p``.

This gives a coupled parameter-to-solution map with outputs ``u``, ``phi`` and
``p`` that can be used for the reproduction pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ol_reproduction.pde.diffusion.fd_solver import (
    DiffusionBoundaryConditions,
    solve_diffusion_fd,
)
from ol_reproduction.pde.navier_stokes_brinkman.fd_solver import (
    NsbForcing,
    NsbSolverParameters,
    flatten_pressure,
    flatten_velocity,
    solve_nsb_fd,
)


@dataclass(frozen=True)
class BoussinesqSolverParameters:
    """Parameters for the simplified Boussinesq solver."""

    temperature_forcing: float = 1.0
    buoyancy_scale: float = 30.0
    brinkman_alpha: float = 1.0
    pressure_stabilization: float = 1.0e-6


@dataclass(frozen=True)
class TemperatureBoundaryConditions:
    """Dirichlet boundary values for the temperature equation."""

    bottom: float = 1.0
    top: float = 100.0
    left: float = 1.0
    right: float = 1.0e-6


def solve_boussinesq_fd(
    viscosity: np.ndarray,
    thermal_conductivity: np.ndarray,
    solver_parameters: BoussinesqSolverParameters | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve the simplified Boussinesq-type system.

    Parameters
    ----------
    viscosity:
        Positive viscosity field with shape ``(ny, nx)``.
    thermal_conductivity:
        Positive thermal conductivity field with shape ``(ny, nx)``.
    solver_parameters:
        Numerical parameters for the simplified coupling.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Velocity field with shape ``(ny, nx, 2)``, temperature field with shape
        ``(ny, nx)``, and pressure field with shape ``(ny, nx)``.
    """
    _validate_positive_field(viscosity, name="viscosity")
    _validate_positive_field(
        thermal_conductivity,
        name="thermal_conductivity",
    )

    if viscosity.shape != thermal_conductivity.shape:
        raise ValueError(
            "viscosity and thermal_conductivity must have the same shape, "
            f"got {viscosity.shape} and {thermal_conductivity.shape}."
        )

    if solver_parameters is None:
        solver_parameters = BoussinesqSolverParameters()

    temperature = _solve_temperature_equation(
        thermal_conductivity=thermal_conductivity,
        temperature_forcing=solver_parameters.temperature_forcing,
    )

    buoyancy_force = float(
        solver_parameters.buoyancy_scale * np.mean(temperature)
    )

    velocity, pressure = solve_nsb_fd(
        viscosity=viscosity,
        forcing=NsbForcing(
            force_x=0.0,
            force_y=buoyancy_force,
        ),
        solver_parameters=NsbSolverParameters(
            brinkman_alpha=solver_parameters.brinkman_alpha,
            pressure_stabilization=solver_parameters.pressure_stabilization,
        ),
    )

    return velocity, temperature, pressure


def flatten_temperature(temperature: np.ndarray) -> np.ndarray:
    """Flatten temperature field.

    Parameters
    ----------
    temperature:
        Temperature array with shape ``(ny, nx)``.

    Returns
    -------
    np.ndarray
        Flattened temperature vector.
    """
    if temperature.ndim != 2:
        raise ValueError(
            "temperature must have shape (ny, nx), "
            f"got {temperature.shape}."
        )

    return np.asarray(temperature, dtype=np.float32).reshape(-1)


def flatten_boussinesq_velocity(velocity: np.ndarray) -> np.ndarray:
    """Flatten Boussinesq velocity field."""
    return flatten_velocity(velocity)


def flatten_boussinesq_pressure(pressure: np.ndarray) -> np.ndarray:
    """Flatten Boussinesq pressure field."""
    return flatten_pressure(pressure)


def _solve_temperature_equation(
    thermal_conductivity: np.ndarray,
    temperature_forcing: float,
) -> np.ndarray:
    """Solve the scalar temperature equation.

    Parameters
    ----------
    thermal_conductivity:
        Positive thermal conductivity field.
    temperature_forcing:
        Constant heat forcing.

    Returns
    -------
    np.ndarray
        Temperature field.
    """
    boundary_conditions = DiffusionBoundaryConditions(
        bottom=TemperatureBoundaryConditions().bottom,
        top=TemperatureBoundaryConditions().top,
        left=TemperatureBoundaryConditions().left,
        right=TemperatureBoundaryConditions().right,
    )

    return solve_diffusion_fd(
        coefficient=thermal_conductivity,
        forcing=temperature_forcing,
        boundary_conditions=boundary_conditions,
    )


def _validate_positive_field(field: np.ndarray, name: str) -> None:
    """Validate a positive two-dimensional field.

    Parameters
    ----------
    field:
        Field to validate.
    name:
        Field name used in error messages.
    """
    if field.ndim != 2:
        raise ValueError(
            f"{name} must have shape (ny, nx), got {field.shape}."
        )

    if field.shape[0] < 4 or field.shape[1] < 4:
        raise ValueError(f"{name} grid must be at least 4 by 4.")

    if np.any(field <= 0.0):
        raise ValueError(f"{name} must be strictly positive.")