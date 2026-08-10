"""Tests for FEM mass-matrix assembly (Y-norm weighting, Phase 5).

Requires legacy FEniCS (dolfin); skipped automatically wherever dolfin is
not importable.
"""

from __future__ import annotations

import numpy as np
import pytest

df = pytest.importorskip("dolfin")

from ol_reproduction.pde.mass_matrix import (  # noqa: E402
    assemble_mass_matrix_csr,
    load_mass_matrix_npz,
    save_mass_matrix_npz,
)


def test_mass_matrix_on_toy_two_triangle_mesh_matches_hand_computation(tmp_path) -> None:
    """A DG0 mass matrix on a mesh is exactly diagonal with entries equal
    to each cell's area (since DG0 basis functions have disjoint support).
    UnitSquareMesh(1, 1) has 2 triangles, each of area 0.5.
    """
    mesh = df.UnitSquareMesh(1, 1)
    assert mesh.num_cells() == 2

    function_space = df.FunctionSpace(mesh, "DG", 0)
    mass_matrix = assemble_mass_matrix_csr(function_space)

    dense = mass_matrix.toarray()
    assert dense.shape == (2, 2)

    # Off-diagonal entries must be zero (DG0 basis functions never overlap).
    np.testing.assert_allclose(dense[0, 1], 0.0, atol=1e-12)
    np.testing.assert_allclose(dense[1, 0], 0.0, atol=1e-12)

    # Diagonal entries equal each triangle's area (0.5 each, summing to 1
    # the total area of the unit square).
    np.testing.assert_allclose(np.diag(dense), [0.5, 0.5], rtol=1e-10)


def test_mass_matrix_quadratic_form_matches_l2_norm_of_constant_field() -> None:
    """For a DG0 field equal to the constant 1 everywhere, v^T M v should
    equal the total mesh area (= int_Omega 1^2 dx)."""
    mesh = df.UnitSquareMesh(4, 4)
    function_space = df.FunctionSpace(mesh, "DG", 0)
    mass_matrix = assemble_mass_matrix_csr(function_space)

    ones = np.ones(function_space.dim())
    quad_form = ones @ (mass_matrix @ ones)

    assert quad_form == pytest.approx(1.0, rel=1e-10)  # unit square area


def test_save_and_load_mass_matrix_roundtrip(tmp_path) -> None:
    mesh = df.UnitSquareMesh(3, 3)
    function_space = df.FunctionSpace(mesh, "DG", 0)
    mass_matrix = assemble_mass_matrix_csr(function_space)

    path = tmp_path / "mass_matrix.npz"
    save_mass_matrix_npz(path, mass_matrix)
    loaded = load_mass_matrix_npz(path)

    np.testing.assert_allclose(loaded.toarray(), mass_matrix.toarray(), rtol=1e-10)
