"""FEM mass-matrix assembly for Bochner (Y-norm) error computation.

The paper's test error (Appendix A.2(vi)) is a relative L^2_mu(X;Y) Bochner
norm error, where ||.||_Y is a genuine function-space norm
(||v||^2_Y = int_Omega v^2, or int_Omega |v|^2 for vector fields) -- not a
naive sum-of-squares over raw FEM DOF values. For a FEM discretization,
the correct discrete Y-norm is the mass-matrix-weighted quadratic form
v^T M v, where M_ij = int_Omega phi_i phi_j (phi_i the FEM basis
functions).

This module assembles that mass matrix once per case (in FEniCS, WSL-side)
and exports it as a plain SciPy CSR sparse matrix, so the Windows-side
training/evaluation pipeline (ol_reproduction.evaluation.relative_error)
never needs FEniCS again -- only NumPy/SciPy.
"""

from __future__ import annotations

import numpy as np

try:
    import dolfin as df
except ImportError as error:  # pragma: no cover - exercised only without FEniCS
    df = None
    _FENICS_IMPORT_ERROR = error
else:
    _FENICS_IMPORT_ERROR = None


def _require_fenics() -> None:
    if df is None:
        raise ImportError(
            "dolfin (legacy FEniCS 2019.1.0) is required to assemble a FEM "
            "mass matrix. Run this module inside the WSL 'fenics2019' "
            f"conda environment. Original import error: {_FENICS_IMPORT_ERROR}"
        )


def assemble_mass_matrix_csr(function_space):
    """Assemble the FEM mass matrix for a scalar or vector FunctionSpace.

    Parameters
    ----------
    function_space:
        A dolfin FunctionSpace (scalar, e.g. DG0/DG1, or vector, e.g. a
        VectorElement space). Must not be a mixed/blended space directly
        (pass the relevant ``.sub(i).collapse()`` sub-space instead).

    Returns
    -------
    scipy.sparse.csr_matrix
        The mass matrix M with M[i, j] = int_Omega phi_i . phi_j, shape
        (dim, dim) where dim = function_space.dim().
    """
    _require_fenics()
    import scipy.sparse as sp

    trial = df.TrialFunction(function_space)
    test = df.TestFunction(function_space)

    if len(trial.ufl_shape) == 0:
        form = trial * test * df.dx
    else:
        form = df.inner(trial, test) * df.dx

    assembled = df.assemble(form)
    petsc_mat = df.as_backend_type(assembled).mat()
    indptr, indices, data = petsc_mat.getValuesCSR()

    dim = function_space.dim()
    return sp.csr_matrix((data, indices, indptr), shape=(dim, dim))


def save_mass_matrix_npz(path, mass_matrix) -> None:
    """Save a SciPy CSR mass matrix to an .npz sidecar file.

    Stored as plain arrays (data/indices/indptr/shape) so it can be loaded
    on the Windows side with only NumPy/SciPy, no FEniCS.
    """
    mass_matrix = mass_matrix.tocsr()
    np.savez_compressed(
        path,
        data=mass_matrix.data,
        indices=mass_matrix.indices,
        indptr=mass_matrix.indptr,
        shape=np.array(mass_matrix.shape, dtype=np.int64),
    )


def load_mass_matrix_npz(path):
    """Load a mass matrix saved by ``save_mass_matrix_npz``.

    Returns
    -------
    scipy.sparse.csr_matrix
    """
    import scipy.sparse as sp

    with np.load(path) as archive:
        return sp.csr_matrix(
            (archive["data"], archive["indices"], archive["indptr"]),
            shape=tuple(archive["shape"]),
        )
