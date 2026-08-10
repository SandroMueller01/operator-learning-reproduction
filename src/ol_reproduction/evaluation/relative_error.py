"""Relative error metrics."""

from __future__ import annotations

import numpy as np


def relative_l2_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray | None = None,
    eps: float = 1.0e-12,
) -> float:
    """Compute relative L2 error over a dataset.

    Parameters
    ----------
    y_true:
        Reference outputs with shape ``(num_samples, output_dim)``.
    y_pred:
        Predicted outputs with shape ``(num_samples, output_dim)``.
    weights:
        Optional sample weights with shape ``(num_samples,)``.
    eps:
        Small number used to avoid division by zero.

    Returns
    -------
    float
        Relative L2 error.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape, "
            f"got {y_true.shape} and {y_pred.shape}."
        )

    difference = y_true - y_pred

    if weights is None:
        numerator = np.sum(difference**2)
        denominator = np.sum(y_true**2)
    else:
        weights = np.asarray(weights)

        if weights.ndim != 1:
            raise ValueError("weights must be one-dimensional.")

        if weights.shape[0] != y_true.shape[0]:
            raise ValueError(
                "weights must have length equal to number of samples."
            )

        reshaped_weights = weights.reshape(-1, 1)
        numerator = np.sum(reshaped_weights * difference**2)
        denominator = np.sum(reshaped_weights * y_true**2)

    return float(np.sqrt(numerator / (denominator + eps)))


def relative_l2_error_mass_weighted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mass_matrix,
    parametric_weights: np.ndarray | None = None,
    eps: float = 1.0e-12,
) -> float:
    """Relative Bochner-norm error using a FEM mass-matrix Y-norm.

    This is the paper's actual test-error formula (Appendix A.2(vi)):

        e_F^test = sqrt(sum_i w_i ||F(X_i) - Fhat(X_i)||^2_Y)
                   / sqrt(sum_i w_i ||F(X_i)||^2_Y)

    where ``||v||^2_Y = v^T M v`` for the FEM mass matrix M (see
    ``ol_reproduction.pde.mass_matrix``), not a naive sum-of-squares over
    raw DOF values (that's what ``relative_l2_error`` computes instead --
    kept separate rather than changed in place, since it's used elsewhere
    for plain unweighted comparisons).

    Parameters
    ----------
    y_true, y_pred:
        Arrays of shape ``(num_samples, output_dim)``, ``output_dim``
        matching ``mass_matrix``'s dimension.
    mass_matrix:
        A ``scipy.sparse`` matrix of shape ``(output_dim, output_dim)``,
        e.g. loaded via ``ol_reproduction.pde.mass_matrix.load_mass_matrix_npz``.
    parametric_weights:
        Optional sparse-grid quadrature weights ``w_i``, shape
        ``(num_samples,)``. If omitted, every sample is weighted equally
        (``w_i = 1``), which is *not* the paper's protocol unless the
        weights are folded in externally -- pass them explicitly for a
        faithful sparse-grid test error.
    eps:
        Small number used to avoid division by zero.

    Returns
    -------
    float
        Mass-matrix-weighted relative Bochner-norm error.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape, "
            f"got {y_true.shape} and {y_pred.shape}."
        )

    if y_true.shape[1] != mass_matrix.shape[0]:
        raise ValueError(
            "mass_matrix dimension does not match output_dim: "
            f"got mass_matrix shape {mass_matrix.shape}, "
            f"output_dim {y_true.shape[1]}."
        )

    if parametric_weights is None:
        parametric_weights = np.ones(y_true.shape[0], dtype=np.float64)
    else:
        parametric_weights = np.asarray(parametric_weights)

        if parametric_weights.ndim != 1:
            raise ValueError("parametric_weights must be one-dimensional.")

        if parametric_weights.shape[0] != y_true.shape[0]:
            raise ValueError(
                "parametric_weights must have length equal to number of samples."
            )

    difference = y_true - y_pred

    # Per-sample quadratic form v^T M v for all samples at once: M @ V.T
    # gives a dense (output_dim, num_samples) array (mass_matrix @ a dense
    # array is dense even when mass_matrix is sparse); summing
    # elementwise-multiplied V.T over the output_dim axis gives the
    # diagonal of V @ M @ V.T, i.e. the mass-weighted squared Y-norm of
    # every row of V, without ever forming the full
    # (num_samples, num_samples) matrix V @ M @ V.T. np.asarray(...) guards
    # against scipy returning np.matrix (whose "*" is matrix, not
    # elementwise, multiplication) for older scipy versions.
    def _quad_form_per_sample(values: np.ndarray) -> np.ndarray:
        mass_times_values = np.asarray(mass_matrix @ values.T)
        return np.sum(np.asarray(values.T) * mass_times_values, axis=0)

    diff_quad_form = _quad_form_per_sample(difference)
    true_quad_form = _quad_form_per_sample(y_true)

    numerator = float(np.sum(parametric_weights * diff_quad_form))
    denominator = float(np.sum(parametric_weights * true_quad_form))

    return float(np.sqrt(numerator / (denominator + eps)))