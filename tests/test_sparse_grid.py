"""Tests for Clenshaw-Curtis sparse-grid quadrature (paper Appendix A.2(vi)).

Requires Tasmanian, which only runs inside the WSL 'sparsegrid' conda
environment (see scripts/wsl/setup_sparsegrid_env.sh) -- skipped
automatically wherever Tasmanian is not importable.
"""

from __future__ import annotations

import numpy as np
import pytest

Tasmanian = pytest.importorskip("Tasmanian")

from ol_reproduction.data.sparse_grid import (  # noqa: E402
    DEFAULT_LEVEL_BY_DIMENSION,
    build_clenshaw_curtis_quadrature,
    validate_quadrature,
)


@pytest.mark.parametrize("dimension", [2, 4])
def test_weights_sum_to_domain_volume(dimension: int) -> None:
    """Weights integrate the constant function 1 over [-1,1]^d exactly."""
    quadrature = build_clenshaw_curtis_quadrature(dimension=dimension, level=3)

    assert quadrature.weights.sum() == pytest.approx(2.0**dimension, rel=1e-10)


@pytest.mark.parametrize("dimension", [2, 4])
def test_quadrature_is_exact_for_low_degree_polynomial(dimension: int) -> None:
    """Clenshaw-Curtis at level>=2 should exactly integrate x_1^2.

    int_{[-1,1]^d} x_1^2 dx = (2/3) * 2^(d-1).
    """
    quadrature = build_clenshaw_curtis_quadrature(dimension=dimension, level=4)

    values = quadrature.points[:, 0] ** 2
    estimate = np.sum(quadrature.weights * values)

    expected = (2.0 / 3.0) * 2.0 ** (dimension - 1)
    assert estimate == pytest.approx(expected, rel=1e-8)


def test_points_stay_within_domain() -> None:
    quadrature = build_clenshaw_curtis_quadrature(dimension=4, level=3)

    assert np.all(quadrature.points >= -1.0 - 1e-9)
    assert np.all(quadrature.points <= 1.0 + 1e-9)


def test_default_levels_produce_valid_quadrature() -> None:
    """The levels actually used by the data-generation pipeline (Phase 6/7)
    must pass validate_quadrature -- this is the real configuration, not
    just an arbitrary test level."""
    for dimension, level in DEFAULT_LEVEL_BY_DIMENSION.items():
        quadrature = build_clenshaw_curtis_quadrature(dimension=dimension, level=level)
        validate_quadrature(quadrature)  # raises on failure


def test_validate_quadrature_rejects_bad_weight_sum() -> None:
    from ol_reproduction.data.sparse_grid import SparseGridQuadrature

    bad = SparseGridQuadrature(
        dimension=2,
        level=1,
        points=np.zeros((3, 2)),
        weights=np.array([1.0, 1.0, 1.0]),  # sums to 3, not 4 = 2^2
    )

    with pytest.raises(ValueError):
        validate_quadrature(bad)
