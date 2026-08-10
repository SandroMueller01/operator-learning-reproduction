"""Phase 5: check sparse-grid point counts across levels for d=4 and d=8."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ol_reproduction.data.sparse_grid import build_clenshaw_curtis_quadrature

for d in (4, 8):
    print(f"--- d={d} ---")
    for level in range(1, 7):
        try:
            q = build_clenshaw_curtis_quadrature(dimension=d, level=level)
        except Exception as exc:  # noqa: BLE001
            print(f"level={level}: FAILED ({exc})")
            continue
        print(f"level={level}: m_test={q.points.shape[0]}, weight_sum={q.weights.sum():.6f}")
