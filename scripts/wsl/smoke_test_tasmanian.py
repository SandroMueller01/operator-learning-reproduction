"""Phase 0 smoke test: confirm Tasmanian can build a Clenshaw-Curtis sparse grid."""

import Tasmanian

grid = Tasmanian.makeGlobalGrid(
    2, 0, 2, "level", "clenshaw-curtis"
)
points = grid.getPoints()
weights = grid.getQuadratureWeights()

print("TASMANIAN_OK, num_points=", points.shape[0], "weight_sum=", weights.sum())
