"""Offline checks for the Safe Passage costmap snapshot helpers."""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
from frames.utils import safe_passage_geometry as geo  # noqa: E402


def test_nearest_other():
    np.testing.assert_allclose(geo.nearest_other([[0, 0], [3, 0], [3, 4]]), [3, 3, 4])
    assert geo.nearest_other([[1, 1]])[0] == math.inf


def test_cost_is_zero_for_unknown_and_outside():
    data = np.array([[0, 50], [-1, 100]], dtype=np.int16)
    grid = geo.CostGrid(data, 1.0, 10.0, 20.0)
    np.testing.assert_array_equal(grid.cost([[10.5, 20.5], [11.5, 20.5], [10.5, 21.5], [11.5, 21.5], [0.0, 0.0]]),
                                  [0, 50, 0, 100, 0])
    assert len(geo.CostGrid.empty().cost([[0.0, 0.0]])) == 1


def test_lethal_points_are_the_centres_of_the_lethal_cells():
    data = np.zeros((4, 5), dtype=np.int16)
    data[1, 3], data[2, 0], data[3, 4] = 100, 100, 99          # the last is inscribed, not lethal
    grid = geo.CostGrid(data, 0.5, 10.0, 20.0)
    points = grid.lethal_points()
    np.testing.assert_allclose(sorted(points.tolist()), [[10.25, 21.25], [11.75, 20.75]])
    np.testing.assert_array_equal(grid.cost(points), [100, 100])
    turned = geo.CostGrid(data, 0.5, 10.0, 20.0, math.pi / 2.0)  # a rotated origin: the cells turn with it
    np.testing.assert_array_equal(turned.cost(turned.lethal_points()), [100, 100])
    assert geo.CostGrid.empty().lethal_points().shape == (0, 2)
