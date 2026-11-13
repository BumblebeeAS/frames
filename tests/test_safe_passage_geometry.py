"""Offline checks for the Safe Passage geometry: order, circle points and radius."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from frames.utils import safe_passage_geometry as geo  # noqa: E402

RES = 0.1


def grid_with(buoys=(), inflation=1.0, size=40.0):
    """size x size m grid centred on 0, lethal disc r=0.3 per buoy and a linear inflation ramp to 0 at +inflation."""
    n = int(size / RES)
    data = np.zeros((n, n), dtype=np.int16)
    xs = -size / 2 + (np.arange(n) + 0.5) * RES
    gx, gy = np.meshgrid(xs, xs)
    for bx, by in buoys:
        d = np.hypot(gx - bx, gy - by) - 0.3
        ramp = np.where(d <= 0, 100, np.clip(98 * (1 - d / inflation), 0, 98)).astype(np.int16)
        data = np.maximum(data, ramp)
    return geo.CostGrid(data, RES, -size / 2, -size / 2)


EMPTY = geo.CostGrid.empty()


def test_order_entry_middles_by_projection_exit_and_no_blacks():
    positions = [[30, 0], [20, 3], [0, 0], [5, -2], [12, 1], [10, 10]]
    states = [geo.EXIT, geo.GREEN, geo.ENTRY, geo.RED, geo.BLACK, geo.RED]
    assert geo.order_buoys(positions, states, boat=[-10, 0]) == [2, 3, 5, 1, 0]


def test_order_without_entry_starts_from_the_boat_and_without_exit_heads_for_the_centroid():
    positions = [[20, 0], [10, 0], [30, 0]]
    states = [geo.RED, geo.GREEN, geo.RED]
    assert geo.order_buoys(positions, states, boat=[0, 0]) == [1, 0, 2]
    assert geo.order_buoys(positions, states, boat=[40, 0]) == [2, 0, 1]


def test_order_nearest_next_when_start_and_end_coincide():
    positions = [[1, 0], [-1, 0], [5, 0]]
    states = [geo.RED, geo.GREEN, geo.RED]  # centroid (1.67, 0) sits 0.67 m from the boat at (1, 0)
    assert geo.order_buoys(positions, states, boat=[1.0, 0.0]) == [0, 1, 2]


def test_nearest_other():
    np.testing.assert_allclose(geo.nearest_other([[0, 0], [3, 0], [3, 4]]), [3, 3, 4])
    assert geo.nearest_other([[1, 1]])[0] == math.inf


@pytest.mark.parametrize("sigma", [1, -1])
def test_circle_points_go_round_in_the_right_direction_with_travel_yaw(sigma):
    points = geo.circle_points([1.0, 2.0], 2.5, 0.0, sigma, 8, overshoot=1)
    assert len(points) == 10
    xy = np.array([p for p, _ in points])
    np.testing.assert_allclose(np.linalg.norm(xy - [1.0, 2.0], axis=1), 2.5)
    np.testing.assert_allclose(xy[0], [3.5, 2.0])
    np.testing.assert_allclose(xy[8], xy[0], atol=1e-9)  # 360 deg, then one more
    # Clockwise from east goes south first; counterclockwise goes north
    assert np.sign(xy[1][1] - 2.0) == -sigma
    for (p, yaw), (q, _) in zip(points, points[1:]):
        travel = math.atan2(q[1] - p[1], q[0] - p[0])
        assert abs(math.remainder(travel - yaw, 2 * math.pi)) < math.radians(23)  # chord vs tangent: half a step


def test_start_angle_is_the_tangent_point_ahead_in_the_direction():
    # Boat due east, 10 m out: clockwise joins south of east, counterclockwise north of east
    assert geo.start_angle([0, 0], 2.5, [10, 0], 1) < 0 < geo.start_angle([0, 0], 2.5, [10, 0], -1)
    assert geo.start_angle([0, 0], 2.5, [0, 1], 1) == pytest.approx(math.pi / 2)  # inside: radially


def test_radius_nominal_on_open_water():
    assert geo.choose_radius(EMPTY, [0, 0], [], 2.5, 1.6, 0.25, 1.25, 22) == (2.5, "nominal")


def test_radius_capped_by_neighbour_and_floored():
    r, why = geo.choose_radius(EMPTY, [0, 0], [[3.5, 0], [10, 0]], 2.5, 1.6, 0.25, 1.25, 22)
    assert r == pytest.approx(2.25) and "neighbour" in why
    r, why = geo.choose_radius(EMPTY, [0, 0], [[2.0, 0]], 2.5, 1.6, 0.25, 1.25, 22)
    assert r == 1.6 and "minimum" in why


def test_radius_shrinks_off_a_marked_neighbour_the_drone_missed():
    grid = grid_with([(0, 0), (2.5, 0)])  # a buoy sitting on the nominal circle
    r, why = geo.choose_radius(grid, [0, 0], [], 2.5, 1.0, 0.25, 1.25, 22)
    assert r < 2.5 and "shrunk" in why
    angles = geo.circle_angles(16)
    assert grid.cost(np.stack([r * np.cos(angles), r * np.sin(angles)], 1)).max() < 22


def test_radius_reports_still_blocked_at_minimum():
    grid = grid_with([(0, 0), (1.6, 0), (2.0, 0), (2.5, 0)])
    r, why = geo.choose_radius(grid, [0, 0], [], 2.5, 1.6, 0.25, 1.25, 22)
    assert r == pytest.approx(1.6) and "still blocked" in why


def test_lethal_centroid():
    grid = grid_with([(3.0, 4.0)])
    np.testing.assert_allclose(geo.lethal_centroid(grid, [3.5, 4.0], 1.0), [3.0, 4.0], atol=0.06)
    assert geo.lethal_centroid(grid, [10, 10], 1.0) is None
    assert geo.lethal_centroid(EMPTY, [0, 0], 1.0) is None
