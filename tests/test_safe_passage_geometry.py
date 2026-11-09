"""Offline checks for the Safe Passage costmap sampling."""
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    'safe_passage_geometry', Path(__file__).parents[1] / 'frames/utils/safe_passage_geometry.py'
)
geo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geo)

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


def test_empty_grid_returns_nominal_answers():
    nominal = np.array([3.0, 4.0])
    assert np.allclose(geo.snap_centre(EMPTY, nominal, 1.0), nominal)
    assert geo.choose_radius(EMPTY, nominal, 2.5, 0.5) == 2.5
    assert np.allclose(geo.sample_point(EMPTY, nominal, 1.0), nominal)
    approach = geo.choose_approach(EMPTY, [0, 0], 2.5, 1.5, boat=[0, -10], clockwise=True, threshold=50)
    assert np.allclose(approach, [0.0, -4.0])  # straight between the boat and the buoy


def test_cost_lookup_handles_unknown_and_outside():
    grid = geo.CostGrid(np.array([[-1, 50], [100, 0]], dtype=np.int16), 1.0, 0.0, 0.0)
    assert list(grid.cost([[0.5, 0.5], [1.5, 0.5], [0.5, 1.5], [5.0, 5.0], [-1.0, 0.5]])) == [0, 50, 100, 0, 0]


def test_snap_moves_a_biased_centre_onto_the_lidar_buoy():
    grid = grid_with([(1.0, 0.0)])
    snapped = geo.snap_centre(grid, [0.2, 0.0], 1.0)
    assert np.linalg.norm(snapped - [1.0, 0.0]) < 0.1
    # No bias: stays put.
    assert np.linalg.norm(geo.snap_centre(grid, [1.0, 0.0], 1.0) - [1.0, 0.0]) < 0.1


def test_radius_avoids_a_neighbouring_buoy():
    grid = grid_with([(0.0, 0.0), (0.0, 3.3)])  # neighbour 3.3 m away clips the 3.0 m circle
    r = geo.choose_radius(grid, [0.0, 0.0], 2.5, 0.5)
    assert r <= 2.25
    assert geo.choose_radius(grid_with([(0.0, 0.0)]), [0.0, 0.0], 2.5, 0.5) == 2.5


def test_tangent_points_follow_the_direction():
    # Boat due south at 2R: clockwise (seen from above) heads west first, so it joins on the west side.
    cw = geo.tangent_point([0, 0], 1.0, [0, -2], clockwise=True)
    ccw = geo.tangent_point([0, 0], 1.0, [0, -2], clockwise=False)
    assert cw[0] < 0 < ccw[0]
    for p in (cw, ccw):
        assert math.isclose(np.linalg.norm(p), 1.0)
        assert abs(np.dot(p, np.array([0, -2]) - p)) < 1e-9  # radius is perpendicular to the run in
    assert np.allclose(geo.tangent_point([0, 0], 2.0, [0, -1], clockwise=True), [0, -2])  # inside: radial


def test_approach_skips_a_blocked_candidate():
    grid = grid_with([(0.0, -4.0)])  # obstacle on the ring between the boat and the buoy
    point = geo.choose_approach(grid, [0, 0], 2.5, 1.5, boat=[0, -10], clockwise=True, threshold=50)
    assert np.linalg.norm(point) == pytest.approx(4.0)
    assert grid.cost(point)[0] < 50 and not np.allclose(point, [0, -4])


def test_waypoint_sample_stays_put_when_clear_and_moves_when_close():
    gate = grid_with([(-1.5, 0.0), (1.5, 0.0)])
    assert np.allclose(geo.sample_point(gate, [0.0, 0.0], 1.0), [0.0, 0.0])
    moved = geo.sample_point(gate, [0.9, 0.0], 1.0)  # biased into the inflation of the right buoy
    assert gate.cost(moved)[0] < gate.cost([0.9, 0.0])[0] and moved[0] < 0.9


def test_standoff_sample_never_crosses_its_buoy():
    # Buoy at origin, travel +y, standoff 1.5 m to the left; nothing may land at x >= 0.
    grid = grid_with([(0.0, 0.0), (-2.6, 0.0)])
    for nominal in ([-1.5, 0.0], [-1.5, 0.5], [-1.5, -0.5]):
        assert geo.sample_point(grid, nominal, 1.0)[0] < 0.0


def test_circles_are_one_round_plus_overshoot():
    assert geo.circle_turn_deg(clockwise=True, overshoot_deg=20.0) == 380.0  # entry
    assert geo.circle_turn_deg(clockwise=False, overshoot_deg=20.0) == -380.0  # exit


def test_crossing_direction():
    assert np.allclose(geo.crossing_direction([0, 0], [0, 4]), [0, 1])
    assert np.allclose(geo.crossing_direction([1, 1], [1, 1]), [1, 0])
