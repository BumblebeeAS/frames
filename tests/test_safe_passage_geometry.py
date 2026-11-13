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
def test_arc_of_half_the_steps_is_a_semicircle_on_the_same_side(sigma):
    full = geo.circle_points([1.0, 2.0], 2.5, 0.3, sigma, 6, overshoot=1)
    half = geo.circle_points([1.0, 2.0], 2.5, 0.3, sigma, 6, overshoot=1, steps=3)
    assert len(full) == 8 and len(half) == 4
    for (a, ya), (b, yb) in zip(half, full):                 # the same first points, so the same direction
        np.testing.assert_allclose(a, b)
        assert ya == yb
    np.testing.assert_allclose(half[0][0] + half[-1][0], [2.0, 4.0], atol=1e-9)   # ends opposite the start


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


def test_course_runs_from_the_entry_to_the_exit_or_the_middle_of_the_reds_and_greens():
    P = [[0, 0], [10, 2], [10, -2], [20, 0]]
    start, d = geo.course(P, ["flashing_blue", "flashing_red", "flashing_green", "steady_blue"], [50, 50])
    np.testing.assert_allclose(start, [0, 0])
    np.testing.assert_allclose(d, [1, 0])
    start, d = geo.course(P[:3], ["flashing_blue", "flashing_red", "flashing_green"], [50, 50])
    np.testing.assert_allclose(d, [1, 0])                          # toward the middle of the red and the green
    start, d = geo.course(P[1:3], ["flashing_red", "flashing_green"], [10, -30])
    np.testing.assert_allclose(start, [10, -30])                   # no entry: from the boat
    np.testing.assert_allclose(d, [0, 1])
    assert geo.course([[0, 0]], ["flashing_blue"], [5, 5])[1] is None


def test_pass_waypoint_keeps_red_on_starboard_and_green_on_port():
    east = [1.0, 0.0]
    np.testing.assert_allclose(geo.pass_waypoint([5, 5], "flashing_red", east, 2.0), [5, 7])     # north of it
    np.testing.assert_allclose(geo.pass_waypoint([5, 5], "flashing_green", east, 2.0), [5, 3])   # south of it
    for state, want in (("flashing_red", -1), ("flashing_green", 1)):
        w = geo.pass_waypoint([5, 5], state, east, 2.0)
        to_buoy = np.array([5, 5]) - w
        side = east[0] * to_buoy[1] - east[1] * to_buoy[0]         # > 0: the buoy is on the boat's left (port)
        assert np.sign(side) == want


def test_gates_are_pairs_that_face_the_course_and_fit_the_width():
    east = [1.0, 0.0]
    states = ["flashing_red", "flashing_green", "flashing_red", "flashing_green", "off"]
    # red south and green north of the course line: crossed going east with the red on starboard
    assert geo.find_gates([[10, -2], [10, 2], [40, 40], [60, 60], [10, 0]], states, east, 2.4, 6.0, math.radians(60)) \
        == [(0, 1)]
    # the other way round only works going west: not a gate
    limits = (2.4, 6.0, math.radians(60))
    assert geo.find_gates([[10, 2], [10, -2], [40, 40], [60, 60], [0, 0]], states, east, *limits) == []
    # too narrow to go between, and too wide to be one manoeuvre
    assert geo.find_gates([[10, -1], [10, 1], [40, -4], [40, 4], [0, 0]], states, east, *limits) == []
    # each buoy in one gate only, the narrowest pairs first
    got = geo.find_gates([[10, -2], [10, 2], [10.5, -2.5], [11, 2.6], [0, 0]], states, east, *limits)
    assert got == [(0, 1), (2, 3)]
