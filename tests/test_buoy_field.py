"""Offline checks for the Safe Passage buoy field: clustering, landmarks, shift, estimates."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from frames.utils import buoy_field as bf  # noqa: E402
from frames.utils import safe_passage_geometry as geo  # noqa: E402

RNG = np.random.default_rng(0)
# A field: entry, two gates, exit, a black buoy
CANDIDATES = np.array([[0, 0], [10, 2], [10, -2], [20, 2], [20, -2], [30, 0], [15, 6]], dtype=float)


def buoy_points(centre, n=12, half=0.2):
    """Returns off two faces of a 0.4 m box, as the lidar sees it from one side."""
    centre = np.asarray(centre, dtype=float)
    face_a = np.stack([np.full(n // 2, -half), np.linspace(-half, half, n // 2)], 1)
    face_b = np.stack([np.linspace(-half, half, n // 2), np.full(n // 2, -half)], 1)
    return centre + np.vstack([face_a, face_b]) + RNG.normal(0, 0.02, (n, 2))


def scan(centres, extra=()):
    parts = [buoy_points(c) for c in centres] + [np.asarray(e, dtype=float) for e in extra]
    return np.vstack(parts) if parts else np.zeros((0, 2))


def test_remove_box():
    points = np.array([[0.5, 0.2, 0.3], [1.0, 0.0, 0.3], [0.0, 0.8, 0.0]])
    np.testing.assert_allclose(bf.remove_box(points, 0.9, 0.75), points[1:])


def test_cluster_and_extent():
    xy = np.vstack([buoy_points([0, 0]), buoy_points([3, 0]), [[10, 10]]])
    groups = bf.cluster(xy, 0.3, 3)
    assert len(groups) == 2
    assert all(0.3 < bf.extent(xy[g]) < 0.7 for g in groups)
    assert bf.cluster(np.zeros((0, 2)), 0.3, 3) == []


def test_derived_values_from_candidate_spacing():
    field = bf.BuoyField(CANDIDATES, bf.FieldParams())
    assert field.nn[1] == pytest.approx(4.0)  # the gate pair
    assert field.max_shift == pytest.approx(min(4.0, 0.8 * 4.0))
    assert field.crop_r == pytest.approx(field.max_shift + 1.5)
    assert field.search_radius(1) == pytest.approx(min(2.0, 0.5 * 4.0))


def test_detect_keeps_buoy_sized_clusters_inside_the_field_only():
    field = bf.BuoyField(CANDIDATES, bf.FieldParams())
    wall = np.stack([np.linspace(8, 12, 60), np.full(60, 4.0)], 1)  # 4 m long: too big
    far = buoy_points([60, 60])  # outside the field
    speck = np.array([[5.0, 5.0]])  # too few points
    detections, stats = field.detect(np.vstack([scan([[10.4, 2.3]]), wall, far, speck]))
    assert stats["detections"] == 1 and len(detections) == 1
    np.testing.assert_allclose(detections[0], [10.4, 2.3], atol=0.25)
    assert stats["field"] < stats["points"]


def test_detect_rejects_clusters_cut_by_the_field_edge():
    field = bf.BuoyField([[0, 0]], bf.FieldParams())  # lone candidate: crop_r = 4 + 1.5
    edge = buoy_points([field.crop_r - 0.1, 0])
    detections, _ = field.detect(edge)
    assert detections == []


def test_landmarks_need_hits_and_expire():
    field = bf.BuoyField(CANDIDATES, bf.FieldParams())
    for k in range(3):
        field.update([np.array([10.3, 2.2]) + RNG.normal(0, 0.05, 2)], now=float(k))
    assert len(field.landmarks) == 1 and len(field.usable()) == 1
    field.update([], now=100.0)
    assert field.landmarks == []


def test_shift_needs_three_agreeing_buoys():
    params = bf.FieldParams()
    field = bf.BuoyField(CANDIDATES, params)
    t = np.array([1.2, -0.8])
    seen = CANDIDATES[:2] + t
    for k in range(3):
        field.update(list(seen), now=float(k))
    assert not field.shift.accepted and field.shift.best_inliers == 2  # two agree: not enough
    seen = CANDIDATES[:4] + t
    for k in range(3, 6):
        field.update(list(seen), now=float(k))
    assert field.shift.accepted and field.shift.inliers == 4
    np.testing.assert_allclose(field.shift.t, t, atol=0.1)
    assert field.search_radius(0) == pytest.approx(1.0)


def test_shift_ignores_one_wrong_landmark():
    field = bf.BuoyField(CANDIDATES, bf.FieldParams())
    t = np.array([-0.5, 0.7])
    seen = list(CANDIDATES[[0, 1, 3, 5]] + t) + [np.array([12.0, 4.0])]  # one lookalike nowhere near the layout
    for k in range(3):
        field.update(seen, now=float(k))
    assert field.shift.accepted and field.shift.inliers == 4
    np.testing.assert_allclose(field.shift.t, t, atol=0.1)


def test_estimate_priority_landmark_then_costmap_then_predicted():
    field = bf.BuoyField(CANDIDATES, bf.FieldParams())
    for k in range(3):
        field.update([np.array([0.6, 0.4])], now=float(k))
    e = field.estimate(0, geo.CostGrid.empty())
    assert e.source == bf.LANDMARK and np.allclose(e.xy, [0.6, 0.4], atol=0.05)
    # Claimed by an earlier buoy: falls through
    assert field.estimate(0, geo.CostGrid.empty(), claimed=[[0.6, 0.4]]).source == bf.PREDICTED
    # Costmap blob near the candidate
    data = np.zeros((100, 100), dtype=np.int16)
    data[52:55, 63:66] = 100  # cells around (10.45, 0.35) with origin (4, -5) at 0.1 m: x 10.3-10.6, y 0.2-0.5
    grid = geo.CostGrid(data, 0.1, 4.0, -5.0)
    e = field.estimate(2, grid)  # candidate (10, -2), r = 2: blob 2.4 m away is outside
    assert e.source == bf.PREDICTED
    data[30:33, 60:63] = 100  # around (10.15, -1.85)
    e = field.estimate(2, geo.CostGrid(data, 0.1, 4.0, -5.0))
    assert e.source == bf.COSTMAP and np.allclose(e.xy, [10.15, -1.85], atol=0.1)


def test_estimate_landmark_outside_search_radius_is_not_used():
    field = bf.BuoyField(CANDIDATES, bf.FieldParams())
    for k in range(3):
        field.update([np.array([10.0, 0.1])], now=float(k))  # between the gate buoys: 1.9 m from each
    e = field.estimate(1, geo.CostGrid.empty())
    assert e.source == bf.LANDMARK and e.radius == pytest.approx(2.0)
    field.p.snap_spacing_factor = 0.4  # r = 1.6: now out of reach
    assert field.estimate(1, geo.CostGrid.empty()).source == bf.PREDICTED


def test_end_to_end_scans_lock_onto_shifted_buoys():
    field = bf.BuoyField(CANDIDATES, bf.FieldParams())
    t = np.array([1.5, 1.0])
    truth = CANDIDATES + t
    for k in range(4):
        detections, _ = field.detect(scan(truth))
        field.update(detections, now=float(k))
    assert field.shift.accepted and field.shift.inliers == len(CANDIDATES)
    for i in range(len(CANDIDATES)):
        e = field.estimate(i, geo.CostGrid.empty())
        assert e.source == bf.LANDMARK
        assert np.linalg.norm(e.xy - truth[i]) < 0.25  # face bias toward the lidar is about 0.1 m here
