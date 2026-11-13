"""buoy_align: costmap blobs, the exhaustive shift search, the lock and the estimates."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
from frames.utils import buoy_align as ba  # noqa: E402
from frames.utils import safe_passage_geometry as geo  # noqa: E402

RES = 0.1
LAYOUT = np.array([[0.0, 0.0], [5.0, 1.0], [9.0, -2.0], [3.0, 6.0], [8.0, 5.0], [12.0, 3.0], [1.0, -5.0],
                   [6.0, -6.0], [11.0, -5.0], [14.0, 7.0]])
SHIFT = np.array([3.0, -5.5])


def grid_with(patches, size=60.0, origin=(-20.0, -30.0)):
    """A costmap with lethal patches: (x, y, width, height) in metres, centred on x, y."""
    n = int(size / RES)
    data = np.zeros((n, n), dtype=np.int16)
    for x, y, w, h in patches:
        c0, c1 = int(round((x - w / 2 - origin[0]) / RES)), int(round((x + w / 2 - origin[0]) / RES))
        r0, r1 = int(round((y - h / 2 - origin[1]) / RES)), int(round((y + h / 2 - origin[1]) / RES))
        data[r0:max(r1, r0 + 1), c0:max(c1, c0 + 1)] = 100
    return geo.CostGrid(data, RES, origin[0], origin[1])


def buoys_at(points, size=0.4):
    return [(x, y, size, size) for x, y in points]


def all_blobs(grid, p=ba.AlignParams()):
    return ba.blobs(grid, LAYOUT, p.max_shift + p.field_margin, p.join_gap)


def test_blobs_size_and_centre():
    found = ba.blobs(grid_with([(2.0, 3.0, 0.4, 0.3)]), [[0, 0]], 20.0, 0.6)
    assert len(found) == 1
    np.testing.assert_allclose(found[0, :2], [2.0, 3.0], atol=0.06)
    assert found[0, 2] == 12 and abs(found[0, 3] - 0.4) < 1e-6


def test_blobs_join_across_small_gaps_only():
    near = ba.blobs(grid_with([(0.0, 0.0, 0.3, 0.3), (0.7, 0.0, 0.3, 0.3)]), [[0, 0]], 20.0, 0.6)  # 0.4 m gap
    far = ba.blobs(grid_with([(0.0, 0.0, 0.3, 0.3), (2.0, 0.0, 0.3, 0.3)]), [[0, 0]], 20.0, 0.6)   # buoy spacing
    assert len(near) == 1 and abs(near[0, 3] - 1.0) < 1e-6
    assert len(far) == 2


def test_structure_with_specks_is_one_large_blob():
    # A pad seen as an elbow (two thin strokes at 90 deg) with a speck beside it
    pad = [(0.0, 1.0, 0.1, 2.0), (1.0, 0.0, 2.0, 0.1), (0.4, 1.5, 0.2, 0.2)]
    found = ba.blobs(grid_with(pad), [[0, 0]], 20.0, 0.6)
    assert len(found) == 1 and found[0, 3] >= 2.0
    aligner = ba.Aligner([[0.4, 1.5]], ba.AlignParams())
    aligner.update(found)
    assert len(aligner.buoy_blobs) == 0 and len(aligner.structures) == 1
    assert aligner.estimate(0).source == ba.PREDICTED  # never snaps onto a structure


def test_blobs_outside_the_reach_are_not_read():
    found = ba.blobs(grid_with([(25.0, 25.0, 0.4, 0.4), (1.0, 1.0, 0.4, 0.4)]), [[0, 0]], 5.0, 0.6)
    assert len(found) == 1


def test_solve_finds_a_large_shift_despite_missing_and_extra_objects():
    truth = LAYOUT + SHIFT
    seen = np.delete(truth, [2, 7], axis=0)                          # two drone buoys have no blob
    extras = np.array([[30.0, 20.0], [-10.0, 15.0], [20.0, -20.0], [0.0, 10.0], [25.0, -3.0]])
    pad = [(22.0, 12.0, 0.1, 2.4), (23.2, 10.8, 2.4, 0.1)]
    p = ba.AlignParams()
    aligner = ba.Aligner(LAYOUT, p)
    fresh = aligner.update(all_blobs(grid_with(buoys_at(np.vstack([seen, extras])) + pad)))
    assert fresh.accepted, fresh
    np.testing.assert_allclose(fresh.t, SHIFT, atol=0.1)
    assert fresh.matched == 8 and fresh.score > 1.5 * fresh.second
    assert len(aligner.structures) == 1


def test_solve_rejects_a_repeating_pattern():
    row = np.array([[2.0 * k, 0.0] for k in range(4)])
    field = np.array([[2.0 * k, 3.0] for k in range(-2, 8)])       # the row fits at many shifts equally
    fresh = ba.solve(row, field, ba.AlignParams())
    assert not fresh.accepted and fresh.second >= 0.9 * fresh.score


def test_solve_rejects_too_little_evidence():
    assert not ba.solve(LAYOUT[:2], LAYOUT[:2] + SHIFT, ba.AlignParams()).accepted
    assert not ba.solve(LAYOUT, np.zeros((0, 2)), ba.AlignParams()).accepted


def test_aligner_uses_the_shift_only_once_locked():
    p = ba.AlignParams()
    aligner = ba.Aligner(LAYOUT, p)
    found = all_blobs(grid_with(buoys_at(LAYOUT + SHIFT)))
    for n in range(p.lock_solves):
        assert not aligner.locked
        np.testing.assert_allclose(aligner.t, 0.0)
        aligner.update(found)
    assert aligner.locked and aligner.matched == len(LAYOUT)
    np.testing.assert_allclose(aligner.t, SHIFT, atol=0.1)


def test_locked_shift_survives_a_bad_costmap_and_follows_small_moves():
    p = ba.AlignParams()
    aligner = ba.Aligner(LAYOUT, p)
    found = all_blobs(grid_with(buoys_at(LAYOUT + SHIFT)))
    for _ in range(p.lock_solves):
        aligner.update(found)
    aligner.update(np.zeros((0, 4)))                                  # costmap cleared around the boat
    assert aligner.locked and aligner.matched == 0
    np.testing.assert_allclose(aligner.t, SHIFT, atol=0.1)
    aligner.update(all_blobs(grid_with(buoys_at(LAYOUT + SHIFT + [6.0, 0.0]))))   # a different layout fit: ignored
    np.testing.assert_allclose(aligner.t, SHIFT, atol=0.1)
    moved = all_blobs(grid_with(buoys_at(LAYOUT + SHIFT + [0.4, 0.0])))          # localisation drift: followed
    for _ in range(p.median_window):
        aligner.update(moved)
    np.testing.assert_allclose(aligner.t, SHIFT + [0.4, 0.0], atol=0.1)


def test_estimates_snap_to_the_blob_within_reach_and_respect_claims():
    p = ba.AlignParams()
    aligner = ba.Aligner(LAYOUT, p)
    truth = LAYOUT + SHIFT
    truth[0] += [0.6, 0.0]                                            # this buoy sits off the layout
    found = all_blobs(grid_with(buoys_at(np.delete(truth, 3, axis=0))))
    for _ in range(p.lock_solves):
        aligner.update(found)
    e = aligner.estimate(0)
    assert e.source == ba.BLOB and np.linalg.norm(e.xy - truth[0]) < 0.1
    assert aligner.estimate(3).source == ba.PREDICTED                 # no blob there
    np.testing.assert_allclose(aligner.estimate(3).xy, LAYOUT[3] + aligner.t)
    assert aligner.estimate(0, claimed=[truth[0]]).source == ba.PREDICTED
    assert aligner.search_radius(0) <= p.snap_radius


def test_search_radius_is_capped_by_the_neighbour_spacing():
    aligner = ba.Aligner([[0.0, 0.0], [1.2, 0.0], [20.0, 0.0]], ba.AlignParams())
    assert abs(aligner.search_radius(0) - 0.6) < 1e-9 and aligner.search_radius(2) == 1.0
