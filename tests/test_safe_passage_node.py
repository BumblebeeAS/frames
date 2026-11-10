"""safe_passage_node._run with stubbed TrackPath, Encircle, TF and costmap: call order, retries and outcomes."""
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("bb_robotx_msgs")

ROOT = Path(__file__).parents[1]
geo_spec = importlib.util.spec_from_file_location("frames.utils.safe_passage_geometry",
                                                  ROOT / "frames/utils/safe_passage_geometry.py")
geo = importlib.util.module_from_spec(geo_spec)
geo_spec.loader.exec_module(geo)
spec = importlib.util.spec_from_file_location("safe_passage_node", ROOT / "scripts/safe_passage_node.py")
spn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(spn)

ENTRY, EXIT = np.array([0.0, 0.0]), np.array([30.0, 0.0])
GATE1, GATE2 = np.array([10.0, 5.0]), np.array([20.0, 5.0])


class Goal:
    def __init__(self):
        self.is_cancel_requested = False
        self.request = type("Request", (), {"path_topic": ""})()

    def publish_feedback(self, _):
        pass


class Harness:
    """Stubs the node's I/O. track_results: per-call bools (default True); grids: per-snapshot CostGrids."""

    def __init__(self, node, points, track_results=(), grids=(), boat=(0.0, -10.0)):
        self.node, self.calls, self.goal = node, [], Goal()
        self.track_results, self.grids, self.boat = list(track_results), list(grids), np.array(boat)
        node._track = self._track
        node._circle = lambda gh, c, r, t: self.calls.append(("circle", np.round(c, 2), r, t)) or True
        node._boat_xy = lambda: None if self.boat is None else self.boat
        node._snapshot = lambda: self.grids.pop(0) if self.grids else geo.CostGrid.empty()
        node._wait_for_path = lambda gh, topic: object()
        node._from_ll = lambda path: [np.asarray(p, dtype=float) for p in points]

    def _track(self, gh, xy, yaw, tol, stab):
        self.node._check_cancel(gh)
        xy = np.asarray(xy, dtype=float)
        self.calls.append(("track", xy.copy(), math.degrees(yaw)))
        ok = self.track_results.pop(0) if self.track_results else True
        if ok:
            self.boat = xy
        return ok

    def run(self):
        return self.node._run(self.goal)

    def tracks(self):
        return [c for c in self.calls if c[0] == "track"]


@pytest.fixture(scope="module")
def ros():
    rclpy.init()
    yield
    rclpy.try_shutdown()


@pytest.fixture
def node(ros):
    n = spn.SafePassageNode()
    yield n
    n.destroy_node()


def kinds(h):
    return [c[0] for c in h.calls]


def test_all_legs_succeed_in_order(node):
    h = Harness(node, [ENTRY, GATE1, GATE2, EXIT])
    assert h.run() == (True, "all legs succeeded")
    # approach, circle, 3 legs x 2 gates, approach, circle
    assert kinds(h) == ["track", "circle"] + ["track"] * 6 + ["track", "circle"]
    assert h.calls[1][3] == 380.0 and h.calls[-1][3] == -380.0
    legs = [c[1] for c in h.tracks()[1:7]]
    d1 = geo.crossing_direction(ENTRY, GATE2)
    assert np.allclose(legs[:3], [GATE1 - d1, GATE1, GATE1 + d1])  # 1 m lead, clean costmap
    for leg in h.tracks()[1:4]:
        assert leg[2] == pytest.approx(math.degrees(math.atan2(d1[1], d1[0])))  # shared crossing yaw


def test_approach_retry_moves_round_the_ring(node):
    h = Harness(node, [ENTRY, EXIT], track_results=[False, True])
    ok, msg = h.run()
    assert ok, msg
    first, second = h.tracks()[0][1], h.tracks()[1][1]
    assert np.linalg.norm(first) == pytest.approx(4.0) and np.linalg.norm(second) == pytest.approx(4.0)
    assert np.linalg.norm(second - first) > node.approach_exclude_radius


def test_approach_failing_every_attempt_still_circles(node):
    h = Harness(node, [ENTRY, EXIT], track_results=[False, False, False])
    ok, msg = h.run()
    assert not ok and msg == "failed: ENTRY_APPROACH"
    targets = [c[1] for c in h.tracks()[:3]]
    assert all(np.linalg.norm(a - b) > 1.0 for i, a in enumerate(targets) for b in targets[i + 1:])  # 3 distinct
    assert kinds(h)[3] == "circle"


def test_leg_retry_resends_the_same_target_on_a_clean_costmap(node):
    # approach ok, circle, before ok, middle fails twice then succeeds
    h = Harness(node, [ENTRY, GATE1, EXIT], track_results=[True, True, False, False, True])
    ok, msg = h.run()
    assert ok, msg
    middles = [c[1] for c in h.tracks()[2:5]]
    assert all(np.allclose(m, GATE1) for m in middles)


def test_leg_failing_every_attempt_is_skipped(node):
    h = Harness(node, [ENTRY, GATE1, EXIT], track_results=[True, True, False, False, False])
    ok, msg = h.run()
    assert not ok and msg == "failed: WAYPOINT 1 leg 2"
    assert kinds(h).count("track") == 1 + 1 + 3 + 1 + 1  # approach, before, 3x middle, after, exit approach


def test_middle_stays_ahead_when_the_costmap_changes_between_legs(node):
    # Clean snapshot for the before leg, then the lidar sees a return just ahead of the midpoint
    blocked = geo.CostGrid(np.zeros((200, 200), dtype=np.int16), 0.1, 0.0, -5.0)
    d = geo.crossing_direction(ENTRY, EXIT)  # (1, 0): gate on the entry-exit line
    gate = np.array([10.0, 0.0])
    xs = np.arange(200) * 0.1 + 0.05
    gx, gy = np.meshgrid(xs, xs - 5.0)
    blocked.data[np.hypot(gx - 10.3, gy) < 0.5] = 100
    # snapshots: entry centre, entry approach, before (clean), middle, after (blocked)
    clean = geo.CostGrid.empty
    h = Harness(node, [ENTRY, gate, EXIT], grids=[clean(), clean(), clean(), blocked, blocked])
    ok, msg = h.run()
    assert ok, msg
    before, middle, after = (c[1] for c in h.tracks()[1:4])
    assert np.allclose(before, gate - d)  # sampled on the clean snapshot
    assert not np.allclose(middle, gate) and blocked.cost(middle)[0] == 0  # moved off the new return
    assert np.dot(middle - before, d) >= node.lead_distance - node.sample_radius - 1e-9
    assert np.dot(after - middle, d) > 0


def test_no_lead_drives_the_middle_only(node):
    node.lead_distance = 0.0
    h = Harness(node, [ENTRY, GATE1, EXIT])
    assert h.run()[0]
    assert kinds(h) == ["track", "circle", "track", "track", "circle"]
    assert np.allclose(h.tracks()[1][1], GATE1)


def test_no_retries_means_one_attempt(node):
    node.track_retries = 0
    h = Harness(node, [ENTRY, EXIT], track_results=[False])
    assert h.run() == (False, "failed: ENTRY_APPROACH")
    assert kinds(h) == ["track", "circle", "track", "circle"]


def test_no_boat_pose_skips_the_approach_but_circles(node):
    h = Harness(node, [ENTRY, EXIT], boat=None)
    h.boat = None
    ok, msg = h.run()
    assert msg == "failed: ENTRY_APPROACH, EXIT_APPROACH"
    assert kinds(h) == ["circle", "circle"]


def test_cancel_during_an_attempt_stops_everything(node):
    h = Harness(node, [ENTRY, GATE1, EXIT])

    def cancel_on_second(gh, xy, yaw, tol, stab):
        h.calls.append(("track", np.asarray(xy), yaw))
        if len(h.tracks()) == 2:
            gh.is_cancel_requested = True
        node._check_cancel(gh)
        return True

    node._track = cancel_on_second
    with pytest.raises(spn.Canceled):
        h.run()
    assert kinds(h) == ["track", "circle", "track"]


def test_entry_and_exit_only(node):
    h = Harness(node, [ENTRY, EXIT])
    assert h.run() == (True, "all legs succeeded")
    assert kinds(h) == ["track", "circle", "track", "circle"]
    # approach heading points at the clockwise tangent point
    point, yaw = h.tracks()[0][1], h.tracks()[0][2]
    run = geo.tangent_point(ENTRY, 2.5, point, clockwise=True) - point
    assert yaw == pytest.approx(math.degrees(math.atan2(run[1], run[0])))
