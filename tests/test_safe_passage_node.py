"""safe_passage_node._run with stubbed TrackPath, Encircle, TF and costmap: call order, retries and outcomes."""
import importlib.util
import math
import threading
import time
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
        self.costmap_waits = 0
        node._wait_for_costmap_since = lambda gh, since: setattr(self, "costmap_waits", self.costmap_waits + 1)

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
    # snapshots: entry centre, entry approach, entry re-snap, before (clean), middle, after (blocked)
    clean = geo.CostGrid.empty
    h = Harness(node, [ENTRY, gate, EXIT], grids=[clean(), clean(), clean(), clean(), blocked, blocked])
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


def watch_zone(h):
    """Records, with every snapshot, track and circle, whether a keep-out zone was active then."""
    node, snapshot, track, circle = h.node, h.node._snapshot, h.node._track, h.node._circle
    h.zone_seen = []

    def seen(kind):
        h.zone_seen.append((kind, node._zone_active))

    node._snapshot = lambda: seen("snapshot") or snapshot()
    node._track = lambda *a: seen("track") or track(*a)
    node._circle = lambda *a: seen("circle") or circle(*a)


def clear_zone_after(node, seconds):
    timer = threading.Timer(seconds, lambda: setattr(node, "_zone_active", False))
    timer.start()
    return timer


def test_zone_active_at_start_holds_everything_until_it_clears(node):
    h = Harness(node, [ENTRY, GATE1, EXIT])
    watch_zone(h)
    node._zone_active = True
    clear_zone_after(node, 0.3)
    start = time.monotonic()
    assert h.run() == (True, "all legs succeeded")
    assert time.monotonic() - start >= 0.3
    assert h.zone_seen and not any(active for _, active in h.zone_seen)
    assert h.costmap_waits == 1  # once, after the first hold


def test_zone_during_a_leg_holds_the_next_attempt_and_the_circle(node):
    h = Harness(node, [ENTRY, EXIT], track_results=[False, True])
    watch_zone(h)
    track = node._track

    def zone_on_first(*a):
        ok = track(*a)
        if len(h.tracks()) == 1:  # the zone lands while the first approach runs, which then fails
            node._zone_active = True
            clear_zone_after(node, 0.3)
        return ok

    node._track = zone_on_first
    start = time.monotonic()
    assert h.run() == (True, "all legs succeeded")
    assert time.monotonic() - start >= 0.3
    # the retry and everything after it waited for the zone to clear
    assert kinds(h) == ["track", "track", "circle", "track", "circle"]
    assert not any(active for _, active in h.zone_seen)
    assert h.costmap_waits == 1


def test_cancel_while_holding_for_a_zone(node):
    h = Harness(node, [ENTRY, EXIT])
    node._zone_active = True
    threading.Timer(0.2, lambda: setattr(h.goal, "is_cancel_requested", True)).start()
    with pytest.raises(spn.Canceled):
        h.run()
    assert h.calls == []


def test_mask_callback_sets_zone_active(node):
    from nav_msgs.msg import OccupancyGrid
    msg = OccupancyGrid()
    msg.data = [0, 0, 100, 0]
    node._on_keepout_mask(msg)
    assert node._zone_active
    msg.data = [0]
    node._on_keepout_mask(msg)
    assert not node._zone_active


def buoy_at(xy, size=0.3):
    """30 x 30 m grid at 0.1 m from (-5, -15), lethal within size of xy."""
    g = geo.CostGrid(np.zeros((300, 300), dtype=np.int16), 0.1, -5.0, -15.0)
    xs = np.arange(300) * 0.1 + 0.05
    gx, gy = np.meshgrid(xs - 5.0, xs - 15.0)
    g.data[np.hypot(gx - xy[0], gy - xy[1]) < size] = 100
    return g


def test_circle_resnaps_onto_a_buoy_seen_only_after_the_approach(node):
    # snapshots: entry centre (buoy out of range), entry approach, entry re-snap (seen 0.6 m off), exit x3
    seen = buoy_at(ENTRY + [0.6, 0.0])
    clean = geo.CostGrid.empty
    h = Harness(node, [ENTRY, EXIT], grids=[clean(), clean(), seen])
    assert h.run() == (True, "all legs succeeded")
    circles = [c for c in h.calls if c[0] == "circle"]
    assert np.allclose(circles[0][1], ENTRY + [0.6, 0.0], atol=0.05)  # the buoy, not the drone's estimate
    assert np.linalg.norm(h.tracks()[0][1] - ENTRY) == pytest.approx(4.0)  # approach still planned on the estimate


def test_circle_resnap_keeps_a_centre_already_snapped(node):
    seen = buoy_at(ENTRY + [0.6, 0.0])
    h = Harness(node, [ENTRY, EXIT], grids=[seen, seen, seen])
    assert h.run() == (True, "all legs succeeded")
    circles = [c for c in h.calls if c[0] == "circle"]
    assert np.allclose(circles[0][1], ENTRY + [0.6, 0.0], atol=0.05)


def test_circle_stays_on_the_estimate_when_the_buoy_is_never_seen(node):
    h = Harness(node, [ENTRY, EXIT])
    assert h.run() == (True, "all legs succeeded")
    circles = [c for c in h.calls if c[0] == "circle"]
    assert np.allclose(circles[0][1], ENTRY) and np.allclose(circles[1][1], EXIT)
    assert circles[0][2] == node.circle_radius


def test_exit_circle_resnaps_too(node):
    seen = buoy_at(EXIT - [30.0, 0.0] + [0.0, 0.5])  # grid spans x -5..25: shift the buoy into it below
    exit_seen = geo.CostGrid(seen.data, 0.1, 25.0, -15.0)  # same grid moved to x 25..55
    clean = geo.CostGrid.empty
    h = Harness(node, [ENTRY, EXIT], grids=[clean(), clean(), clean(), clean(), clean(), exit_seen])
    assert h.run() == (True, "all legs succeeded")
    circles = [c for c in h.calls if c[0] == "circle"]
    assert np.allclose(circles[0][1], ENTRY) and np.allclose(circles[1][1], EXIT + [0.0, 0.5], atol=0.05)


def test_costmap_wait_takes_only_a_costmap_stamped_after_the_clear(node):
    from nav_msgs.msg import OccupancyGrid
    from rclpy.duration import Duration
    goal = Goal()
    since = node.get_clock().now() + Duration(seconds=0.3)
    old = OccupancyGrid()
    old.header.stamp = node.get_clock().now().to_msg()
    node._costmap = old
    new = OccupancyGrid()

    def publish_new():
        new.header.stamp = node.get_clock().now().to_msg()
        node._costmap = new

    threading.Timer(0.5, publish_new).start()
    start = time.monotonic()
    node._wait_for_costmap_since(goal, since)
    assert 0.4 <= time.monotonic() - start < 1.5 and node._costmap is new


def test_costmap_wait_gives_up_without_costmaps(node):
    node.zone_clear_settle = 0.1
    node._costmap = None
    start = time.monotonic()
    node._wait_for_costmap_since(Goal(), node.get_clock().now())
    assert time.monotonic() - start == pytest.approx(2.1, abs=0.3)
