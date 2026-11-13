"""safe_passage_node with stubbed TrackPath, TF, fromLL and costmap: order, directions, retries, pause, lidar."""
import math
import sys
import threading
import time
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("bb_robotx_msgs")
from bb_robotx_msgs.msg import SafePassageBuoy  # noqa: E402

if "robot_localization" not in sys.modules:
    try:
        import robot_localization.srv  # noqa: F401
    except ImportError:  # only fromLL's type is needed, and _from_ll is stubbed below
        from std_srvs.srv import Trigger
        package = types.ModuleType("robot_localization")
        package.srv = types.ModuleType("robot_localization.srv")
        package.srv.FromLL = Trigger
        sys.modules["robot_localization"], sys.modules["robot_localization.srv"] = package, package.srv

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
import importlib.util  # noqa: E402

from frames.utils import safe_passage_geometry as geo  # noqa: E402

spec = importlib.util.spec_from_file_location("safe_passage_node", ROOT / "scripts/safe_passage_node.py")
spn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(spn)

ENTRY, RED, GREEN, EXIT, BLACK = [0.0, 0.0], [9.0, 3.0], [10.0, -3.0], [20.0, 0.0], [15.0, 8.0]


def buoy(state, xy):
    b = SafePassageBuoy(state=state)
    b.position.latitude, b.position.longitude = xy  # the stubbed fromLL maps lat, lon -> x, y
    return b


class Goal:
    def __init__(self, buoys, pause=False):
        self.is_cancel_requested = False
        self.request = SimpleNamespace(buoys=buoys, pause_on_incident=pause)
        self.feedback = []

    def publish_feedback(self, msg):
        self.feedback.append(msg)


class Harness:
    """Stubs the node's I/O. results: per-goal bools (default True), or callables run when that goal is sent."""

    def __init__(self, node, buoys, results=(), boat=(-10.0, 0.0), pause=False):
        self.node, self.goal = node, Goal(buoys, pause)
        self.results, self.boat, self.targets = list(results), np.array(boat), []
        node._boat_xy = lambda: self.boat
        node._snapshot = geo.CostGrid.empty
        node._from_ll = lambda lat, lon: np.array([lat, lon])
        node._run_action = self._run_action
        node.preempt_grace = 0.2
        node.resume_settle = 0.05

    def _run_action(self, gh, client, name, goal, timeout, ok):
        xy = np.array([goal.goal_pose.pose.position.x, goal.goal_pose.pose.position.y])
        self.targets.append(xy)
        result = self.results.pop(0) if self.results else True
        if callable(result):
            result = result()
        self.node._check(gh)
        if result:
            self.boat = xy
        return result

    def run(self):
        return self.node._run(self.goal)

    def phases(self):
        return [f.phase for f in self.goal.feedback]


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


def swept(points, centre):
    """Signed swept angle (deg) of a run of points about centre: negative is clockwise."""
    angles = np.unwrap([math.atan2(p[1] - centre[1], p[0] - centre[0]) for p in points])
    return math.degrees(angles[-1] - angles[0])


def test_order_directions_and_point_counts(node):
    h = Harness(node, [buoy("steady_blue", EXIT), buoy("flashing_green", GREEN), buoy("off", BLACK),
                       buoy("flashing_red", RED), buoy("flashing_blue", ENTRY)])
    ok, circled, failed, message = h.run()
    assert ok and circled == 4 and failed == 0, message
    per = node.circle_points + node.circle_overshoot_points + 1  # start point + circle
    assert len(h.targets) == 4 * per
    # entry, red (x 9), green (x 10), exit; black never circled
    centres = [ENTRY, RED, GREEN, EXIT]
    order = message.split("; ")
    assert [o.split(" ")[0] for o in order] == ["4:flashing_blue", "3:flashing_red", "1:flashing_green",
                                                "0:steady_blue"]
    for n, centre in enumerate(centres):
        run = h.targets[n * per:(n + 1) * per]
        np.testing.assert_allclose(np.linalg.norm(np.array(run) - centre, axis=1), run and
                                   np.linalg.norm(run[0] - np.array(centre)), atol=1e-6)
        sigma = 1 if n < 2 else -1
        assert swept(run, centre) == pytest.approx(-sigma * 405.0, abs=1e-6)
    assert h.phases().count("TRANSIT") == 4


def test_radius_shrinks_for_a_close_neighbour(node):
    h = Harness(node, [buoy("flashing_red", [0, 0]), buoy("off", [3.2, 0])])
    h.run()
    r = np.linalg.norm(h.targets[0])
    assert r == pytest.approx(3.2 - node.neighbour_clearance)


def test_failed_point_is_retried_then_skipped(node):
    # start ok, point 1 fails twice (retry included) -> skipped, rest ok
    h = Harness(node, [buoy("flashing_red", [0, 0])], results=[True, False, False])
    ok, circled, failed, message = h.run()
    assert ok and circled == 1, message
    np.testing.assert_allclose(h.targets[1], h.targets[2])  # the retry re-sends the same point on a still map
    assert len(h.targets) == 1 + 2 + (node.circle_points + node.circle_overshoot_points - 1)


def test_two_skips_in_a_row_fail_the_attempt_and_the_second_attempt_starts_over(node):
    h = Harness(node, [buoy("flashing_green", [0, 0])], results=[True, False, False, False, False])
    ok, circled, failed, message = h.run()
    assert ok and "2 attempt(s)" in message, message
    assert h.phases().count("TRANSIT") == 2


def test_start_point_unreachable_fails_the_attempt(node):
    node.buoy_attempts = 1
    h = Harness(node, [buoy("flashing_blue", [0, 0])], results=[False, False])
    ok, circled, failed, message = h.run()
    assert not ok and failed == 1 and "failed" in message
    assert len(h.targets) == 2


def test_failed_red_requeued_once_before_the_exit(node):
    node.buoy_attempts = 1
    h = Harness(node, [buoy("flashing_red", RED), buoy("steady_blue", EXIT)], results=[False, False])
    ok, circled, failed, message = h.run()
    assert ok and circled == 2, message  # the requeued attempt succeeded
    np.testing.assert_allclose(np.linalg.norm(h.targets[2] - RED), np.linalg.norm(h.targets[0] - RED))
    per = node.circle_points + node.circle_overshoot_points + 1
    assert len(h.targets) == 2 + per + per
    assert np.linalg.norm(h.targets[-1] - EXIT) < 3.0


def test_incident_mid_circle_holds_then_redoes_the_buoy(node):
    def incident():
        node._incident = True
        threading.Timer(0.2, lambda: setattr(node, "_incident", False)).start()
        return True

    h = Harness(node, [buoy("flashing_red", [0, 0])], results=[True, True, incident], pause=True)
    ok, circled, failed, message = h.run()
    assert ok and "1 attempt(s)" in message, message  # the paused attempt does not count
    assert "PAUSED" in h.phases()
    after = h.phases()[h.phases().index("PAUSED") + 1]
    assert after == "TRANSIT"
    per = node.circle_points + node.circle_overshoot_points + 1
    assert len(h.targets) == 3 + per


def test_preempted_goal_waits_for_the_late_incident_flag(node):
    def preempted():
        threading.Timer(0.05, lambda: setattr(node, "_incident", True)).start()
        threading.Timer(0.4, lambda: setattr(node, "_incident", False)).start()
        return False

    h = Harness(node, [buoy("flashing_red", [0, 0])], results=[True, preempted], pause=True)
    ok, circled, failed, message = h.run()
    assert ok and "1 attempt(s)" in message, message
    assert "PAUSED" in h.phases()


def test_incident_ignored_without_pause_on_incident(node):
    node._incident = True
    h = Harness(node, [buoy("flashing_red", [0, 0])])
    assert h.run()[0]
    assert "PAUSED" not in h.phases()
    node._incident = False


def test_cancel_stops_everything(node):
    h = Harness(node, [buoy("flashing_red", RED), buoy("steady_blue", EXIT)])

    def cancel():
        h.goal.is_cancel_requested = True
        return True

    h.results = [True, cancel]
    with pytest.raises(spn.Canceled):
        h.run()
    assert len(h.targets) == 2


def test_cancel_while_paused(node):
    node._incident = True
    h = Harness(node, [buoy("flashing_red", RED)], pause=True)
    threading.Timer(0.2, lambda: setattr(h.goal, "is_cancel_requested", True)).start()
    with pytest.raises(spn.Canceled):
        h.run()
    assert h.targets == []
    node._incident = False


def test_cloud_feeds_landmarks_and_moves_the_circle_onto_them(node):
    """A buoy 1.2 m off the drone's position, seen by the lidar: the circle is centred on what the lidar sees."""
    from sensor_msgs.msg import PointCloud2, PointField
    from sensor_msgs_py.point_cloud2 import create_cloud
    from frames.utils import buoy_field as bf

    truth = np.array([11.2, 0.0])
    h = Harness(node, [buoy("flashing_red", [10.0, 0.0])], boat=(0.0, 0.0))
    node._field = bf.BuoyField([[10.0, 0.0]], node.field_params)
    node._buoys = [spn.Buoy(0, "flashing_red", np.array([10.0, 0.0]), 1)]
    node.base_link_frame_id = "asv5/base_link"
    # Boat at the origin, base_link = map: the transform is identity
    identity = SimpleNamespace(transform=SimpleNamespace(
        translation=SimpleNamespace(x=0.0, y=0.0, z=0.0), rotation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)))
    node.tf_buffer.lookup_transform = lambda *a, **k: identity
    face = np.stack([np.full(8, truth[0] - 0.2), np.linspace(-0.2, 0.2, 8), np.full(8, 0.2)], 1)
    hull = np.array([[0.3, 0.1, 0.2], [-0.4, 0.3, 0.2], [0.5, -0.2, 0.3]])  # own boat: masked
    fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1) for i, n in enumerate("xyz")]
    for _ in range(4):
        from std_msgs.msg import Header
        cloud = create_cloud(Header(frame_id="asv5/base_link"), fields, np.vstack([face, hull]).tolist())
        node._last_cloud = 0.0
        node._on_cloud(cloud)
    assert len(node._field.usable()) == 1
    estimate = node._estimate(node._buoys[0])
    assert estimate.source == bf.LANDMARK
    assert np.linalg.norm(estimate.xy - [truth[0] - 0.2, 0.0]) < 0.05  # the visible face
    node._publish_markers()  # must not raise
    node._field, node._buoys = None, []


def test_cloud_falls_back_to_the_latest_transform_when_tf_lags(node):
    import tf2_ros
    from sensor_msgs.msg import PointField
    from sensor_msgs_py.point_cloud2 import create_cloud
    from std_msgs.msg import Header
    from frames.utils import buoy_field as bf

    node._field = bf.BuoyField([[10.0, 0.0]], node.field_params)
    node.base_link_frame_id = "asv5/base_link"
    identity = SimpleNamespace(transform=SimpleNamespace(
        translation=SimpleNamespace(x=0.0, y=0.0, z=0.0), rotation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)))
    asked = []

    def lookup(target, source, when, timeout=None):
        asked.append(when.nanoseconds)
        if when.nanoseconds:  # at the cloud stamp: TF has not caught up
            raise tf2_ros.ExtrapolationException("would require extrapolation into the future")
        return identity
    node.tf_buffer.lookup_transform = lookup
    face = np.stack([np.full(8, 9.8), np.linspace(-0.2, 0.2, 8), np.full(8, 0.2)], 1)
    fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1) for i, n in enumerate("xyz")]
    header = Header(frame_id="asv5/base_link")
    header.stamp.sec = 100
    for _ in range(3):
        node._last_cloud = 0.0
        node._on_cloud(create_cloud(header, fields, face.tolist()))
    assert asked[:2] == [100 * 10**9, 0]  # stamp first, then latest
    assert len(node._field.usable()) == 1
    node._field = None
