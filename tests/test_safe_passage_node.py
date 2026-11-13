"""safe_passage_node with stubbed TrackPath, TF, fromLL and costmap: alignment, order, retries, Task 4 Core."""
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
from bb_robotx_msgs.msg import AssistanceRequest, ReadinessConfirm, SafePassageBuoy  # noqa: E402

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

from frames.utils import buoy_align as ba  # noqa: E402
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
    def __init__(self, buoys, respond=False):
        self.is_cancel_requested = False
        self.request = SimpleNamespace(buoys=buoys, respond_to_assistance=respond)
        self.feedback = []

    def publish_feedback(self, msg):
        self.feedback.append(msg)


class Harness:
    """Stubs the node's I/O. results: per-goal bools (default True), or callables run when that goal is sent."""

    def __init__(self, node, buoys, results=(), boat=(-10.0, 0.0), respond=False):
        self.node, self.goal = node, Goal(buoys, respond)
        self.acks, self.tasks, self.reports, self.report_seq = [], [], [], 412
        node._send_incident_ack = self.acks.append
        node._send_task = lambda task, goal_handle=None: self.tasks.append(task)
        self.report_unsure = False
        node._send_readiness = lambda gh, seq: (self.reports.append(seq), (self.report_seq, self.report_unsure))[1]
        self.results, self.boat, self.targets = list(results), np.array(boat), []
        node._boat_xy = lambda: self.boat
        node._snapshot = geo.CostGrid.empty
        node._from_ll = lambda lat, lon: np.array([lat, lon])
        node._run_action = self._run_action
        node.align_timeout = 0.0  # no costmap in most tests: go straight on with the drone's positions
        node.readiness_confirm_timeout = 5.0

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


STATION = [30.0, 12.0]


def assistance(seq=30, xy=STATION, vehicle_type="usv"):
    msg = AssistanceRequest(seq=seq, vehicle_type=vehicle_type)
    msg.position.latitude, msg.position.longitude = xy  # the stubbed fromLL maps lat, lon -> x, y
    return msg


def confirm(report_seq=412, vehicle_id="USV1", seq=99):
    return ReadinessConfirm(seq=seq, report_seq=report_seq, vehicle_id=vehicle_id)


def later(delay, fn, *args):
    threading.Timer(delay, fn, args).start()


def request_now(node, *args, **kwargs):
    """A result callable: the request arrives while this TrackPath goal is being driven."""
    def arrive():
        node._on_assistance(assistance(*args, **kwargs))
        return True
    return arrive


def test_assistance_request_mid_circle_runs_the_core_chain_then_redoes_the_buoy(node):
    h = Harness(node, [buoy("flashing_red", [0, 0])], results=[True, True, request_now(node)], respond=True)
    later(0.3, node._on_confirm, confirm())
    ok, circled, failed, message = h.run()
    assert ok and "1 attempt(s)" in message, message      # the interrupted attempt does not count
    assert "assistance request 30 confirmed" in message
    assert h.acks == [30] and h.reports == [30]
    assert h.tasks == ["dynamic_incident", "safe_passage"]
    phases = h.phases()
    assert phases.index("INCIDENT_TRANSIT") < phases.index("INCIDENT_ON_STATION")
    assert phases[phases.index("INCIDENT_ON_STATION") + 1] == "TRANSIT"   # the buoy starts over
    per = node.circle_points + node.circle_overshoot_points + 1
    np.testing.assert_allclose(h.targets[3], STATION)
    assert len(h.targets) == 3 + 1 + per
    assert node._pending is None and node._incident is None
    assert [x.state for x in node._incidents] == ["confirmed"] and node._incidents[0].report_seq == 412


def test_request_for_another_vehicle_type_is_ignored(node):
    h = Harness(node, [buoy("flashing_red", [0, 0])], results=[True, request_now(node, vehicle_type="uuv")],
                respond=True)
    ok, circled, failed, message = h.run()
    assert ok and h.acks == [] and h.tasks == [] and "assistance" not in message
    assert "INCIDENT_TRANSIT" not in h.phases()


def test_request_is_ignored_when_the_goal_does_not_ask_for_it(node):
    h = Harness(node, [buoy("flashing_red", [0, 0])], results=[True, request_now(node)])
    assert h.run()[0] and h.acks == [] and "INCIDENT_TRANSIT" not in h.phases()


def test_request_with_no_goal_running_is_dropped(node):
    node._on_assistance(assistance(seq=30))               # nothing running: dropped, nothing kept
    assert node._pending is None and node._incidents == [] and node._seen == set()
    h = Harness(node, [buoy("flashing_red", [0, 0])], respond=True)
    assert h.run()[0] and h.acks == [] and "assistance" not in h.run()[3]


def test_seq_can_come_again_in_another_goal(node):
    """RoboCommand may number from 1 again in the next run: the node must not turn that request down."""
    for _ in range(2):
        h = Harness(node, [buoy("flashing_red", [0, 0])], results=[True, request_now(node, seq=1)], respond=True)
        later(0.2, node._on_confirm, confirm())
        ok, circled, failed, message = h.run()
        assert ok and h.acks == [1] and "assistance request 1 confirmed" in message
        node._respond = False                             # as _execute leaves it when a goal ends


def test_incident_transit_gets_its_own_goal_timeout(node):
    timeouts = []
    h = Harness(node, [buoy("flashing_red", [0, 0])], results=[True, request_now(node)], respond=True)
    inner = node._run_action
    node._run_action = lambda gh, client, name, goal, timeout, ok: (timeouts.append(timeout),
                                                                    inner(gh, client, name, goal, timeout, ok))[1]
    later(0.2, node._on_confirm, confirm())
    assert h.run()[0]
    assert timeouts[1] == node.goal_timeout and timeouts[2] == node.incident_goal_timeout == 300.0


def test_older_task_switch_gives_way_to_a_newer_one(node):
    """SetTask dynamic_incident still being retried must not land after the switch back to safe_passage."""
    sent, release = [], threading.Event()

    def send_blocking(client, goal, stale=lambda: False):
        if goal.task == "dynamic_incident":
            release.wait(2.0)                             # the OCS is slow to answer this one
            return "no heartbeat"                         # ... and it fails: a retry would follow
        sent.append(goal.task)
        return None

    node._send_blocking = send_blocking
    node._send_task("dynamic_incident")                   # background, stuck
    node._send_task("safe_passage", Goal([]))             # the switch back, waited for
    release.set()
    time.sleep(1.5)                                       # the retry pause of the first one
    assert sent == ["safe_passage"]


def test_waited_task_switch_tries_once_then_retries_behind(node):
    """The switch back is waited for, but a dead OCS link must not hold the boat for every retry."""
    calls = []

    def send_blocking(client, goal, stale=lambda: False):
        calls.append(time.monotonic())
        return "server unavailable" if len(calls) < 3 else None

    node._send_blocking = send_blocking
    started = time.monotonic()
    node._send_task("safe_passage", Goal([]))
    assert len(calls) == 1 and time.monotonic() - started < 0.5   # returned after the first try
    deadline = time.monotonic() + 5.0
    while len(calls) < 3 and time.monotonic() < deadline:
        time.sleep(0.05)
    assert len(calls) == 3                                        # and it still got through


def test_repeated_request_is_answered_once(node):
    def repeat():
        for _ in range(3):                                # delivered more than once
            node._on_assistance(assistance())
        return True

    h = Harness(node, [buoy("flashing_red", [0, 0])], results=[True, repeat], respond=True)
    later(0.2, node._on_assistance, assistance())
    later(0.3, node._on_confirm, confirm())
    ok, circled, failed, message = h.run()
    assert ok and h.acks == [30] and h.reports == [30] and len(node._incidents) == 1


def test_only_our_confirm_for_our_report_releases_the_hold(node):
    h = Harness(node, [buoy("flashing_red", [0, 0])], results=[True, request_now(node)], respond=True)
    node.readiness_confirm_timeout = 0.6
    later(0.1, node._on_confirm, confirm(vehicle_id="USV2"))        # another vehicle's
    later(0.15, node._on_confirm, confirm(report_seq=411))          # ours, another report
    started = time.monotonic()
    ok, circled, failed, message = h.run()
    assert time.monotonic() - started >= 0.6
    assert ok and circled == 1 and "assistance request 30 confirm_timeout" in message
    assert h.tasks == ["dynamic_incident", "safe_passage"]          # resumes on its own after the timeout


def test_confirm_that_beats_the_report_seq_still_counts(node):
    def arrive():
        node._on_assistance(assistance())
        node._on_confirm(confirm())                       # before the ReadinessReport action has returned 412
        return True

    h = Harness(node, [buoy("flashing_red", [0, 0])], results=[True, arrive], respond=True)
    started = time.monotonic()
    ok, circled, failed, message = h.run()
    assert ok and "assistance request 30 confirmed" in message and time.monotonic() - started < 2.0


def test_newer_request_replaces_the_one_in_hand(node):
    other = [-20.0, -9.0]
    h = Harness(node, [buoy("flashing_red", [0, 0])], respond=True,
                results=[True, request_now(node, seq=30), request_now(node, seq=31, xy=other)])
    later(0.3, node._on_confirm, confirm())
    ok, circled, failed, message = h.run()
    assert ok and "assistance request 30 superseded" in message and "assistance request 31 confirmed" in message
    assert h.acks == [30, 31] and h.reports == [31]
    np.testing.assert_allclose(h.targets[2], STATION)
    np.testing.assert_allclose(h.targets[3], other)
    assert h.tasks[-1] == "safe_passage" and h.tasks.count("safe_passage") == 1


def test_unreachable_station_sends_no_readiness_report_and_resumes(node):
    h = Harness(node, [buoy("flashing_red", [0, 0])], respond=True,
                results=[True, request_now(node), False, False, False])
    ok, circled, failed, message = h.run()
    assert ok and circled == 1 and "assistance request 30 unreachable" in message
    assert h.acks == [30] and h.reports == [] and h.tasks == ["dynamic_incident", "safe_passage"]
    assert "INCIDENT_ON_STATION" not in h.phases()


def test_readiness_report_that_cannot_be_sent_resumes_without_waiting(node):
    h = Harness(node, [buoy("flashing_red", [0, 0])], results=[True, request_now(node)], respond=True)
    h.report_seq = None
    started = time.monotonic()
    ok, circled, failed, message = h.run()
    assert ok and "assistance request 30 report_failed" in message and time.monotonic() - started < 2.0


def test_request_during_align_is_answered_then_alignment_goes_on(node):
    h = Harness(node, [buoy("flashing_red", [0, 0])], respond=True)
    node.align_timeout = 0.5
    later(0.15, node._on_assistance, assistance())
    later(0.4, node._on_confirm, confirm())
    ok, circled, failed, message = h.run()
    assert ok and circled == 1 and "assistance request 30 confirmed" in message
    phases = h.phases()
    assert phases[0] == "ALIGN" and "ALIGN" in phases[phases.index("INCIDENT_ON_STATION"):]
    np.testing.assert_allclose(h.targets[0], STATION)


def test_cancel_stops_everything(node):
    h = Harness(node, [buoy("flashing_red", RED), buoy("steady_blue", EXIT)])

    def cancel():
        h.goal.is_cancel_requested = True
        return True

    h.results = [True, cancel]
    with pytest.raises(spn.Canceled):
        h.run()
    assert len(h.targets) == 2


def test_cancel_while_holding_on_station(node):
    h = Harness(node, [buoy("flashing_red", RED)], results=[True, request_now(node)], respond=True)
    later(0.3, setattr, h.goal, "is_cancel_requested", True)
    with pytest.raises(spn.Canceled):
        h.run()
    assert h.reports == [30] and h.tasks == ["dynamic_incident"]
    assert node._incidents[0].state == "on_station"   # _execute closes it as canceled when the goal ends


LAYOUT = np.array([[0.0, 0.0], [9.0, 3.0], [10.0, -3.0], [20.0, 0.0], [15.0, 8.0], [5.0, -8.0]])
STATES = ["flashing_blue", "flashing_red", "flashing_green", "steady_blue", "off", "off"]
SHIFT = np.array([2.0, 6.0])


def costmap(points, size=80.0, origin=(-30.0, -40.0), res=0.1):
    n = int(size / res)
    data = np.zeros((n, n), dtype=np.int16)
    for x, y in points:
        c, r = int((x - origin[0]) / res), int((y - origin[1]) / res)
        data[r - 2:r + 2, c - 2:c + 2] = 100
    return geo.CostGrid(data, res, origin[0], origin[1])


def feed(node, grid, times=6, period=0.05):
    """Costmaps arriving while the goal runs: a new message, so a new grid, each time."""
    def loop():
        for _ in range(times):
            time.sleep(period)
            node._align_step(geo.CostGrid(grid.data, grid.resolution, grid.origin_x, grid.origin_y))
    threading.Thread(target=loop, daemon=True).start()


def test_layout_is_aligned_to_the_costmap_before_any_driving(node):
    """Drone layout 6 m off: the node holds in ALIGN, locks the shift, then circles the real buoys."""
    grid = costmap(LAYOUT + SHIFT)
    h = Harness(node, [buoy(s, xy) for s, xy in zip(STATES, LAYOUT)], boat=(-10.0, 5.0))
    node._snapshot = lambda: grid
    node.align_timeout = 5.0
    feed(node, grid)
    ok, circled, failed, message = h.run()
    assert ok and circled == 4, message
    assert h.phases()[0] == "ALIGN" and "ALIGN" not in h.phases()[h.phases().index("TRANSIT"):]
    per = node.circle_points + node.circle_overshoot_points + 1
    for n, i in enumerate([0, 1, 2, 3]):  # entry, red, green, exit
        run = np.array(h.targets[n * per:(n + 1) * per])
        radii = np.linalg.norm(run - (LAYOUT[i] + SHIFT), axis=1)
        assert radii.max() - radii.min() < 0.15, (i, radii)  # circled around the real buoy, not the drone's
    assert "BLOB" in message
    assert h.goal.feedback[-1].shift_inliers == len(LAYOUT)
    node._aligner, node._buoys = ba.Aligner(LAYOUT, node.align_params), []
    node._publish_markers()  # must not raise
    node._aligner = None


def test_no_lock_goes_on_with_the_drone_positions_after_the_timeout(node):
    h = Harness(node, [buoy("flashing_red", [0.0, 0.0])])
    node.align_timeout = 0.3
    started = time.monotonic()
    ok, circled, failed, message = h.run()
    assert ok and circled == 1 and "PREDICTED" in message
    assert 0.3 <= time.monotonic() - started < 2.0
    radii = np.linalg.norm(np.array(h.targets), axis=1)
    assert radii.max() - radii.min() < 1e-6


def test_cancel_while_aligning(node):
    h = Harness(node, [buoy("flashing_red", [0.0, 0.0])])
    node.align_timeout = 5.0
    threading.Timer(0.2, lambda: setattr(h.goal, "is_cancel_requested", True)).start()
    with pytest.raises(spn.Canceled):
        h.run()
    assert h.targets == []


def test_circle_follows_a_blob_near_the_drone_position_without_a_lock(node):
    """One buoy cannot lock a shift, but a blob within the snap radius still centres the circle."""
    grid = costmap([[10.7, 0.0]])
    h = Harness(node, [buoy("flashing_red", [10.0, 0.0])], boat=(0.0, 0.0))
    node._snapshot = lambda: grid
    feed(node, grid, times=2, period=0.0)
    time.sleep(0.05)
    ok, circled, failed, message = h.run()
    assert ok and "BLOB" in message, message
    radii = np.linalg.norm(np.array(h.targets) - [10.7, 0.0], axis=1)
    assert radii.max() - radii.min() < 0.15


def test_reports_go_out_through_real_actions(ros):
    """The Task 4 senders against real action servers: IncidentAck, SetTask, and the ReadinessReport's seq."""
    from bb_robotx_msgs.action import SendIncidentAck, SendReadinessReport, SetTask
    from rclpy.action import ActionServer
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node

    got = []

    def serve(kind, result):
        def execute(goal_handle):
            request = goal_handle.request
            got.append((kind, request.task if kind == "task" else request.report.command_seq))
            goal_handle.succeed()
            return result()
        return execute

    ocs = Node("fake_ocs_bridge")
    servers = [
        ActionServer(ocs, SendIncidentAck, "/asv5/report/incident_ack",
                     serve("ack", lambda: SendIncidentAck.Result(seq=411))),
        ActionServer(ocs, SendReadinessReport, "/asv5/report/readiness",
                     serve("ready", lambda: SendReadinessReport.Result(seq=412))),
        ActionServer(ocs, SetTask, "/asv5/set_task", serve("task", lambda: SetTask.Result(error="no heartbeat"))),
    ]
    node = spn.SafePassageNode()
    node.report_retries, node.report_timeout = 0, 5.0
    executor = MultiThreadedExecutor()
    executor.add_node(ocs)
    executor.add_node(node)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()
    try:
        ack = SendIncidentAck.Goal()
        ack.report.command_seq = 30
        assert node._send_blocking(node.incident_ack, ack) is None
        assert node._send_blocking(node.set_task, SetTask.Goal(task="dynamic_incident")) == "no heartbeat"
        assert node._send_readiness(Goal([]), 30) == (412, False)
        assert got == [("ack", 30), ("task", "dynamic_incident"), ("ready", 30)]
        node._send_incident_ack(31)                       # the background path, as the chain uses it
        deadline = time.monotonic() + 5.0
        while ("ack", 31) not in got and time.monotonic() < deadline:
            time.sleep(0.05)
        assert ("ack", 31) in got
    finally:
        executor.shutdown(timeout_sec=2.0)
        for server in servers:
            server.destroy()
        node.destroy_node()
        ocs.destroy_node()


def test_confirm_counts_when_an_earlier_report_try_may_have_gone_out(node):
    """A failed try can reach RoboCommand under a seq we never learn; its confirm must still release the hold."""
    h = Harness(node, [buoy("flashing_red", [0, 0])], results=[True, request_now(node)], respond=True)
    h.report_seq, h.report_unsure = 413, True
    node._on_confirm(confirm(report_seq=400))                       # before the goal: never counts
    later(0.3, node._on_confirm, confirm(report_seq=412))           # the try we think failed
    started = time.monotonic()
    ok, circled, failed, message = h.run()
    assert ok and "assistance request 30 confirmed" in message and time.monotonic() - started < 3.0


def test_confirm_from_before_the_station_does_not_count_even_when_unsure(node):
    def arrive():
        node._on_assistance(assistance())
        node._on_confirm(confirm(report_seq=390))                   # in this goal, but before we got on station
        return True

    h = Harness(node, [buoy("flashing_red", [0, 0])], results=[True, arrive], respond=True)
    h.report_seq, h.report_unsure = 413, True
    node.readiness_confirm_timeout = 0.5
    ok, circled, failed, message = h.run()
    assert ok and "assistance request 30 confirm_timeout" in message


def test_request_during_the_switch_back_is_answered_before_going_on(node):
    other = [-20.0, -9.0]
    h = Harness(node, [buoy("flashing_red", [0, 0])], results=[True, request_now(node, seq=30)], respond=True)
    sent = []

    def send_task(task, goal_handle=None):
        sent.append(task)
        if sent == ["dynamic_incident", "safe_passage"]:            # a second request lands during the switch back
            node._on_assistance(assistance(seq=31, xy=other))
    node._send_task = send_task
    later(0.2, node._on_confirm, confirm())
    ok, circled, failed, message = h.run()
    assert ok and "assistance request 30 confirmed" in message and "assistance request 31 confirmed" in message
    assert sent == ["dynamic_incident", "safe_passage", "dynamic_incident", "safe_passage"]
    np.testing.assert_allclose(h.targets[2], STATION)
    np.testing.assert_allclose(h.targets[3], other)
    per = node.circle_points + node.circle_overshoot_points + 1
    assert len(h.targets) == 2 + 2 + per                            # the buoy started over once, after both


def test_blob_missing_from_one_costmap_does_not_move_the_circle(node):
    truth = np.array([10.7, 0.0])
    grid = costmap([truth])
    h = Harness(node, [buoy("flashing_red", [10.0, 0.0])], boat=(0.0, 0.0))
    node._snapshot = lambda: grid

    def blob_gone():
        node._aligner.buoy_blobs = np.zeros((0, 2))                 # the costmap cleared it for a moment
        return True

    h.results = [True, True, blob_gone]
    ok, circled, failed, message = h.run()
    assert ok and circled == 1
    radii = np.linalg.norm(np.array(h.targets) - truth, axis=1)
    assert radii.max() - radii.min() < 0.15, radii                  # every point still around the blob


def test_layout_locking_mid_circle_starts_the_buoy_over(node):
    """No lock by the timeout, so the circle starts on the drone's position; the lock then moves it metres."""
    shift = np.array([4.0, 0.0])
    h = Harness(node, [buoy("flashing_red", [0.0, 0.0])])

    def lock_now():
        node._aligner.t, node._aligner.locked = shift.copy(), True
        return True

    h.results = [True, True, lock_now]
    ok, circled, failed, message = h.run()
    assert ok and circled == 1 and "1 attempt(s)" in message         # the restart is not a failed attempt
    per = node.circle_points + node.circle_overshoot_points + 1
    assert h.phases().count("TRANSIT") == 2 and len(h.targets) == 3 + per
    radii = np.linalg.norm(np.array(h.targets[3:]) - shift, axis=1)
    assert radii.max() - radii.min() < 1e-6                           # a whole circle around where it really is


def test_restarts_are_capped_then_points_are_only_re_placed(node):
    node.buoy_restarts = 1
    h = Harness(node, [buoy("flashing_red", [0.0, 0.0])])
    jumps = iter([np.array([4.0, 0.0]), np.array([8.0, 0.0])])

    def jump():
        node._aligner.t = next(jumps)
        return True

    h.results = [True, jump, True, jump]
    ok, circled, failed, message = h.run()
    assert ok and circled == 1 and h.phases().count("TRANSIT") == 2   # one restart, the second jump re-places


def test_same_costmap_counts_once_towards_the_lock(node):
    grid = costmap(LAYOUT + SHIFT)
    node._aligner = ba.Aligner(LAYOUT, node.align_params)
    for _ in range(5):
        node._align_step(grid)
    assert not node._aligner.locked                                   # one costmap, however often it is offered
    for _ in range(2):
        node._align_step(costmap(LAYOUT + SHIFT))
    assert node._aligner.locked
    node._aligner = None
