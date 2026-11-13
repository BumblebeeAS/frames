"""safe_passage_node with stubbed TrackPath, TF, fromLL and costmap: alignment, the route, retries, Task 4 Core."""
import math
import sys
import threading
import time
import types
from dataclasses import replace
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
from frames.utils import safe_route as sr  # noqa: E402

spec = importlib.util.spec_from_file_location("safe_passage_node", ROOT / "scripts/safe_passage_node.py")
spn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(spn)

# A channel running north: reds to the east (starboard going north), greens to the west, a black off to the side
ENTRY, EXIT = [0.0, 0.0], [0.0, 20.0]
REDS, GREENS, BLACK = [[3.0, 7.0], [3.0, 13.0]], [[-3.0, 7.0], [-3.0, 13.0]], [9.0, 10.0]
STATION = [30.0, 12.0]


def buoy(state, xy):
    b = SafePassageBuoy(state=state)
    b.position.latitude, b.position.longitude = xy  # the stubbed fromLL maps lat, lon -> x, y
    return b


def channel(entry=True, exit_=True):
    out = [buoy("flashing_red", xy) for xy in REDS] + [buoy("flashing_green", xy) for xy in GREENS]
    out.append(buoy("off", BLACK))
    if entry:
        out.append(buoy("flashing_blue", ENTRY))
    if exit_:
        out.append(buoy("steady_blue", EXIT))
    return out


def positions(buoys):
    return [[b.position.latitude, b.position.longitude] for b in buoys]


def costmap(points, size=80.0, origin=(-30.0, -40.0), res=0.1):
    n = int(size / res)
    data = np.zeros((n, n), dtype=np.int16)
    for x, y in points:
        c, r = int((x - origin[0]) / res), int((y - origin[1]) / res)
        data[r - 2:r + 2, c - 2:c + 2] = 100
    return geo.CostGrid(data, res, origin[0], origin[1])


PLANS = {}     # the planner is deterministic: one search per distinct problem for the whole module
CALLS = []     # every call a node made: (start, need_red, need_green)
GOALS = []     # the goals of the test in hand
_plan_route = sr.plan_route


def plan_route(buoys, params=None, obstacles=None, start=None, need_red=True, need_green=True, **kwargs):
    CALLS.append((None if start is None else tuple(np.round(start, 2)), need_red, need_green))
    if len(CALLS) > 12 and GOALS:      # a test that would plan for ever ends as a canceled goal instead
        GOALS[-1].is_cancel_requested = True
    key = (tuple((round(b.x, 2), round(b.y, 2), b.state, b.confirmed) for b in buoys), CALLS[-1],
           0 if obstacles is None else len(obstacles))
    if key not in PLANS:
        PLANS[key] = _plan_route(buoys, params, obstacles, start=start, need_red=need_red, need_green=need_green,
                                 **kwargs)
    return PLANS[key]


@pytest.fixture(autouse=True)
def cached_planner(monkeypatch):
    monkeypatch.setattr(sr, "plan_route", plan_route)
    del CALLS[:]
    del GOALS[:]


class Goal:
    def __init__(self, buoys, respond=False):
        self.is_cancel_requested = False
        self.request = SimpleNamespace(buoys=buoys, respond_to_assistance=respond)
        self.feedback = []

    def publish_feedback(self, msg):
        self.feedback.append(msg)


class Harness:
    """Stubs the node's I/O. results: per-goal bools (default True), or callables run when that goal is sent.

    A goal that succeeds is driven the way the boat drives it (turn on the spot, straight there, turn to the goal's
    heading), every pose going through the node's pass tracker. The costmap has a blob on every buoy unless given.
    """

    def __init__(self, node, buoys=None, results=(), boat=(0.0, -10.0), respond=False, grid=None):
        buoys = channel() if buoys is None else buoys
        self.node, self.goal = node, Goal(buoys, respond)
        GOALS.append(self.goal)
        self.acks, self.tasks, self.reports, self.report_seq = [], [], [], 412
        node._send_incident_ack = self.acks.append
        node._send_task = lambda task, goal_handle=None: self.tasks.append(task)
        self.report_unsure = False
        node._send_readiness = lambda gh, seq: (self.reports.append(seq), (self.report_seq, self.report_unsure))[1]
        self.results, self.pose, self.targets, self.timeouts = list(results), (boat[0], boat[1], math.pi / 2), [], []
        node._boat_pose = lambda quiet=False: self.pose
        self.grid = costmap(positions(buoys)) if grid is None else grid
        node._snapshot = lambda: self.grid
        node._from_ll = lambda lat, lon: np.array([lat, lon])
        node._run_action = self._run_action
        node.route_params = replace(node.route_params, mc_trials=0, choose="strictest")  # one search per plan
        node.align_timeout = 0.0  # one buoy-sized blob per buoy and no shift: go straight on
        node.plan_retry_period = 0.05
        node.readiness_confirm_timeout = 5.0

    def _run_action(self, gh, client, name, goal, timeout, ok):
        pose = goal.goal_pose.pose
        xy = np.array([pose.position.x, pose.position.y])
        self.targets.append(xy)
        self.timeouts.append(timeout)
        result = self.results.pop(0) if self.results else True
        if callable(result):
            result = result()
        self.node._check(gh)
        if result:
            self._sail(xy, 2.0 * math.atan2(pose.orientation.z, pose.orientation.w))
        return result

    def _sail(self, xy, yaw):
        x, y, heading = self.pose
        travel = math.atan2(xy[1] - y, xy[0] - x) if np.hypot(xy[0] - x, xy[1] - y) > 1e-6 else heading

        def turn(to):
            nonlocal heading
            delta = (to - heading + math.pi) % (2.0 * math.pi) - math.pi
            for k in range(1, int(abs(delta) / math.radians(3.0)) + 2):
                self.pose = (x, y, heading + delta * k / (int(abs(delta) / math.radians(3.0)) + 1))
                self.node._track_passes()
            heading = to

        turn(travel)
        steps = max(1, int(np.hypot(xy[0] - x, xy[1] - y) / 0.1))
        x0, y0 = x, y
        for k in range(1, steps + 1):
            x, y = x0 + (xy[0] - x0) * k / steps, y0 + (xy[1] - y0) * k / steps
            self.pose = (x, y, heading)
            self.node._track_passes()
        turn(yaw)

    @property
    def boat(self):
        return np.array(self.pose[:2])

    def run(self):
        return self.node._run(self.goal)

    def phases(self):
        return [f.phase for f in self.goal.feedback]

    def stages(self):
        """The phases in the order they began, each once per stretch."""
        out = []
        for phase in self.phases():
            if not out or out[-1] != phase:
                out.append(phase)
        return out


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


def count(h, phase):
    """Goals a clean run sends in that phase."""
    return max(f.point_total for f in h.goal.feedback if f.phase == phase)


LOOP = 7       # goals of the entry loop: its start point, then six more round to it again


# The route

def test_run_is_align_plan_entry_transit_exit(node):
    h = Harness(node)
    ok, circled, failed, message = h.run()
    assert ok and circled == 2 and failed == 0, message
    assert h.stages() == ["ALIGN", "PLAN", "ENTRY", "TRANSIT", "EXIT"]
    assert CALLS == [(None, True, True)]                                  # planned once, with the entry loop
    n_entry, n_transit, n_exit = count(h, "ENTRY"), count(h, "TRANSIT"), count(h, "EXIT")
    assert n_entry == LOOP and len(h.targets) == n_entry + n_transit + n_exit
    loop = h.targets[:n_entry]
    np.testing.assert_allclose(loop[0], loop[-1], atol=1e-6)              # it closes on the point it began at
    assert swept(loop, ENTRY) == pytest.approx(-360.0, abs=1e-6)          # clockwise
    ring = h.targets[n_entry + n_transit - 1:]                            # from the join point on
    assert swept(ring, EXIT) == pytest.approx(360.0 + 60.0, abs=1e-6)     # counterclockwise, one goal past the lap
    assert "red to starboard yes, green to port yes" in message and "route ok" in message
    assert "1 plan(s), 0 goal(s) failed" in message and "BLOB" in message
    assert not any("violation" in line for line in message.split("; "))
    assert node._tracker is None


def test_passes_are_followed_on_the_boat_pose_between_the_circles_only(node):
    h = Harness(node)
    seen = []
    track = node._track
    node._track = lambda on, progress=None: (seen.append((on, len(h.targets))), track(on, progress))[1]
    ok, circled, failed, message = h.run()
    n_entry, n_transit = count(h, "ENTRY"), count(h, "TRANSIT")
    assert (True, n_entry) in seen                                         # from the entry loop closing ...
    assert (False, n_entry + n_transit) in seen                            # ... to the exit circle being joined
    assert not any(on for on, at in seen if at != n_entry)
    passes = [line for line in message.split("; ") if line.startswith("pass ")]
    assert any("flashing_red abeam to starboard" in p and "credit" in p for p in passes)
    assert any("flashing_green abeam to port" in p and "credit" in p for p in passes)


def test_buoys_without_a_blob_are_obeyed_but_earn_nothing(node):
    h = Harness(node, grid=geo.CostGrid.empty())
    ok, circled, failed, message = h.run()
    assert ok and circled == 2 and "PREDICTED" in message and "BLOB" not in message
    assert "no confirmed red to pass" in message and "red to starboard no, green to port no" in message
    assert not any("violation" in line for line in message.split("; "))


def test_no_entry_buoy_starts_from_the_boat(node):
    h = Harness(node, channel(entry=False), boat=(0.0, 1.0))
    ok, circled, failed, message = h.run()
    assert ok and circled == 1 and h.stages() == ["ALIGN", "PLAN", "TRANSIT", "EXIT"]
    assert CALLS == [((0.0, 1.0), True, True)]
    assert "red to starboard yes, green to port yes" in message


def test_no_exit_buoy_ends_after_the_passes(node):
    h = Harness(node, channel(exit_=False))
    ok, circled, failed, message = h.run()
    assert ok and circled == 1 and h.stages() == ["ALIGN", "PLAN", "ENTRY", "TRANSIT"]
    assert "route missing_exit" in message and "red to starboard yes, green to port yes" in message


def route_inputs(node, h):
    """Run the goal and return what its (last) plan was made on."""
    seen = {}
    drive = node._drive_route
    node._drive_route = lambda *args: (
        seen.update(buoys=list(node._route_buoys), candidates=node._aligner.candidates.copy()), drive(*args))[1]
    return h.run(), seen


def test_unknown_buoys_are_dropped_and_take_no_part_in_the_alignment(node):
    h = Harness(node, channel() + [buoy("unknown", [1.0, 10.0]), buoy("", [5.0, 5.0])],
                grid=costmap(positions(channel())))
    (ok, circled, failed, message), seen = route_inputs(node, h)
    assert ok and circled == 2 and "unknown" not in message
    np.testing.assert_allclose(seen["candidates"], positions(channel()))  # the seven known ones only
    assert [b.state for b in seen["buoys"]] == [b.state for b in channel()]
    assert goal_rejected_without_known_buoys(node)


def goal_rejected_without_known_buoys(node):
    from rclpy.action import GoalResponse
    request = SimpleNamespace(buoys=[buoy("unknown", [0.0, 0.0]), buoy("off", [1.0, 1.0])])
    return node._on_goal(request) == GoalResponse.REJECT


def test_planner_is_told_about_blobs_the_drone_missed(node):
    extra = [-6.0, 4.0]                                                  # a buoy-sized object the drone did not report
    h = Harness(node, grid=costmap(positions(channel()) + [extra]))
    (ok, circled, failed, message), seen = route_inputs(node, h)
    assert ok
    unknown = [b for b in seen["buoys"] if b.state == sr.UNKNOWN]
    assert len(seen["buoys"]) == len(channel()) + 1 and len(unknown) == 1
    np.testing.assert_allclose([unknown[0].x, unknown[0].y], extra, atol=0.15)
    assert all(b.confirmed for b in seen["buoys"])


def test_obstacle_cells_are_the_structures_not_the_buoys(node):
    wall = [[14.0 + 0.1 * k, 16.0] for k in range(40)]
    h = Harness(node, grid=costmap(positions(channel()) + wall))
    node._aligner = ba.Aligner(np.array(positions(channel())), node.align_params)
    node._solved_grid = None
    node._align_step(h.grid)
    buoys = [spn.Buoy(i, b.state, np.array(xy)) for i, (b, xy) in enumerate(zip(channel(), positions(channel())))]
    inputs, cells = node._planner_inputs(buoys, h.grid)
    assert len(inputs) == len(buoys) and len(cells) > 100
    assert cells[:, 0].min() > 13.5 and abs(cells[:, 1] - 16.0).max() < 0.3   # the wall only
    node._aligner.buoy_blobs = np.zeros((0, 2))                           # the costmap cleared the blobs for a moment
    again, _ = node._planner_inputs(buoys, h.grid)
    assert all(b.confirmed for b in again)                                # every buoy stays where it was seen
    node._aligner = None


# Retries

def test_failed_goal_is_sent_again_up_to_three_times_in_all(node):
    h = Harness(node, results=[True, False, False])
    ok, circled, failed, message = h.run()
    assert ok and failed == 0 and "1 plan(s)" in message
    np.testing.assert_allclose(h.targets[1], h.targets[2])
    np.testing.assert_allclose(h.targets[1], h.targets[3])


def test_goal_that_fails_three_times_in_the_entry_loop_plans_the_whole_run_again(node):
    h = Harness(node, results=[True, True, False, False, False])
    ok, circled, failed, message = h.run()
    assert ok and circled == 2 and failed == 1 and "2 plan(s), 1 goal(s) failed" in message
    assert h.stages() == ["ALIGN", "PLAN", "ENTRY", "PLAN", "ENTRY", "TRANSIT", "EXIT"]
    assert CALLS == [(None, True, True), (None, True, True)]             # the loop had not closed: it is done again
    np.testing.assert_allclose(h.targets[5], h.targets[0])                # ... from its start point


def test_goal_that_fails_in_the_transit_plans_again_from_the_boat_without_the_entry_loop(node):
    h = Harness(node, results=[True] * LOOP + [False] * 3)                # the transit is one goal here
    ok, circled, failed, message = h.run()
    assert ok and circled == 2 and failed == 1, message
    assert h.stages() == ["ALIGN", "PLAN", "ENTRY", "TRANSIT", "PLAN", "TRANSIT", "EXIT"]
    assert CALLS[0] == (None, True, True) and len(CALLS) == 2
    start, need_red, need_green = CALLS[1]
    np.testing.assert_allclose(start, h.targets[LOOP - 1], atol=0.01)     # where the boat stopped
    assert need_red and need_green
    assert "red to starboard yes, green to port yes" in message
    assert not any("violation" in line for line in message.split("; "))


def test_passes_in_hand_are_not_planned_for_again(node):
    h = Harness(node)
    transit = []
    ok, circled, failed, message = h.run()                                # a clean run, to learn the transit
    transit = count(h, "TRANSIT")
    del CALLS[:]
    h = Harness(node, results=[True] * (LOOP + transit) + [False] * 3)    # the first exit goal fails every time
    ok, circled, failed, message = h.run()
    assert ok and circled == 2 and failed == 1
    assert CALLS[1][1:] == (False, False)                                 # both passes were made on the transit


def test_no_route_is_planned_again_until_there_is_one(node):
    h = Harness(node)
    answers = [sr.RoutePlan("no_route", "nothing"), sr.RoutePlan("no_route", "nothing")]
    inner = sr.plan_route
    sr.plan_route = lambda *a, **k: answers.pop(0) if answers else inner(*a, **k)
    try:
        ok, circled, failed, message = h.run()
    finally:
        sr.plan_route = inner
    assert ok and circled == 2 and "3 plan(s)" in message and h.phases().count("PLAN") == 3


def test_planner_fault_is_planned_again(node):
    h = Harness(node)
    faults = [True]
    inner = sr.plan_route

    def faulty(*a, **k):
        if faults:
            faults.pop()
            raise RuntimeError("boom")
        return inner(*a, **k)

    sr.plan_route = faulty
    try:
        ok, circled, failed, message = h.run()
    finally:
        sr.plan_route = inner
    assert ok and "2 plan(s)" in message


def test_goal_timeout_grows_with_the_distance(node):
    h = Harness(node, boat=(0.0, -40.0))
    assert h.run()[0]
    far = float(np.hypot(*(h.targets[0] - [0.0, -40.0])))
    assert h.timeouts[0] == pytest.approx(node.goal_timeout + far / node.goal_speed)
    assert h.timeouts[1] < h.timeouts[0] and h.timeouts[1] > node.goal_timeout


def test_node_and_yaml_agree(node):
    import yaml
    cfg = yaml.safe_load(open(ROOT / "cfg/asv5_safe_passage.yaml"))["safe_passage_node"]["ros__parameters"]
    assert cfg["point_retries"] == node.point_retries == 2
    assert cfg["circle_radius"] == node.route_params.circle_radius and cfg["circle_points"] == 6
    assert cfg["incident_return"] is True and node.incident_return is True
    for gone in ("pass_points", "gate_min_width", "buoy_attempts", "requeue_failed", "estimate_restart_threshold"):
        assert gone not in cfg
    declared = set(node._parameters)
    assert set(cfg) <= declared, set(cfg) - declared


def test_goal_not_yet_accepted_when_the_interrupt_arrives_is_canceled_once_it_is(ros):
    """Else it would go on driving the boat beside the goal sent next."""
    from rclpy.task import Future

    node = spn.SafePassageNode()
    try:
        send, canceled = Future(), []
        client = SimpleNamespace(wait_for_server=lambda timeout_sec: True, send_goal_async=lambda goal: send)
        gh = Goal([])
        gh.is_cancel_requested = True                                     # the interrupt: before the goal is accepted
        with pytest.raises(spn.Canceled):
            node._run_action(gh, client, "track_path", object(), 5.0, lambda r: True)
        assert not canceled
        send.set_result(SimpleNamespace(accepted=True, cancel_goal_async=lambda: canceled.append(True)))
        for callback in list(send._callbacks):                            # no executor here: run them as it would
            callback(send)
        assert canceled == [True]
    finally:
        node.destroy_node()


def test_pass_tracker_uses_the_margins_of_the_route_in_hand(node):
    h = Harness(node)
    seen = []
    track = node._track
    node._track = lambda on, progress=None: (track(on, progress), seen.append(node._tracker))[0]
    assert h.run()[0]
    tracker = next(t for t in seen if t is not None)
    level = node.route_params.levels[node._route_level]
    assert (tracker.p.shield_margin, tracker.p.credit_margin, tracker.p.near_range) == level[1:]


# Task 4 Core

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


def test_request_in_the_entry_loop_runs_the_core_chain_goes_back_and_does_the_loop_again(node):
    h = Harness(node, results=[True, True, request_now(node)], respond=True)
    later(0.3, node._on_confirm, confirm())
    ok, circled, failed, message = h.run()
    assert ok and circled == 2 and failed == 0, message     # an interrupt is not a failed goal
    assert "assistance request 30 confirmed" in message
    assert h.acks == [30] and h.reports == [30]
    assert h.tasks == ["dynamic_incident", "safe_passage"]
    assert h.stages() == ["ALIGN", "PLAN", "ENTRY", "INCIDENT_TRANSIT", "INCIDENT_ON_STATION", "INCIDENT_RETURN",
                          "PLAN", "ENTRY", "TRANSIT", "EXIT"]
    np.testing.assert_allclose(h.targets[3], STATION)
    np.testing.assert_allclose(h.targets[4], h.targets[1])  # back to where it was interrupted
    np.testing.assert_allclose(h.targets[5], h.targets[0])  # the loop had not closed: all of it again
    assert CALLS == [(None, True, True), (None, True, True)]
    assert node._pending is None and node._incident is None
    assert [x.state for x in node._incidents] == ["confirmed"] and node._incidents[0].report_seq == 412


def test_request_in_the_transit_goes_back_then_plans_from_there_without_the_entry_loop(node):
    h = Harness(node, results=[True] * LOOP + [request_now(node)], respond=True)
    later(0.3, node._on_confirm, confirm())
    ok, circled, failed, message = h.run()
    assert ok and circled == 2, message
    assert h.stages() == ["ALIGN", "PLAN", "ENTRY", "TRANSIT", "INCIDENT_TRANSIT", "INCIDENT_ON_STATION",
                          "INCIDENT_RETURN", "PLAN", "TRANSIT", "EXIT"]
    np.testing.assert_allclose(h.targets[LOOP + 1], STATION)
    np.testing.assert_allclose(h.targets[LOOP + 2], h.targets[LOOP - 1])     # back to the last goal it reached
    np.testing.assert_allclose(CALLS[1][0], h.targets[LOOP - 1], atol=0.01)  # and planned from there
    assert "red to starboard yes, green to port yes" in message
    assert not any("violation" in line for line in message.split("; "))  # the way out and back is not followed


def test_request_in_the_exit_circle_plans_back_to_it_with_nothing_left_to_pass(node):
    h = Harness(node)
    h.run()
    transit = count(h, "TRANSIT")
    del CALLS[:]
    h = Harness(node, results=[True] * (LOOP + transit + 1) + [request_now(node)], respond=True)
    later(0.3, node._on_confirm, confirm())
    ok, circled, failed, message = h.run()
    assert ok and circled == 2 and CALLS[1][1:] == (False, False)
    assert h.stages()[-3:] == ["PLAN", "TRANSIT", "EXIT"]


def test_without_incident_return_the_plan_starts_at_the_station(node):
    node.incident_return = False
    h = Harness(node, results=[True] * LOOP + [request_now(node)], respond=True)
    later(0.3, node._on_confirm, confirm())
    ok, circled, failed, message = h.run()
    assert ok and "INCIDENT_RETURN" not in h.phases()
    np.testing.assert_allclose(CALLS[1][0], STATION, atol=0.01)


def test_request_while_planning_is_answered_at_once_and_the_plan_is_made_again(node):
    h = Harness(node, respond=True)
    inner, slow = sr.plan_route, [True]

    def plan(*a, **k):
        if slow:
            slow.pop()
            time.sleep(0.6)
        return inner(*a, **k)

    sr.plan_route = plan
    later(0.2, node._on_assistance, assistance())
    later(0.5, node._on_confirm, confirm())
    try:
        ok, circled, failed, message = h.run()
    finally:
        sr.plan_route = inner
    assert ok and circled == 2 and "assistance request 30 confirmed" in message
    assert h.stages()[:6] == ["ALIGN", "PLAN", "INCIDENT_TRANSIT", "INCIDENT_ON_STATION", "INCIDENT_RETURN", "PLAN"]
    np.testing.assert_allclose(h.targets[0], STATION)
    np.testing.assert_allclose(h.targets[1], [0.0, -10.0])               # back to where the boat was holding
    assert h.tasks == ["dynamic_incident", "safe_passage"]


def test_request_during_align_is_answered_then_alignment_goes_on(node):
    h = Harness(node, respond=True, grid=geo.CostGrid.empty())           # nothing to lock on
    node.align_timeout = 0.5
    later(0.15, node._on_assistance, assistance())
    later(0.4, node._on_confirm, confirm())
    ok, circled, failed, message = h.run()
    assert ok and circled == 2 and "assistance request 30 confirmed" in message
    assert h.stages()[:5] == ["ALIGN", "INCIDENT_TRANSIT", "INCIDENT_ON_STATION", "INCIDENT_RETURN", "ALIGN"]
    np.testing.assert_allclose(h.targets[0], STATION)
    np.testing.assert_allclose(h.targets[1], [0.0, -10.0])


def test_request_for_another_vehicle_type_is_ignored(node):
    h = Harness(node, results=[True, request_now(node, vehicle_type="uuv")], respond=True)
    ok, circled, failed, message = h.run()
    assert ok and h.acks == [] and h.tasks == [] and "assistance" not in message
    assert "INCIDENT_TRANSIT" not in h.phases()


def test_request_is_ignored_when_the_goal_does_not_ask_for_it(node):
    h = Harness(node, results=[True, request_now(node)])
    assert h.run()[0] and h.acks == [] and "INCIDENT_TRANSIT" not in h.phases()


def clear(node):
    from std_srvs.srv import Trigger
    return node._on_clear_assistance(Trigger.Request(), Trigger.Response())


def test_request_with_no_goal_running_is_held_and_answered_first_by_the_next_goal(node):
    node._on_assistance(assistance(seq=30))               # nothing running: held, however long
    node._held.received -= 3600.0                         # an hour ago: no time limit
    assert node._pending is None and node._held.seq == 30
    h = Harness(node, respond=True)
    later(0.2, node._on_confirm, confirm())
    ok, circled, failed, message = h.run()
    assert ok and circled == 2 and "assistance request 30 confirmed after 36" in message
    assert h.acks == [30] and h.reports == [30] and node._held is None
    np.testing.assert_allclose(h.targets[0], STATION)     # before anything else, ALIGN included
    assert h.stages()[:3] == ["INCIDENT_TRANSIT", "INCIDENT_ON_STATION", "ALIGN"]   # nothing to go back to


def test_clear_assistance_drops_a_held_request(node):
    node._on_assistance(assistance(seq=30))
    response = clear(node)
    assert response.success and "dropped held assistance request 30" in response.message and node._held is None
    assert "no assistance request was held" in clear(node).message
    h = Harness(node, respond=True)
    ok, circled, failed, message = h.run()
    assert ok and h.acks == [] and "assistance" not in message


def test_clear_assistance_leaves_a_request_being_answered(node):
    h = Harness(node, results=[True, request_now(node)], respond=True)
    cleared = []
    later(0.15, lambda: cleared.append(clear(node).message))
    later(0.3, node._on_confirm, confirm())
    ok, circled, failed, message = h.run()
    assert ok and "assistance request 30 confirmed" in message
    assert "request 30 is being answered" in cleared[0]


def test_only_the_newest_idle_request_for_us_is_held(node):
    node._on_assistance(assistance(seq=30))
    node._on_assistance(assistance(seq=31, vehicle_type="uuv"))   # not ours: never held
    assert node._held.seq == 30
    node._on_assistance(assistance(seq=32, xy=[-20.0, -9.0]))
    assert node._held.seq == 32 and node._held.latitude == -20.0


def test_goal_that_does_not_answer_requests_leaves_the_held_one(node):
    node._on_assistance(assistance(seq=30))
    h = Harness(node)                                     # respond_to_assistance false
    assert h.run()[0] and h.acks == [] and node._held.seq == 30


def test_holding_can_be_switched_off(node):
    node.hold_idle_requests = False
    node._on_assistance(assistance(seq=30))
    assert node._held is None


def test_seq_can_come_again_in_another_goal(node):
    """RoboCommand may number from 1 again in the next run: the node must not turn that request down."""
    for _ in range(2):
        h = Harness(node, results=[True, request_now(node, seq=1)], respond=True)
        later(0.2, node._on_confirm, confirm())
        ok, circled, failed, message = h.run()
        assert ok and h.acks == [1] and "assistance request 1 confirmed" in message
        node._respond = False                             # as _execute leaves it when a goal ends


def test_incident_transit_and_the_way_back_get_the_incident_goal_timeout(node):
    h = Harness(node, results=[True, request_now(node)], respond=True)
    later(0.2, node._on_confirm, confirm())
    assert h.run()[0]
    assert h.timeouts[2] == h.timeouts[3] == node.incident_goal_timeout == 300.0
    assert h.timeouts[1] < 300.0


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

    h = Harness(node, results=[True, repeat], respond=True)
    later(0.2, node._on_assistance, assistance())
    later(0.3, node._on_confirm, confirm())
    ok, circled, failed, message = h.run()
    assert ok and h.acks == [30] and h.reports == [30] and len(node._incidents) == 1


def test_only_our_confirm_for_our_report_releases_the_hold(node):
    h = Harness(node, results=[True, request_now(node)], respond=True)
    node.readiness_confirm_timeout = 0.6
    later(0.1, node._on_confirm, confirm(vehicle_id="USV2"))        # another vehicle's
    later(0.15, node._on_confirm, confirm(report_seq=411))          # ours, another report
    started = time.monotonic()
    ok, circled, failed, message = h.run()
    assert time.monotonic() - started >= 0.6
    assert ok and circled == 2 and "assistance request 30 confirm_timeout" in message
    assert h.tasks == ["dynamic_incident", "safe_passage"]          # resumes on its own after the timeout


def test_confirm_that_beats_the_report_seq_still_counts(node):
    def arrive():
        node._on_assistance(assistance())
        node._on_confirm(confirm())                       # before the ReadinessReport action has returned 412
        return True

    h = Harness(node, results=[True, arrive], respond=True)
    started = time.monotonic()
    ok, circled, failed, message = h.run()
    assert ok and "assistance request 30 confirmed" in message and time.monotonic() - started < 2.0 + 10.0


def test_newer_request_replaces_the_one_in_hand(node):
    other = [-20.0, -9.0]
    h = Harness(node, respond=True, results=[True, request_now(node, seq=30), request_now(node, seq=31, xy=other)])
    later(0.3, node._on_confirm, confirm())
    ok, circled, failed, message = h.run()
    assert ok and "assistance request 30 superseded" in message and "assistance request 31 confirmed" in message
    assert h.acks == [30, 31] and h.reports == [31]
    np.testing.assert_allclose(h.targets[2], STATION)
    np.testing.assert_allclose(h.targets[3], other)
    np.testing.assert_allclose(h.targets[4], h.targets[0])          # back once, after both
    assert h.tasks[-1] == "safe_passage" and h.tasks.count("safe_passage") == 1


def test_request_on_the_way_back_is_answered_then_the_boat_goes_back_again(node):
    other = [-20.0, -9.0]
    h = Harness(node, respond=True,
                results=[True, request_now(node, seq=30), True, request_now(node, seq=31, xy=other)])
    later(0.2, node._on_confirm, confirm())
    later(0.6, node._on_confirm, confirm(report_seq=412, seq=100))
    ok, circled, failed, message = h.run()
    assert ok and "assistance request 30 confirmed" in message and "assistance request 31 confirmed" in message
    np.testing.assert_allclose(h.targets[2], STATION)
    np.testing.assert_allclose(h.targets[3], h.targets[0])          # on the way back when 31 arrives
    np.testing.assert_allclose(h.targets[4], other)
    np.testing.assert_allclose(h.targets[5], h.targets[0])          # back again
    assert h.tasks.count("safe_passage") == 1                       # the task is switched back once, at the end


def test_unreachable_station_sends_no_readiness_report_and_resumes(node):
    h = Harness(node, respond=True, results=[True, request_now(node), False, False, False])
    ok, circled, failed, message = h.run()
    assert ok and circled == 2 and "assistance request 30 unreachable" in message
    assert h.acks == [30] and h.reports == [] and h.tasks == ["dynamic_incident", "safe_passage"]
    assert "INCIDENT_ON_STATION" not in h.phases()


def test_way_back_that_cannot_be_driven_plans_from_where_the_boat_is(node):
    h = Harness(node, respond=True, results=[True] * LOOP + [request_now(node), True, False, False, False])
    later(0.3, node._on_confirm, confirm())
    ok, circled, failed, message = h.run()
    assert ok and circled == 2 and failed == 0
    np.testing.assert_allclose(CALLS[1][0], STATION, atol=0.01)


def test_readiness_report_that_cannot_be_sent_resumes_without_waiting(node):
    h = Harness(node, results=[True, request_now(node)], respond=True)
    h.report_seq = None
    started = time.monotonic()
    ok, circled, failed, message = h.run()
    assert ok and "assistance request 30 report_failed" in message and time.monotonic() - started < 12.0


def test_cancel_stops_everything(node):
    h = Harness(node)

    def cancel():
        h.goal.is_cancel_requested = True
        return True

    h.results = [True, cancel]
    with pytest.raises(spn.Canceled):
        h.run()
    assert len(h.targets) == 2 and node._tracker is None


def test_cancel_while_planning(node):
    h = Harness(node)
    inner = sr.plan_route
    sr.plan_route = lambda *a, **k: (time.sleep(0.5), inner(*a, **k))[1]
    later(0.2, setattr, h.goal, "is_cancel_requested", True)
    try:
        with pytest.raises(spn.Canceled):
            h.run()
    finally:
        sr.plan_route = inner
    assert h.targets == []


def test_cancel_while_holding_on_station(node):
    h = Harness(node, results=[True, request_now(node)], respond=True)
    later(0.3, setattr, h.goal, "is_cancel_requested", True)
    with pytest.raises(spn.Canceled):
        h.run()
    assert h.reports == [30] and h.tasks == ["dynamic_incident"]
    assert node._incidents[0].state == "on_station"   # _execute closes it as canceled when the goal ends


def test_confirm_counts_when_an_earlier_report_try_may_have_gone_out(node):
    """A failed try can reach RoboCommand under a seq we never learn; its confirm must still release the hold."""
    h = Harness(node, results=[True, request_now(node)], respond=True)
    h.report_seq, h.report_unsure = 413, True
    node._on_confirm(confirm(report_seq=400))                       # before the goal: never counts
    later(0.3, node._on_confirm, confirm(report_seq=412))           # the try we think failed
    started = time.monotonic()
    ok, circled, failed, message = h.run()
    assert ok and "assistance request 30 confirmed" in message and time.monotonic() - started < 4.0


def test_confirm_from_before_the_station_does_not_count_even_when_unsure(node):
    def arrive():
        node._on_assistance(assistance())
        node._on_confirm(confirm(report_seq=390))                   # in this goal, but before we got on station
        return True

    h = Harness(node, results=[True, arrive], respond=True)
    h.report_seq, h.report_unsure = 413, True
    node.readiness_confirm_timeout = 0.5
    ok, circled, failed, message = h.run()
    assert ok and "assistance request 30 confirm_timeout" in message


def test_request_during_the_switch_back_is_answered_before_going_on(node):
    other = [-20.0, -9.0]
    h = Harness(node, results=[True, request_now(node, seq=30)], respond=True)
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
    np.testing.assert_allclose(h.targets[4], other)
    assert h.phases().count("PLAN") == 2                            # planned again once, after both


# Alignment

LAYOUT = np.array(positions(channel()))
STATES = [b.state for b in channel()]
SHIFT = np.array([2.0, 6.0])


def feed(node, grid, times=6, period=0.05):
    """Costmaps arriving while the goal runs: a new message, so a new grid, each time."""
    def loop():
        for _ in range(times):
            time.sleep(period)
            node._align_step(geo.CostGrid(grid.data, grid.resolution, grid.origin_x, grid.origin_y))
    threading.Thread(target=loop, daemon=True).start()


def test_layout_is_aligned_to_the_costmap_before_anything_is_planned(node):
    """Drone layout 6 m off: the node holds in ALIGN, locks the shift, then routes around the real buoys."""
    grid = costmap(LAYOUT + SHIFT)
    h = Harness(node, boat=(2.0, -4.0), grid=grid)
    node.align_timeout = 5.0
    feed(node, grid)
    ok, circled, failed, message = h.run()
    assert ok and circled == 2, message
    assert h.stages() == ["ALIGN", "PLAN", "ENTRY", "TRANSIT", "EXIT"]
    loop = np.array(h.targets[:LOOP])
    radii = np.linalg.norm(loop - (np.array(ENTRY) + SHIFT), axis=1)
    assert radii.max() - radii.min() < 0.15, radii                 # around the real entry buoy, not the drone's
    assert "BLOB" in message and "PREDICTED" not in message
    assert "red to starboard yes, green to port yes" in message
    assert h.goal.feedback[-1].shift_inliers == len(LAYOUT)
    node._aligner, node._buoys = ba.Aligner(LAYOUT, node.align_params), []
    node._publish_markers()  # must not raise
    node._aligner = None


def test_no_lock_goes_on_with_the_drone_positions_after_the_timeout(node):
    h = Harness(node, grid=geo.CostGrid.empty())
    node.align_timeout = 0.3
    started = time.monotonic()
    ok, circled, failed, message = h.run()
    assert ok and circled == 2 and "PREDICTED" in message
    assert time.monotonic() - started >= 0.3
    radii = np.linalg.norm(np.array(h.targets[:LOOP]) - ENTRY, axis=1)
    assert radii.max() - radii.min() < 0.05                        # the loop is around the drone's entry position


def test_cancel_while_aligning(node):
    h = Harness(node, grid=geo.CostGrid.empty())
    node.align_timeout = 5.0
    threading.Timer(0.2, lambda: setattr(h.goal, "is_cancel_requested", True)).start()
    with pytest.raises(spn.Canceled):
        h.run()
    assert h.targets == []


def test_blob_near_the_drone_position_is_used_without_a_lock(node):
    """A layout that cannot lock a shift: a blob within the snap radius still places its buoy."""
    moved = LAYOUT.copy()
    moved[STATES.index("flashing_blue")] += [0.6, 0.0]
    h = Harness(node, grid=costmap(moved))
    ok, circled, failed, message = h.run()
    assert ok and "5:flashing_blue BLOB 0.6" in message, message
    radii = np.linalg.norm(np.array(h.targets[:LOOP]) - (np.array(ENTRY) + [0.6, 0.0]), axis=1)
    assert radii.max() - radii.min() < 0.15


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
