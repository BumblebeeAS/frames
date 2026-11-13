#!/usr/bin/env python3
"""RobotX 2026 Task 1 (Safe Passage) on the ASV, from the drone's classified buoys.

Serves bb_robotx_msgs/SafePassage. A goal carries the drone's buoys (lat/lon + state). They are converted once to
the global frame with fromLL, ordered (entry, reds and greens along the course, exit; blacks are never circled, and
buoys in any other state, such as unknown, are dropped on arrival and take no part in the alignment either), and
circled one at a time with stop-and-go TrackPath goals:

  TRANSIT  TrackPath to p0, where the boat joins the circle tangentially in the buoy's direction
  CIRCLE   TrackPath to p1 .. pN+overshoot, N points per 360 deg; each goal waits for the previous one's result

The entry is fully circled clockwise and the exit counterclockwise. Reds and greens only have to be passed on the
right side, so each gets one TrackPath goal: a waypoint pass_offset beside it, left of a red (it stays on starboard)
and right of a green (on port) going along the course, which runs from the entry to the exit (to the middle of the
reds and greens when no exit is known). A red and a green close enough to go between, facing the course, are a gate:
one waypoint midway passes both. Reds, greens and gates are visited in order along the course. With pass_points > 0
a red or green gets that many circle steps from a tangent start point instead (3 of 6 is a semicircle).

The drone's positions can be metres off as one layout, so before driving the node lines the layout up with the
costmap (frames.utils.buoy_align): ALIGN holds until the shift locks or align_timeout passes, and the buoys are
ordered only then. Each circle is centred on the buoy estimate E_i, read again before every goal: the small
costmap blob nearest the shifted drone position, else that position. Without a lock the shift is zero.

Retries: a failed point is re-placed and tried again point_retries times, then skipped; max_consecutive_skips in a
row, or a start point that cannot be reached, fails the attempt. A buoy gets buoy_attempts attempts. Failed reds and
greens are tried once more before the exit. A buoy that still fails is logged and skipped.

Task 4 Core: with respond_to_assistance, an AssistanceRequest for this vehicle type interrupts whatever the goal is
doing. The node sends the IncidentAck and switches the heartbeat task, drives to the requested position, sends the
ReadinessReport once there, and holds (the boat station-keeps without a goal) until the ReadinessConfirm naming that
report arrives or readiness_confirm_timeout passes. Then it switches the task back and the interrupted buoy starts
over from TRANSIT. Requests are taken once per seq; a newer request replaces the one in hand. A request that
arrives while no goal answers them is held, with no time limit, and answered first by the next goal that does. The
clear_assistance service (std_srvs/Trigger) drops a held request: the mission tree calls it at run start, so nothing
from before the run is acted on. The hold is in memory only, so a relaunched node starts without one.

Conventions: everything is planned in global_frame_id (asv5/map, ENU). TrackPath goals are GOAL_POSE there with
specified_heading false and the heading along the circle, so the boat ends each point facing the next.
"""

import math
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import rclpy
import tf2_ros
from bb_controls_msgs.action import TrackPath
from bb_robotx_msgs.action import SafePassage, SendIncidentAck, SendReadinessReport, SetTask
from bb_robotx_msgs.msg import AssistanceRequest, ReadinessConfirm
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time as RosTime
from robot_localization.srv import FromLL
from std_msgs.msg import ColorRGBA
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray

from frames.utils import buoy_align as ba
from frames.utils import safe_passage_geometry as geo

# nav2 costmaps publish reliable transient local
LATCHED_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST, depth=1,
    reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL,
)
# Task 4 Core commands: the bridge does not latch them (an old request must not move a relaunched node)
COMMAND_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST, depth=10,
    reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE,
)
POLL_S = 0.05
MARKER_PERIOD_S = 0.5

STATE_COLOURS = {
    geo.ENTRY: (0.1, 0.3, 1.0), geo.EXIT: (0.0, 0.9, 0.9), geo.RED: (1.0, 0.1, 0.1),
    geo.GREEN: (0.1, 0.9, 0.1), geo.BLACK: (0.3, 0.3, 0.3),
}


class Canceled(Exception):
    """The SafePassage goal was canceled."""


class Assistance(Exception):
    """An AssistanceRequest is waiting: drop what is being driven and answer it."""


class Restart(Exception):
    """The buoy turned out to be somewhere else entirely: start its circle over."""


@dataclass
class Incident:
    """One Task 4 Core AssistanceRequest and how far its answer got."""
    seq: int                            # RxCommand seq, quoted by the IncidentAck and the ReadinessReport
    latitude: float
    longitude: float
    received: float                     # monotonic
    state: str = "received"             # received | transit | on_station | then how it ended:
    #                                     confirmed | confirm_timeout | unreachable | report_failed | no_position |
    #                                     superseded | canceled
    report_seq: Optional[int] = None    # seq of the report that carried our ReadinessReport
    report_unsure: bool = False         # a try at that report failed: it may still have reached RoboCommand
    on_station: Optional[float] = None  # monotonic, when the boat got there
    ended: Optional[float] = None


@dataclass
class Buoy:
    index: int
    state: str
    xy: np.ndarray                      # C_i
    sigma: int = 0                      # +1 clockwise, -1 counterclockwise, 0 never circled
    status: str = "pending"             # pending | circled (dealt with: circled, or passed for red/green) | failed
    attempts: int = 0
    estimate: Optional[ba.Estimate] = None  # the one the last circle used
    radius: float = 0.0
    gate: Optional[int] = None          # index of the buoy it was passed with as a gate


class SafePassageNode(Node):
    def __init__(self):
        super().__init__("safe_passage_node")
        param = lambda name, default: self.declare_parameter(name, default).value  # noqa: E731

        # Wiring
        self.costmap_topic = param("costmap_topic", "/asv5/global_costmap/costmap")
        self.vehicle_id = param("vehicle_id", "USV1")      # as the OCS and RoboCommand name this vehicle
        self.vehicle_type = param("vehicle_type", "usv")   # AssistanceRequests for other types are not ours
        self.global_frame_id = param("global_frame_id", "asv5/map")
        self.base_link_frame_id = param("base_link_frame_id", "asv5/base_link")
        self.service_timeout = param("service_timeout", 2.0)

        # Alignment of the drone's layout with the costmap
        defaults = ba.AlignParams()
        self.align_params = ba.AlignParams(**{
            name: type(getattr(defaults, name))(param(name, getattr(defaults, name)))
            for name in ba.AlignParams.__dataclass_fields__})
        self.align_timeout = param("align_timeout", 30.0)

        # Estimate and circle
        self.estimate_move_threshold = param("estimate_move_threshold", 0.3)
        self.estimate_restart_threshold = param("estimate_restart_threshold", 1.5)
        self.buoy_restarts = param("buoy_restarts", 2)
        self.circle_radius = param("circle_radius", 2.5)
        self.circle_radius_min = param("circle_radius_min", 1.6)
        self.circle_radius_step = param("circle_radius_step", 0.25)
        self.neighbour_clearance = param("neighbour_clearance", 1.25)
        self.circle_points = param("circle_points", 6)
        self.circle_overshoot_points = param("circle_overshoot_points", 1)
        self.pass_points = param("pass_points", 0)      # reds and greens: 0 = one waypoint beside the buoy
        self.pass_offset = param("pass_offset", 1.8)
        self.gate_min_width = param("gate_min_width", 2.4)
        self.gate_max_width = param("gate_max_width", 6.0)
        self.gate_max_angle = param("gate_max_angle", 60.0)  # deg
        self.cost_threshold = param("cost_threshold", 22)

        # TrackPath
        self.goal_tolerance = param("goal_tolerance", 1.0)
        self.goal_stabilize = param("goal_stabilize", 0)
        self.goal_sample_radius = param("goal_sample_radius", 0.5)
        self.max_replans = param("max_replans", 5)
        self.goal_timeout = param("goal_timeout", 60.0)

        # Retries
        self.point_retries = param("point_retries", 1)
        self.max_consecutive_skips = param("max_consecutive_skips", 2)
        self.buoy_attempts = param("buoy_attempts", 2)
        self.requeue_failed = param("requeue_failed", True)

        # Task 4 Core
        self.hold_idle_requests = param("hold_idle_requests", True)
        self.readiness_confirm_timeout = param("readiness_confirm_timeout", 900.0)
        self.incident_goal_retries = param("incident_goal_retries", 2)
        self.incident_sample_radius = param("incident_sample_radius", 0.5)
        self.incident_tolerance = param("incident_tolerance", 1.0)
        self.incident_goal_timeout = param("incident_goal_timeout", 300.0)  # the station can be across the course
        self.report_retries = param("report_retries", 3)
        self.report_timeout = param("report_timeout", 40.0)   # the bridge gives the OCS 30 s
        self.incident_task = param("incident_task", "dynamic_incident")
        self.resume_task = param("resume_task", "safe_passage")

        if self.goal_sample_radius > 0.0 and self.goal_sample_radius < 0.25:
            self.get_logger().warning("goal_sample_radius under the tracker's sample_spacing (0.25 m) samples nothing")

        self.cb_group = ReentrantCallbackGroup()
        self.costmap_group = MutuallyExclusiveCallbackGroup()  # a solve takes a while: never two at once
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self._grid: Optional[geo.CostGrid] = None
        self._solved_grid: Optional[geo.CostGrid] = None  # the last costmap the running goal's aligner saw
        self.create_subscription(OccupancyGrid, self.costmap_topic, self._on_costmap, LATCHED_QOS,
                                 callback_group=self.costmap_group)
        self.create_subscription(AssistanceRequest, param("assistance_topic", "/asv5/task4/assistance_request"),
                                 self._on_assistance, COMMAND_QOS, callback_group=self.cb_group)
        self.create_subscription(ReadinessConfirm, param("readiness_confirm_topic", "/asv5/task4/readiness_confirm"),
                                 self._on_confirm, COMMAND_QOS, callback_group=self.cb_group)
        self.marker_pub = self.create_publisher(MarkerArray, param("marker_topic", "/asv5/safe_passage/markers"), 1)
        self.create_timer(MARKER_PERIOD_S, self._publish_markers, callback_group=self.cb_group)

        # Goal state shared with the costmap thread: the aligner, the buoys, the circle being driven
        self._lock = threading.Lock()
        self._aligner: Optional[ba.Aligner] = None
        self._buoys: list[Buoy] = []
        self._circle: list = []      # [(xy, yaw)] being driven, for markers
        self._target: Optional[np.ndarray] = None
        self._logged_t = np.zeros(2)
        self._course: Optional[np.ndarray] = None     # unit direction of the course for the running goal, if it has one
        # Task 4 Core, all under _lock. A request moves (_held ->) _pending -> _incident -> ended and never back.
        self._respond = False                         # the running goal answers AssistanceRequests
        self._seen: set[int] = set()                  # request seqs taken or turned down by the running goal
        self._task_wanted: Optional[str] = None       # the heartbeat task last asked for: older sends give way
        self._held: Optional[Incident] = None         # arrived with no goal answering: kept until one does, or cleared
        self._pending: Optional[Incident] = None      # accepted, not yet picked up by the goal thread
        self._incident: Optional[Incident] = None     # being answered
        self._incidents: list[Incident] = []          # of the running goal, in order, for its result
        self._confirms: dict[int, float] = {}         # report_seq -> when its ReadinessConfirm for us arrived

        self.track_path_name = param("track_path_action", "/asv5/track_path")
        self.track_path = ActionClient(self, TrackPath, self.track_path_name, callback_group=self.cb_group)
        self.from_ll = self.create_client(FromLL, param("from_ll_service", "/asv5/fromLL"),
                                          callback_group=self.cb_group)
        self.incident_ack_name = param("incident_ack_action", "/asv5/report/incident_ack")
        self.incident_ack = ActionClient(self, SendIncidentAck, self.incident_ack_name, callback_group=self.cb_group)
        self.readiness_name = param("readiness_action", "/asv5/report/readiness")
        self.readiness = ActionClient(self, SendReadinessReport, self.readiness_name, callback_group=self.cb_group)
        self.set_task_name = param("set_task_action", "/asv5/set_task")
        self.set_task = ActionClient(self, SetTask, self.set_task_name, callback_group=self.cb_group)
        self.create_service(Trigger, param("clear_assistance_service", "/asv5/safe_passage/clear_assistance"),
                            self._on_clear_assistance, callback_group=self.cb_group)

        self._busy = threading.Lock()
        self.server = ActionServer(
            self, SafePassage, param("action_name", "/asv5/safe_passage"),
            execute_callback=self._execute, goal_callback=self._on_goal,
            cancel_callback=lambda _: CancelResponse.ACCEPT, callback_group=self.cb_group,
        )
        self.get_logger().info(f"Safe Passage ready: costmap {self.costmap_topic}; answers Task 4 assistance requests "
                               f"for {self.vehicle_type} as {self.vehicle_id} while a goal asks for it")

    # Goal handling

    def _on_goal(self, goal) -> GoalResponse:
        if self._busy.locked():
            self.get_logger().error("Rejected Safe Passage goal: one is already running")
            return GoalResponse.REJECT
        if not any(b.state in geo.SIGMA for b in goal.buoys):
            self.get_logger().error("Rejected Safe Passage goal: no entry, red, green or exit buoy")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _execute(self, goal_handle):
        with self._busy:
            result = SafePassage.Result()
            try:
                summary = self._run(goal_handle)
            except Canceled:
                result.success, result.message = False, "canceled"
                goal_handle.canceled()
                return result
            finally:
                with self._lock:
                    self._aligner, self._buoys, self._circle, self._target = None, [], [], None
                    self._respond = False
                    self._task_wanted = None  # a task switch still being retried stops: the tree owns the task again
                    for incident in (self._pending, self._incident):
                        if incident is not None and incident.ended is None:
                            self._end(incident, "canceled")
                    self._pending = self._incident = None
            if summary is None:
                result.success, result.message = False, "no buoy could be converted to the map frame"
                goal_handle.abort()
            else:
                result.success, result.circled, result.failed, result.message = summary
                goal_handle.succeed()
            self.get_logger().info(f"Safe Passage done: {result.message}")
            return result

    def _run(self, goal_handle):
        request = goal_handle.request
        with self._lock:
            self._respond, self._incidents, self._confirms = bool(request.respond_to_assistance), [], {}
            self._seen = set()  # per goal: RoboCommand may number its commands from 1 again in another run
            if self._respond and self._held is not None:
                held, self._held = self._held, None
                self._seen.add(held.seq)
                self._pending = held
                self._incidents.append(held)
                self.get_logger().warning(f"Assistance request {held.seq}, held for "
                                          f"{time.monotonic() - held.received:.0f} s, is answered first")
        buoys = []
        known = (geo.ENTRY, geo.EXIT, geo.RED, geo.GREEN, geo.BLACK)
        for i, b in enumerate(request.buoys):
            if b.state not in known:
                # An unclassified detection may not be a buoy at all: it is not circled and not aligned on either
                self.get_logger().info(f"Buoy {i} ({b.state or 'no state'}): not a known state, dropped")
                continue
            xy = self._from_ll(b.position.latitude, b.position.longitude)
            if xy is None:
                self.get_logger().error(f"Buoy {i} ({b.state}): fromLL failed, dropped")
                continue
            buoys.append(Buoy(len(buoys), b.state, xy, geo.SIGMA.get(b.state, 0)))
        if not any(b.sigma for b in buoys):
            return None
        aligner = ba.Aligner([b.xy for b in buoys], self.align_params)
        with self._lock:
            self._aligner, self._buoys, self._logged_t, self._solved_grid = aligner, buoys, np.zeros(2), None
        self.get_logger().info(
            "Safe Passage buoys: " + ", ".join(f"{b.index}:{b.state}({b.xy[0]:.1f},{b.xy[1]:.1f})" for b in buoys)
            + f"; tightest spacing {float(aligner.nn.min()):.2f} m; respond_to_assistance {self._respond}")
        self._answer_assistance(goal_handle, len(buoys))  # a request held from before this goal comes first
        while True:
            try:
                self._align(goal_handle, len(buoys))
                break
            except Assistance:
                self._answer_assistance(goal_handle, len(buoys))

        # Ordered where the buoys are now believed to be: a shift of metres changes which is nearest
        boat = self._boat_xy()
        with self._lock:
            placed = aligner.predicted()
        states = [b.state for b in buoys]
        origin = boat if boat is not None else placed.mean(axis=0)
        start, self._course = geo.course(list(placed), states, origin)
        order = geo.order_buoys(list(placed), states, origin)
        legs = self._legs(buoys, order, start)
        heading = ("none" if self._course is None
                   else f"{math.degrees(math.atan2(self._course[1], self._course[0])):.0f} deg")
        self.get_logger().info(f"Course {heading}; legs " + ", ".join(
            "gate " + "-".join(str(b.index) for b in leg) if len(leg) > 1 else str(leg[0].index) for leg in legs))

        total = len(legs)
        for position, leg in enumerate(legs):
            if leg[0].state != geo.EXIT:
                self._do_leg(goal_handle, leg, position, total)
        if self.requeue_failed:
            for position, leg in enumerate(legs):
                for b in leg:  # a failed gate is retried one buoy at a time
                    if b.status == "failed" and b.state in (geo.RED, geo.GREEN):
                        self.get_logger().info(f"Buoy {b.index} ({b.state}): requeued for one more attempt before "
                                               "the exit")
                        b.attempts, b.gate = max(0, int(self.buoy_attempts) - 1), None
                        self._do_leg(goal_handle, [b], position, total)
        for position, leg in enumerate(legs):
            if leg[0].state == geo.EXIT:
                self._do_leg(goal_handle, leg, position, total)
        self._answer_assistance(goal_handle, total)  # a request that came in as the last buoy finished

        circled = [b for b in buoys if b.status == "circled"]
        failed = [b for b in buoys if b.status == "failed"]
        lines = []
        for i in order:
            b = buoys[i]
            e = b.estimate
            detail = (f", {e.source} {np.linalg.norm(e.xy - b.xy):.2f} m from the drone, R {b.radius:.2f}"
                      if e is not None else "")
            outcome = self._done_word(b) if b.status == "circled" else b.status
            if b.gate is not None:
                outcome += f" as a gate with {b.gate}"
            lines.append(f"{i}:{b.state} {outcome} ({b.attempts} attempt(s){detail})")
        with self._lock:
            incidents = [f"assistance request {x.seq} {x.state}"
                         + (f" after {x.ended - x.received:.0f} s" if x.ended is not None else "")
                         for x in self._incidents]
        return not failed, len(circled), len(failed), "; ".join(lines + incidents)

    # Alignment

    def _align(self, goal_handle, total: int):
        """Hold until the drone layout locks onto the costmap or align_timeout passes. Raises Canceled."""
        self._align_step(self._snapshot())  # the costmap already here; later ones arrive through _on_costmap
        start = time.monotonic()

        shown = [None]

        def locked():
            self._check(goal_handle)
            with self._lock:
                matched, done = self._aligner.matched, self._aligner.locked
            if matched != shown[0]:
                shown[0] = matched
                goal_handle.publish_feedback(SafePassage.Feedback(phase="ALIGN", buoy_total=total,
                                                                  shift_inliers=matched))
            return done

        if self.align_timeout > 0.0 and self._wait(goal_handle, locked, self.align_timeout):
            return
        with self._lock:
            aligner = self._aligner
            if aligner.locked:
                return
            last = aligner.last
        self.get_logger().warning(
            f"Layout not aligned after {time.monotonic() - start:.0f} s: continuing with the drone's positions as "
            f"they are (last solve t ({last.t[0]:.2f}, {last.t[1]:.2f}) score {last.score:.1f}, runner-up "
            f"{last.second:.1f}, {len(aligner.buoy_blobs)} buoy-sized blobs). It keeps solving and may still lock")

    def _align_step(self, grid: geo.CostGrid):
        """One costmap through the aligner of the running goal, logged."""
        with self._lock:
            aligner = self._aligner
            if aligner is None or grid is self._solved_grid:  # nothing running, or this costmap already counted
                return
            self._solved_grid = grid
        started = time.monotonic()
        all_blobs = ba.blobs(grid, aligner.candidates, aligner.reach(), aligner.p.join_gap)
        with self._lock:
            if self._aligner is not aligner:
                return
            was_locked = aligner.locked
            fresh = aligner.update(all_blobs)
            t, locked, matched = aligner.t.copy(), aligner.locked, aligner.matched
            small, large = len(aligner.buoy_blobs), len(aligner.structures)
        took = (time.monotonic() - started) * 1e3
        verdict = "accepted" if fresh.accepted else "rejected"
        detail = (f"solve t ({fresh.t[0]:.2f}, {fresh.t[1]:.2f}) score {fresh.score:.1f} runner-up {fresh.second:.1f} "
                  f"at ({fresh.second_t[0]:.1f}, {fresh.second_t[1]:.1f}) {verdict}; {small} buoy-sized blobs, "
                  f"{large} larger; {took:.0f} ms")
        if locked and not was_locked:
            self._logged_t = t
            self.get_logger().info(f"Layout LOCKED: shift t ({t[0]:.2f}, {t[1]:.2f}), {matched} of "
                                   f"{len(aligner.candidates)} drone buoys on a blob. {detail}")
        elif not locked:
            self.get_logger().info(f"Aligning: {detail}")
        elif np.linalg.norm(t - self._logged_t) > 0.2:
            self.get_logger().info(f"Shift now t ({t[0]:.2f}, {t[1]:.2f}), was ({self._logged_t[0]:.2f}, "
                                   f"{self._logged_t[1]:.2f}); {matched} drone buoys on a blob. {detail}")
            self._logged_t = t
        else:
            self.get_logger().debug(f"Locked t ({t[0]:.2f}, {t[1]:.2f}), {matched} matched. {detail}")

    # Legs: one manoeuvre each

    def _legs(self, buoys: list, order: list, start) -> list:
        """The goal's manoeuvres in order: [entry], reds, greens and [red, green] gates along the course, [exit]."""
        singles = [[buoys[i]] for i in order]
        if self._course is None or self.pass_points > 0:
            return singles
        where = [self._estimate(b).xy for b in buoys]  # on their blobs where they have one
        gates = geo.find_gates(where, [b.state for b in buoys], self._course, self.gate_min_width,
                               self.gate_max_width, math.radians(self.gate_max_angle))
        paired = {i for gate in gates for i in gate}
        middle = [leg for leg in singles if leg[0].state in (geo.RED, geo.GREEN) and leg[0].index not in paired]
        middle += [[buoys[r], buoys[g]] for r, g in gates]
        middle.sort(key=lambda leg: float((np.mean([where[b.index] for b in leg], axis=0) - start) @ self._course))
        return ([leg for leg in singles if leg[0].state == geo.ENTRY] + middle
                + [leg for leg in singles if leg[0].state == geo.EXIT])

    def _do_leg(self, goal_handle, leg: list, position: int, total: int):
        waypoint = self.pass_points <= 0 and self._course is not None and leg[0].state in (geo.RED, geo.GREEN)
        if waypoint:
            self._do_pass(goal_handle, leg, position, total)
        else:
            self._do_buoy(goal_handle, leg[0], position, total)

    def _do_pass(self, goal_handle, leg: list, position: int, total: int):
        """Pass a red or green, or a red-green gate, with one waypoint. The buoys of a gate share their attempts."""
        what = (f"Gate {leg[0].index}-{leg[1].index} (red-green)" if len(leg) > 1
                else f"Buoy {leg[0].index} ({leg[0].state})")
        while max(b.attempts for b in leg) < self.buoy_attempts:
            for b in leg:
                b.attempts += 1
            label = f"{what} attempt {leg[0].attempts}/{self.buoy_attempts}"
            try:
                ok = self._pass_attempt(goal_handle, leg, label, position, total)
            except Assistance:
                for b in leg:
                    b.attempts -= 1  # an incident is not a failed attempt
                self._answer_assistance(goal_handle, total)
                continue
            if ok:
                for b in leg:
                    b.status = "circled"
                self.get_logger().info(f"{label}: passed")
                return
            self.get_logger().warning(f"{label}: failed")
        for b in leg:
            b.status = "failed"

    def _pass_attempt(self, goal_handle, leg: list, label: str, position: int, total: int) -> bool:
        """One waypoint, read again and retried point_retries times. Raises Assistance or Canceled."""
        for retry in range(1 + max(0, int(self.point_retries))):
            estimates = [self._estimate(b) for b in leg]
            for b, e in zip(leg, estimates):
                b.estimate = e
            if len(leg) > 1:  # between them, heading the way that keeps the red on starboard
                red, green = estimates[0].xy, estimates[1].xy
                across = green - red
                width = float(np.linalg.norm(across))
                target = (red + green) / 2.0
                yaw = math.atan2(-across[0], across[1])
                for b, other in zip(leg, reversed(leg)):
                    b.radius, b.gate = width / 2.0, other.index
                detail = (f"midway between ({red[0]:.2f}, {red[1]:.2f}) from {estimates[0].source} and "
                          f"({green[0]:.2f}, {green[1]:.2f}) from {estimates[1].source}, {width:.2f} m apart")
            else:
                b, e = leg[0], estimates[0]
                nearest = min((float(np.linalg.norm(np.asarray(n) - e.xy)) for n in self._neighbours(b)),
                              default=math.inf)
                # Beside it by pass_offset, less when another buoy is close, never less than half a gate's width
                offset = max(min(self.pass_offset, nearest - self.neighbour_clearance), self.gate_min_width / 2.0)
                target = geo.pass_waypoint(e.xy, b.state, self._course, offset)
                yaw = math.atan2(self._course[1], self._course[0])
                b.radius = offset
                detail = (f"{offset:.2f} m to the {'left' if b.state == geo.RED else 'right'} of ({e.xy[0]:.2f}, "
                          f"{e.xy[1]:.2f}) from {e.source}, {np.linalg.norm(e.xy - b.xy):.2f} m from the drone "
                          f"(nearest other buoy {nearest:.2f} m)")
            self.get_logger().info(f"{label}: waypoint ({target[0]:.2f}, {target[1]:.2f}) {detail}")
            with self._lock:
                self._circle = [(target, yaw)]
            self._feedback(goal_handle, "TRANSIT", leg[0], position, total, 0, 0, estimates[0].source)
            if self._drive(goal_handle, target, yaw, f"{label} waypoint"):
                return True
        return False

    # One buoy, circled

    def _do_buoy(self, goal_handle, buoy: Buoy, position: int, total: int):
        restarts = 0
        while buoy.attempts < self.buoy_attempts:
            buoy.attempts += 1
            label = f"Buoy {buoy.index} ({buoy.state}) attempt {buoy.attempts}/{self.buoy_attempts}"
            try:
                ok = self._attempt(goal_handle, buoy, label, position, total, restarts < self.buoy_restarts)
            except Assistance:
                buoy.attempts -= 1  # an incident is not a failed attempt: the buoy starts over afterwards
                self._answer_assistance(goal_handle, total)
                continue
            except Restart:
                buoy.attempts -= 1  # nor is finding the buoy elsewhere (the layout locked late, say)
                restarts += 1
                continue
            if ok:
                buoy.status = "circled"
                self.get_logger().info(f"{label}: {self._done_word(buoy)}")
                return
            self.get_logger().warning(f"{label}: failed")
        buoy.status = "failed"

    def _attempt(self, goal_handle, buoy: Buoy, label: str, position: int, total: int,
                 may_restart: bool = True) -> bool:
        estimate = self._estimate(buoy)
        boat = self._boat_xy()
        if boat is None:
            return False
        radius, why = geo.choose_radius(self._snapshot(), estimate.xy, self._neighbours(buoy), self.circle_radius,
                                        self.circle_radius_min, self.circle_radius_step, self.neighbour_clearance,
                                        self.cost_threshold)
        phi0 = geo.start_angle(estimate.xy, radius, boat, buoy.sigma)
        centre = estimate.xy

        # Entry and exit are fully circled; a red or green is passed with a semicircle on its side
        steps = int(self.pass_points) if buoy.state in (geo.RED, geo.GREEN) else None

        def place(c):
            return geo.circle_points(c, radius, phi0, buoy.sigma, self.circle_points, self.circle_overshoot_points,
                                     steps)

        points = place(centre)
        buoy.estimate, buoy.radius = estimate, radius
        self.get_logger().info(
            f"{label}: centre ({centre[0]:.2f}, {centre[1]:.2f}) from {estimate.source}, "
            f"{np.linalg.norm(centre - buoy.xy):.2f} m from the drone, "
            f"{np.linalg.norm(centre - estimate.predicted):.2f} m from predicted, search radius "
            f"{estimate.radius:.2f}; R {radius:.2f} ({why}); "
            f"{'clockwise' if buoy.sigma > 0 else 'counterclockwise'} from {math.degrees(phi0):.0f} deg")
        with self._lock:
            self._circle = points

        skips = 0
        for k in range(len(points)):
            for retry in range(1 + max(0, int(self.point_retries))):
                if k > 0 or retry > 0:
                    estimate = self._estimate(buoy)
                    moved = float(np.linalg.norm(estimate.xy - centre))
                    if may_restart and moved > self.estimate_restart_threshold:
                        # Points driven so far were around the wrong place: the rest would not enclose the buoy
                        self.get_logger().warning(f"{label}: centre moved {moved:.2f} m ({estimate.source}) at "
                                                  f"point {k}: starting this buoy over")
                        raise Restart()
                    if moved > self.estimate_move_threshold or retry > 0:
                        if moved > 1e-6:
                            self.get_logger().info(
                                f"{label}: centre moved {moved:.2f} m "
                                f"({estimate.source}), points re-placed")
                        centre, points = estimate.xy, place(estimate.xy)
                        buoy.estimate = estimate
                        with self._lock:
                            self._circle = points
                xy, yaw = points[k]
                self._feedback(goal_handle, "TRANSIT" if k == 0 else "CIRCLE", buoy, position, total, k,
                               len(points) - 1, estimate.source)
                if self._drive(goal_handle, xy, yaw, f"{label} point {k}/{len(points) - 1}"):
                    skips = 0
                    break
            else:
                if k == 0:
                    self.get_logger().warning(f"{label}: start point unreachable")
                    return False
                skips += 1
                self.get_logger().warning(f"{label}: point {k} skipped ({skips} in a row)")
                if skips >= self.max_consecutive_skips:
                    return False
        return True

    @staticmethod
    def _done_word(buoy: Buoy) -> str:
        """What was done to a buoy that is dealt with: reds and greens are passed, entry and exit circled."""
        return "passed" if buoy.state in (geo.RED, geo.GREEN) else "circled"

    def _estimate(self, buoy: Buoy) -> ba.Estimate:
        """Where the buoy is now. A costmap that misses the blob it was last seen as does not move it."""
        with self._lock:
            claimed = [b.estimate.xy for b in self._buoys if b.status == "circled" and b.estimate is not None]
            fresh = self._aligner.estimate(buoy.index, claimed)
        held = buoy.estimate
        if (fresh.source == ba.PREDICTED and held is not None and held.source == ba.BLOB
                and np.linalg.norm(fresh.predicted - held.xy) <= fresh.radius):
            return held
        return fresh

    def _neighbours(self, buoy: Buoy) -> list:
        """Every other buoy where we best know it: the centre it was circled around, else its predicted position."""
        with self._lock:
            t = self._aligner.t
            return [b.estimate.xy if b.status == "circled" and b.estimate is not None else b.xy + t
                    for b in self._buoys if b.index != buoy.index]

    def _feedback(self, goal_handle, phase, buoy: Buoy, position, total, point=0, point_total=0, source=""):
        with self._lock:
            matched = self._aligner.matched if self._aligner is not None else 0
        goal_handle.publish_feedback(SafePassage.Feedback(
            phase=phase, buoy_index=position, buoy_total=total, buoy_state=buoy.state, point=point,
            point_total=point_total, estimate_source=source, shift_inliers=matched))

    # Driving

    def _drive(self, goal_handle, xy, yaw: float, label: str, sample_radius: Optional[float] = None,
               tolerance: Optional[float] = None, timeout: Optional[float] = None) -> bool:
        """TrackPath to xy, ending at yaw (ENU). Raises Assistance or Canceled."""
        self._check(goal_handle)
        with self._lock:
            self._target = np.asarray(xy)
        goal = TrackPath.Goal()
        goal.goal_type = TrackPath.Goal.GOAL_POSE
        goal.goal_pose = PoseStamped()
        goal.goal_pose.header.frame_id = self.global_frame_id
        goal.goal_pose.header.stamp = Time()  # latest TF when the tracker resolves it
        goal.goal_pose.pose.position.x, goal.goal_pose.pose.position.y = float(xy[0]), float(xy[1])
        goal.goal_pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal.goal_pose.pose.orientation.w = math.cos(yaw / 2.0)
        goal.anchor_frame_name = self.base_link_frame_id
        goal.last_goal_anchor_only = True
        goal.max_replans = int(self.max_replans)
        goal.sample_radius = float(self.goal_sample_radius if sample_radius is None else sample_radius)
        goal.specified_heading = False
        goal.ignore_depth = True
        goal.forward_tolerance = goal.sidemove_tolerance = float(self.goal_tolerance if tolerance is None
                                                                 else tolerance)
        goal.heading_tolerance = math.pi  # heading unspecified
        goal.depth_tolerance = 1.0  # must stay positive, controls compares strictly
        goal.stabilize_duration = int(self.goal_stabilize)
        self.get_logger().info(f"{label}: TrackPath to ({xy[0]:.2f}, {xy[1]:.2f}) heading {math.degrees(yaw):.0f}")
        return self._run_action(goal_handle, self.track_path, self.track_path_name, goal,
                                self.goal_timeout if timeout is None else timeout,
                                lambda r: r.status == TrackPath.Result.STATUS_SUCCESS)

    # Task 4 Core

    def _on_assistance(self, msg: AssistanceRequest):
        """Accept a request once per seq, only for this vehicle type and while a goal answers them."""
        with self._lock:
            where = f"({msg.position.latitude:.7f}, {msg.position.longitude:.7f})"
            if not self._respond:
                if msg.vehicle_type != self.vehicle_type or not self.hold_idle_requests:
                    self.get_logger().info(f"Assistance request {msg.seq} to {where} for '{msg.vehicle_type}' "
                                           "dropped: no Safe Passage goal answering requests is running")
                    return
                if self._held is not None and self._held.seq == msg.seq:
                    return
                replaced = f", replacing request {self._held.seq}" if self._held is not None else ""
                self._held = Incident(msg.seq, msg.position.latitude, msg.position.longitude, time.monotonic())
                self.get_logger().warning(f"Assistance request {msg.seq} to {where} held{replaced}: no Safe Passage "
                                          "goal answering requests is running. The next one answers it first, "
                                          "unless clear_assistance is called before")
                return
            if msg.seq in self._seen:  # each request once per goal, however often it is delivered
                return
            self._seen.add(msg.seq)
            if msg.vehicle_type != self.vehicle_type:
                self.get_logger().info(f"Assistance request {msg.seq} to {where} is for '{msg.vehicle_type}', not "
                                       f"'{self.vehicle_type}': ignored")
                return
            if self._pending is not None:
                self._end(self._pending, "superseded")
            self._pending = Incident(msg.seq, msg.position.latitude, msg.position.longitude, time.monotonic())
            self._incidents.append(self._pending)
            self.get_logger().warning(f"Assistance request {msg.seq} to {where}: accepted, interrupting")

    def _on_clear_assistance(self, request, response):
        """Drop a held request (a new run starts). A request a running goal is answering is not touched."""
        with self._lock:
            held, self._held = self._held, None
            busy = self._pending or self._incident
        response.success = True
        response.message = (f"dropped held assistance request {held.seq}" if held is not None
                            else "no assistance request was held")
        if busy is not None:
            response.message += f"; request {busy.seq} is being answered by the running goal and goes on"
        self.get_logger().info(f"clear_assistance: {response.message}")
        return response

    def _on_confirm(self, msg: ReadinessConfirm):
        if msg.vehicle_id != self.vehicle_id:
            self.get_logger().info(f"Readiness confirm {msg.seq} is for '{msg.vehicle_id}', not '{self.vehicle_id}': "
                                   "ignored", throttle_duration_sec=5.0)
            return
        with self._lock:
            new = msg.report_seq not in self._confirms
            if new:  # kept with its arrival time: it may arrive before we know our report's seq
                self._confirms[msg.report_seq] = time.monotonic()
        if new:
            self.get_logger().info(f"Readiness confirm {msg.seq} for report {msg.report_seq} received")

    def _end(self, incident: Incident, state: str):
        """Final state of a request. Call with _lock held."""
        incident.state, incident.ended = state, time.monotonic()

    def _answer_assistance(self, goal_handle, total: int):
        """Answer every waiting request, newest last, then switch the task back. Raises Canceled."""
        while True:
            answered = False
            while True:
                with self._lock:
                    incident, self._pending = self._pending, None
                    self._incident = incident
                if incident is None:
                    break
                answered = True
                try:
                    outcome = self._respond_to(goal_handle, incident, total)
                except Assistance:  # a newer request arrived while this one was being answered
                    outcome = "superseded"
                with self._lock:
                    self._end(incident, outcome)
                    self._incident = None
                    self._target = None
                took = incident.ended - incident.received
                if outcome == "confirmed":
                    self.get_logger().info(f"Assistance request {incident.seq}: confirmed after {took:.0f} s")
                else:
                    self.get_logger().warning(f"Assistance request {incident.seq}: {outcome} after {took:.0f} s")
            if not answered:
                return
            # Waited for, so it cannot land after a task the mission tree sets once this goal returns
            self._send_task(self.resume_task, goal_handle)
            if self._pending is None:  # else one more came in during that switch: answer it before going on
                self.get_logger().info("Back to Safe Passage: the interrupted step starts over")
                return

    def _respond_to(self, goal_handle, incident: Incident, total: int) -> str:
        """Ack, go there, report ready, wait for the confirm. Returns how it ended. Raises Assistance or Canceled."""
        seq = incident.seq

        def state(name: str, phase: str):
            with self._lock:
                incident.state = name
            goal_handle.publish_feedback(SafePassage.Feedback(phase=phase, buoy_total=total))

        state("transit", "INCIDENT_TRANSIT")
        self._send_incident_ack(seq)          # both in the background: the boat moves at once
        self._send_task(self.incident_task)
        station = self._from_ll(incident.latitude, incident.longitude)
        if station is None:
            self.get_logger().error(f"Assistance request {seq}: fromLL failed for its position")
            return "no_position"
        tries = 1 + max(0, int(self.incident_goal_retries))
        for attempt in range(1, tries + 1):
            boat = self._boat_xy()
            yaw = math.atan2(station[1] - boat[1], station[0] - boat[0]) if boat is not None else 0.0
            if self._drive(goal_handle, station, yaw, f"Assistance request {seq} transit {attempt}/{tries}",
                           self.incident_sample_radius, self.incident_tolerance, self.incident_goal_timeout):
                break
        else:
            self.get_logger().error(f"Assistance request {seq}: position ({station[0]:.2f}, {station[1]:.2f}) not "
                                    "reached, no ReadinessReport sent")
            return "unreachable"

        state("on_station", "INCIDENT_ON_STATION")
        with self._lock:
            incident.on_station = time.monotonic()
        report_seq, unsure = self._send_readiness(goal_handle, seq)
        if report_seq is None:
            self.get_logger().error(f"Assistance request {seq}: on station but the ReadinessReport did not go out")
            return "report_failed"
        with self._lock:
            incident.report_seq, incident.report_unsure = report_seq, unsure
        self.get_logger().info(
            f"Assistance request {seq}: on station, ReadinessReport sent as report {report_seq}; holding for its "
            f"ReadinessConfirm, up to {self.readiness_confirm_timeout:.0f} s"
            + (". An earlier try failed and may have reached RoboCommand under another seq: any confirm for "
               f"{self.vehicle_id} from now on counts" if unsure else ""))

        def confirmed():
            self._check(goal_handle)
            with self._lock:
                if report_seq in self._confirms:
                    return True
                # A failed try may have gone out under a seq we never learned, and that is the one confirmed
                return unsure and any(at >= incident.on_station for at in self._confirms.values())

        if self._wait(goal_handle, confirmed, self.readiness_confirm_timeout):
            return "confirmed"
        return "confirm_timeout"

    def _send_incident_ack(self, seq: int):
        goal = SendIncidentAck.Goal()
        goal.report.command_seq = seq
        self._send_with_retries(self.incident_ack, self.incident_ack_name, goal, f"IncidentAck for {seq}",
                                lambda: False, background=True)

    def _send_task(self, task: str, goal_handle=None):
        """Switch the heartbeat task. In the background, or waited for when goal_handle is given (cancel ends the
        wait). A switch gives way as soon as a newer one is asked for, so they cannot land out of order."""
        with self._lock:
            self._task_wanted = task

        def stale():
            return self._task_wanted != task or (goal_handle is not None and goal_handle.is_cancel_requested)

        self._send_with_retries(self.set_task, self.set_task_name, SetTask.Goal(task=task), f"SetTask {task}",
                                stale, background=goal_handle is None)

    def _send_readiness(self, goal_handle, seq: int) -> tuple[Optional[int], bool]:
        """Send the ReadinessReport, retried. Returns (seq of the report that carried it, whether a try failed)."""
        goal = SendReadinessReport.Goal()
        goal.report.command_seq = seq
        unsure = False
        for attempt in range(1, 2 + max(0, int(self.report_retries))):
            sent = {}

            def ok(result):
                sent["seq"] = result.seq
                return not result.error

            if self._run_action(goal_handle, self.readiness, self.readiness_name, goal, self.report_timeout, ok):
                return int(sent["seq"]), unsure
            unsure = True
            self.get_logger().warning(f"ReadinessReport for {seq}: attempt {attempt} failed")
        return None, unsure

    def _send_with_retries(self, client: ActionClient, name: str, goal, what: str, stale: Callable[[], bool],
                           background: bool):
        """Send a report-style goal, retried, every outcome logged. Gives up as soon as stale() holds."""
        last = 1 + max(0, int(self.report_retries))

        def attempt(n: int) -> bool:
            """True when there is nothing left to do: sent, or no longer wanted."""
            if stale():
                self.get_logger().info(f"{what}: dropped, no longer wanted")
                return True
            error = self._send_blocking(client, goal, stale)
            if error is None:
                self.get_logger().info(f"{what}: sent")
                return True
            self.get_logger().warning(f"{what}: attempt {n}/{last} failed ({error})")
            return False

        def run(first: int):
            for n in range(first, last + 1):
                if n > 1:
                    time.sleep(1.0)
                if attempt(n):
                    return
            self.get_logger().error(f"{what}: not sent")

        if background:
            threading.Thread(target=run, args=(1,), name=what, daemon=True).start()
        elif not attempt(1):
            # Waited for one try only, so a dead OCS link cannot hold the boat; the retries go on behind
            threading.Thread(target=run, args=(2,), name=what, daemon=True).start()

    def _send_blocking(self, client: ActionClient, goal, stale: Callable[[], bool] = lambda: False) -> Optional[str]:
        """Send a goal and wait for its result on events (the executor runs the callbacks). None when it went out."""
        if not client.wait_for_server(timeout_sec=self.service_timeout):
            return "server unavailable"
        accepted, done = threading.Event(), threading.Event()
        send = client.send_goal_async(goal)
        send.add_done_callback(lambda _: accepted.set())
        if not accepted.wait(self.service_timeout) or not send.result().accepted:
            return "goal not accepted"
        result = send.result().get_result_async()
        result.add_done_callback(lambda _: done.set())
        deadline = time.monotonic() + self.report_timeout
        while not done.wait(0.1):
            if stale():
                send.result().cancel_goal_async()
                return "no longer wanted"
            if time.monotonic() > deadline:
                return f"no result within {self.report_timeout:.0f} s"
        return result.result().result.error or None

    # Plumbing

    def _check(self, goal_handle):
        if goal_handle.is_cancel_requested:
            raise Canceled()
        if self._pending is not None:
            raise Assistance()

    def _wait(self, goal_handle, done: Callable[[], bool], timeout: float) -> bool:
        """Poll done until it holds or timeout passes. Never spins: the executor runs the callbacks."""
        deadline = time.monotonic() + timeout
        event = threading.Event()
        while not done():
            if goal_handle.is_cancel_requested:
                raise Canceled()
            if time.monotonic() > deadline:
                return False
            event.wait(POLL_S)
        return True

    def _run_action(self, goal_handle, client: ActionClient, name: str, goal, timeout: float, ok: Callable) -> bool:
        if not client.wait_for_server(timeout_sec=self.service_timeout):
            self.get_logger().error(f"{name} unavailable")
            return False
        send = client.send_goal_async(goal)

        def interrupted():
            self._check(goal_handle)
            return False

        try:
            if not self._wait(goal_handle, lambda: send.done() or interrupted(), self.service_timeout) \
                    or not send.result().accepted:
                self.get_logger().error(f"{name} did not accept the goal")
                return False
            child = send.result()
            result = child.get_result_async()
            if not self._wait(goal_handle, lambda: result.done() or interrupted(), timeout):
                self.get_logger().error(f"{name} timed out after {timeout:.0f} s, canceling")
                child.cancel_goal_async()
                return False
        except (Canceled, Assistance):
            if send.done() and send.result().accepted:
                send.result().cancel_goal_async()
            raise
        outcome = result.result().result
        succeeded = ok(outcome)
        if not succeeded:
            detail = f"status {outcome.status}" if hasattr(outcome, "status") else f"error '{outcome.error}'"
            self.get_logger().warning(f"{name} did not succeed ({detail})")
        return succeeded

    def _from_ll(self, latitude: float, longitude: float) -> Optional[np.ndarray]:
        request = FromLL.Request()
        request.ll_point.latitude, request.ll_point.longitude, request.ll_point.altitude = latitude, longitude, 0.0
        response = self._call(self.from_ll, request)
        if response is None:
            return None
        return np.array([response.map_point.x, response.map_point.y])

    def _call(self, client, request):
        """Short service call, waited on without spinning."""
        if not client.wait_for_service(timeout_sec=self.service_timeout):
            self.get_logger().error(f"{client.srv_name} unavailable")
            return None
        future = client.call_async(request)
        event = threading.Event()
        future.add_done_callback(lambda _: event.set())
        if not event.wait(self.service_timeout):
            client.remove_pending_request(future)
            self.get_logger().error(f"{client.srv_name} timed out")
            return None
        return future.result()

    def _boat_xy(self) -> Optional[np.ndarray]:
        try:
            t = self.tf_buffer.lookup_transform(self.global_frame_id, self.base_link_frame_id, RosTime(),
                                                Duration(seconds=0.5)).transform.translation
        except tf2_ros.TransformException as e:
            self.get_logger().error(f"Boat pose lookup failed: {e}")
            return None
        return np.array([t.x, t.y])

    def _on_costmap(self, msg: OccupancyGrid):
        if msg.header.frame_id != self.global_frame_id:
            self.get_logger().warning(f"Costmap in '{msg.header.frame_id}', not '{self.global_frame_id}': ignored",
                                      throttle_duration_sec=5.0)
            return
        self._grid = geo.CostGrid.from_msg(msg)
        self._align_step(self._grid)

    def _snapshot(self) -> geo.CostGrid:
        grid = self._grid
        if grid is None:
            self.get_logger().warning(f"No costmap on {self.costmap_topic}, using an empty one",
                                      throttle_duration_sec=5.0)
            return geo.CostGrid.empty()
        return grid

    def _publish_markers(self):
        with self._lock:
            aligner, buoys, circle, target = self._aligner, list(self._buoys), list(self._circle), self._target
            if aligner is None:
                return
            t, locked = aligner.t.copy(), aligner.locked
            small, large = aligner.buoy_blobs.copy(), aligner.structures.copy()
        stamp = self.get_clock().now().to_msg()
        markers = MarkerArray()
        markers.markers.append(Marker(action=Marker.DELETEALL))

        def marker(ns, mid, kind, xy, scale, rgb, alpha=1.0, z=0.0):
            m = Marker()
            m.header.frame_id, m.header.stamp = self.global_frame_id, stamp
            m.ns, m.id, m.type, m.action = ns, mid, kind, Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = float(xy[0]), float(xy[1]), z
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = float(scale)
            m.color = ColorRGBA(r=rgb[0], g=rgb[1], b=rgb[2], a=alpha)
            return m

        for b in buoys:
            rgb = STATE_COLOURS.get(b.state, (1.0, 1.0, 1.0))
            markers.markers.append(marker("drone", b.index, Marker.SPHERE, b.xy, 0.4, rgb, 0.4))
            markers.markers.append(marker("predicted", b.index, Marker.SPHERE, b.xy + t, 0.25, rgb,
                                          1.0 if locked else 0.4))
            if b.estimate is not None:
                markers.markers.append(marker("estimates", b.index, Marker.CYLINDER, b.estimate.xy, 0.5, rgb, 0.9))
        for k, xy in enumerate(small):
            markers.markers.append(marker("buoy blobs", k, Marker.CUBE, xy, 0.3, (0.6, 1.0, 0.2), 0.8, z=0.3))
        for k, blob in enumerate(large):
            markers.markers.append(marker("structures", k, Marker.CYLINDER, blob[:2], max(blob[3], 0.3),
                                          (1.0, 0.0, 0.0), 0.3))
        if circle:
            strip = marker("circle", 0, Marker.LINE_STRIP, (0.0, 0.0), 0.05, (1.0, 1.0, 1.0))
            strip.points = [Point(x=float(p[0]), y=float(p[1])) for p, _ in circle]
            markers.markers.append(strip)
        if target is not None:
            markers.markers.append(marker("target", 0, Marker.SPHERE, target, 0.35, (1.0, 0.0, 1.0)))
        self.marker_pub.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = SafePassageNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        # Under ros2 launch a Ctrl-C arrives twice (terminal and launch): the second can land in here
        try:
            executor.shutdown(timeout_sec=1.0)
            node.destroy_node()
        except (Exception, KeyboardInterrupt):
            pass
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
