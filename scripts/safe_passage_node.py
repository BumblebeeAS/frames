#!/usr/bin/env python3
"""RobotX 2026 Task 1 (Safe Passage) on the ASV, from the drone's classified buoys.

Serves bb_robotx_msgs/SafePassage. A goal carries the drone's buoys (lat/lon + state). They are converted once to
the global frame with fromLL; buoys in a state other than entry, exit, red, green or black (unknown, say) are
dropped on arrival. The run is one route (frames.utils.safe_route), driven as stop-and-go TrackPath goals:

  ALIGN    hold until the drone's layout is lined up with the costmap (frames.utils.buoy_align) or align_timeout
           passes: the drone's positions can be metres off as one layout. Without a lock the shift is zero.
  PLAN     hold while the planner routes the whole run on the buoys where they are now believed to be: each drone
           buoy on its costmap blob, else at its shifted drone position; buoy-sized blobs the drone did not report
           and larger structures are kept clear of. No full route from the lattice: the sampling planner
           (frames.utils.safe_route_rrt) is asked and the better route kept. No route: planned again every
           plan_retry_period, with no limit, by the sampler, and by the lattice too every lattice_every attempts.
  ENTRY    the entry loop, clockwise, from a point the planner chose back to that point
  TRANSIT  from there to the exit circle, never a red to port or a green to starboard, passing at least one red to
           starboard and one green to port where the field allows; one goal per corner
  EXIT     the exit circle, counterclockwise

The plan is made once and driven as it is. What the goal has done (entry loop closed, a red passed, a green passed)
outlives every plan: passes are followed on the boat's own pose between the entry loop closing and the exit circle
being joined. A goal that fails point_retries + 1 times sends the node back to PLAN, from where the boat is, for
what is still to do; without an entry buoy the first plan starts from the boat as well.

Task 4 Core: with respond_to_assistance, an AssistanceRequest for this vehicle type interrupts whatever the goal is
doing. The node sends the IncidentAck and switches the heartbeat task, drives to the requested position, sends the
ReadinessReport once there, and holds (the boat station-keeps without a goal) until the ReadinessConfirm naming that
report arrives or readiness_confirm_timeout passes. Then it drives back to where it was interrupted
(incident_return), switches the task back and goes on: ALIGN resumes; anything later goes back to PLAN, with the
entry loop if it had not closed and without it if it had. Requests are taken once per seq; a newer request replaces
the one in hand. A request that arrives while no goal answers them is held and answered first by the next goal that
does, unless that goal starts more than assistance_max_age after the request was received: then it is discarded. The clear_assistance service (std_srvs/Trigger) drops a held request: the mission tree
calls it at run start, so nothing from before the run is acted on. The hold is in memory only, so a relaunched node
starts without one.

Conventions: everything is planned in global_frame_id (asv5/map, ENU). TrackPath goals are GOAL_POSE there with
specified_heading false and the heading of the next leg, so the boat ends each goal facing the next.
"""

import math
import threading
import time
from dataclasses import dataclass, field, replace
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
from scipy.spatial import cKDTree
from std_msgs.msg import ColorRGBA
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray

from frames.utils import buoy_align as ba
from frames.utils import safe_passage_geometry as geo
from frames.utils import safe_route as sr
from frames.utils import safe_route_rrt as srr

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
TRACK_PERIOD_S = 0.1  # the boat's pose through the pass tracker: 4 cm apart at 0.44 m/s
KNOWN_STATES = (geo.ENTRY, geo.EXIT, geo.RED, geo.GREEN, geo.BLACK)

STATE_COLOURS = {
    geo.ENTRY: (0.1, 0.3, 1.0), geo.EXIT: (0.0, 0.9, 0.9), geo.RED: (1.0, 0.1, 0.1),
    geo.GREEN: (0.1, 0.9, 0.1), geo.BLACK: (0.3, 0.3, 0.3),
}


class Canceled(Exception):
    """The SafePassage goal was canceled."""


class Assistance(Exception):
    """An AssistanceRequest is waiting: drop what is being driven and answer it."""


@dataclass
class Incident:
    """One Task 4 Core AssistanceRequest and how far its answer got."""
    seq: int                            # RxCommand seq, quoted by the IncidentAck and the ReadinessReport
    latitude: float
    longitude: float
    received: float                     # monotonic
    state: str = "received"             # received | transit | on_station | then how it ended:
    #                                     confirmed | confirm_timeout | unreachable | report_failed | no_position |
    #                                     superseded | canceled | expired
    report_seq: Optional[int] = None    # seq of the report that carried our ReadinessReport
    report_unsure: bool = False         # a try at that report failed: it may still have reached RoboCommand
    on_station: Optional[float] = None  # monotonic, when the boat got there
    ended: Optional[float] = None


@dataclass
class Buoy:
    index: int
    state: str
    xy: np.ndarray                      # C_i
    estimate: Optional[ba.Estimate] = None  # where the last plan had it


@dataclass
class Progress:
    """What a goal has done so far. It outlives every plan: a new plan is only for what is still to do."""
    entry_done: bool = False            # the entry loop closed
    exit_done: bool = False             # the exit circle was driven to its end
    red: bool = False                   # a red was passed to starboard on the judged path
    green: bool = False                 # a green was passed to port there
    plans: int = 0                      # planner runs
    failed_goals: int = 0               # goals that failed every try and sent the node back to PLAN
    passes: list = field(default_factory=list)  # credits and violations seen on the boat's pose, in order


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
        self.align_timeout = param("align_timeout", 60.0)

        # Route (frames/utils/safe_route.py): the circles as the planner lays them out, the rest at its defaults
        self.route_params = sr.RouteParams(
            circle_radius=float(param("circle_radius", 2.5)),
            circle_radius_min=float(param("circle_radius_min", 1.6)),
            neighbour_clearance=float(param("neighbour_clearance", 1.25)),
            circle_points=int(param("circle_points", 6)),
        )
        self.plan_retry_period = param("plan_retry_period", 2.0)
        # The sampling planner (frames/utils/safe_route_rrt.py), asked when the lattice has no full route
        self.fallback_planner = param("fallback_planner", True)
        self.fallback_budget = param("fallback_budget", 10.0)  # s of search per ask; its checks come on top
        self.lattice_every = param("lattice_every", 5)         # after a miss the lattice sits out the attempts between
        # What the planner is told about the field
        self.blob_sigma = param("blob_sigma", 0.3)            # m, position error of a buoy on its costmap blob
        self.predicted_sigma = param("predicted_sigma", 1.0)  # ... of one at its shifted drone position
        self.unreported_margin = param("unreported_margin", 4.0)  # m around the drone's buoys: blobs counted
        self.blob_cell_radius = param("blob_cell_radius", 0.9)    # m, lethal cells this close to a buoy blob are it
        self.obstacle_range = param("obstacle_range", 14.0)       # m, lethal cells further from every buoy are left out

        # TrackPath
        self.goal_tolerance = param("goal_tolerance", 1.0)
        self.goal_stabilize = param("goal_stabilize", 0)
        self.goal_sample_radius = param("goal_sample_radius", 0.5)
        self.max_replans = param("max_replans", 5)
        self.goal_timeout = param("goal_timeout", 60.0)
        self.goal_speed = param("goal_speed", 0.3)      # m/s, a goal's timeout grows by its distance over this

        # Retries
        self.point_retries = param("point_retries", 2)

        # Task 4 Core
        self.hold_idle_requests = param("hold_idle_requests", True)
        self.assistance_max_age = param("assistance_max_age", 30.0)  # s from receipt to starting on it; 0: no limit
        self.readiness_confirm_timeout = param("readiness_confirm_timeout", 900.0)
        self.incident_goal_retries = param("incident_goal_retries", 2)
        self.incident_sample_radius = param("incident_sample_radius", 0.5)
        self.incident_tolerance = param("incident_tolerance", 1.0)
        self.incident_goal_timeout = param("incident_goal_timeout", 300.0)  # the station can be across the course
        self.incident_return = param("incident_return", True)  # afterwards, back to where the boat was interrupted
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
        self.create_timer(TRACK_PERIOD_S, self._track_passes, callback_group=self.cb_group)

        # Goal state shared with the costmap thread: the aligner, the buoys, the route being driven
        self._lock = threading.Lock()
        self._aligner: Optional[ba.Aligner] = None
        self._buoys: list[Buoy] = []
        self._circle: list = []      # [(xy, yaw)] the goals of the route being driven, for markers
        self._target: Optional[np.ndarray] = None
        self._logged_t = np.zeros(2)
        self._route_planner = ""                      # which planner the route in hand came from
        self._route_buoys: list = []                  # what the route in hand was planned on (safe_route.Buoy)
        self._route_level = 0                         # ... and the planner's margin level that produced it
        self._tracker: Optional[sr.PassTracker] = None  # set while passes count, fed the boat's pose by a timer
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
        if not any(b.state in KNOWN_STATES and b.state != geo.BLACK for b in goal.buoys):
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
                    self._route_buoys, self._tracker = [], None
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
        for i, b in enumerate(request.buoys):
            if b.state not in KNOWN_STATES:
                # An unclassified detection may not be a buoy at all: it is neither planned around nor aligned on
                self.get_logger().info(f"Buoy {i} ({b.state or 'no state'}): not a known state, dropped")
                continue
            xy = self._from_ll(b.position.latitude, b.position.longitude)
            if xy is None:
                self.get_logger().error(f"Buoy {i} ({b.state}): fromLL failed, dropped")
                continue
            buoys.append(Buoy(len(buoys), b.state, xy))
        if not any(b.state != geo.BLACK for b in buoys):
            return None
        aligner = ba.Aligner([b.xy for b in buoys], self.align_params)
        with self._lock:
            self._aligner, self._buoys, self._logged_t, self._solved_grid = aligner, buoys, np.zeros(2), None
        self.get_logger().info(
            "Safe Passage buoys: " + ", ".join(f"{b.index}:{b.state}({b.xy[0]:.1f},{b.xy[1]:.1f})" for b in buoys)
            + f"; tightest spacing {float(aligner.nn.min()):.2f} m; respond_to_assistance {self._respond}")
        total = len(buoys)
        progress = Progress()
        self._answer_assistance(goal_handle, total)  # a request held from before this goal comes first

        # ALIGN once, then PLAN and drive until a plan has been driven to its end. An interrupt (a Task 4 request)
        # or a goal that failed every try leads back to PLAN, which only asks for what is still to do
        aligned, judged = False, False
        while True:
            try:
                if not aligned:
                    self._align(goal_handle, total)
                    aligned = True
                route = self._plan(goal_handle, buoys, progress, total)
                if self._drive_route(goal_handle, route, progress, total, judged):
                    break
                judged = progress.entry_done  # stopped on the judged path: the way on from here counts at once
            except Assistance:
                self._answer_assistance(goal_handle, total, self._boat_xy())
                judged = False                # back from the incident: it counts again from the route's first point
        self._answer_assistance(goal_handle, total)  # a request that came in as the last goal finished

        done = [name for name, flag in (("entry loop", progress.entry_done), ("exit circle", progress.exit_done))
                if flag]
        lines = [f"route {route.status} by the {self._route_planner}"
                 + (f" ({route.message})" if route.message else ""),
                 "driven: " + (", ".join(done) or "no circle"),
                 f"red to starboard {'yes' if progress.red else 'no'}, green to port "
                 f"{'yes' if progress.green else 'no'}",
                 f"{progress.plans} plan(s), {progress.failed_goals} goal(s) failed"]
        lines += [f"pass {p}" for p in progress.passes]
        lines += [f"{b.index}:{b.state} {b.estimate.source} {np.linalg.norm(b.estimate.xy - b.xy):.2f} m from the "
                  "drone" for b in buoys if b.estimate is not None]
        with self._lock:
            lines += [f"assistance request {x.seq} {x.state}"
                      + (f" after {x.ended - x.received:.0f} s" if x.ended is not None else "")
                      for x in self._incidents]
        return True, len(done), progress.failed_goals, "; ".join(lines)

    # Alignment

    def _align(self, goal_handle, total: int):
        """Hold until the drone layout locks onto the costmap or align_timeout passes. Raises Canceled."""
        self._feedback(goal_handle, "ALIGN", total)
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

    # Plan

    def _plan(self, goal_handle, buoys: list, progress: Progress, total: int) -> sr.RoutePlan:
        """PLAN: hold until the planner returns a route, one attempt per plan_retry_period, with no limit: each
        attempt sees the costmap of its moment. Raises Assistance or Canceled."""
        attempt = 0
        while True:
            self._feedback(goal_handle, "PLAN", total)
            inputs, obstacles = self._planner_inputs(buoys, self._snapshot())
            # The lattice gives the same answer on the same field and takes long to say no: after a miss the
            # attempts in between are the sampler's alone, which searches differently every time
            lattice = not self.fallback_planner or attempt % max(1, int(self.lattice_every)) == 0
            # The entry loop is done, or there is no entry buoy: the route starts where the boat is
            from_boat = progress.entry_done or not any(b.state == sr.ENTRY for b in inputs)
            start = self._boat_xy() if from_boat else None
            box = {}
            if start is not None or not from_boat:
                progress.plans += 1
                attempt += 1
                started = time.monotonic()

                def work():
                    ask = dict(start=start, need_red=not progress.red, need_green=not progress.green)
                    rank = {"ok": 2, "ok_partial": 1}
                    planners = [("lattice", lambda: sr.plan_route(inputs, self.route_params, obstacles, **ask))]
                    if self.fallback_planner:
                        planners.append(("sampler", lambda: srr.plan_route(
                            inputs, self.route_params, obstacles, **ask, time_budget=float(self.fallback_budget))))
                    for name, planner in planners[0 if lattice else 1:]:
                        held = box.get("plan")
                        if box.get("dropped") or (held is not None
                                                  and held.status not in ("no_route", "ok_partial")):
                            break             # a full route, or a field the sampler reads the same way
                        try:
                            plan = planner()
                        except Exception as e:  # noqa: BLE001  a planner fault must not end the goal: tried again
                            box["error"] = f"{name}: {e!r}"
                            continue
                        # The sampler's route is taken only when it is the better one: full over partial over none
                        if held is None or rank.get(plan.status, 0) > rank.get(held.status, 0):
                            box["plan"], box["planner"] = plan, name

                worker = threading.Thread(target=work, name="safe_passage_plan", daemon=True)
                worker.start()
                # Not waited for when interrupted: the thread finishes on its own and its route is dropped
                try:
                    self._wait(goal_handle, lambda: self._check(goal_handle) or not worker.is_alive(), math.inf)
                except BaseException:
                    box["dropped"] = True     # interrupted: the planner that is running is the thread's last
                    raise
                if box.get("error"):
                    self.get_logger().error(f"Plan {progress.plans}: planner failed ({box['error']})")
                plan = box.get("plan")
                where = "the entry loop" if start is None else f"({start[0]:.2f}, {start[1]:.2f})"
                if plan is not None:
                    counts = [sum(g[3] == phase for g in plan.goals) for phase in ("entry", "transit", "exit")]
                    self.get_logger().info(
                        f"Plan {progress.plans} from {where}: {plan.status} by the {box['planner']} in "
                        f"{time.monotonic() - started:.1f} s; "
                        f"{len(inputs)} objects ({sum(b.confirmed and b.state != sr.UNKNOWN for b in inputs)} drone "
                        f"buoys on a blob, {sum(b.state == sr.UNKNOWN for b in inputs)} blobs the drone did not "
                        f"report), {len(obstacles)} obstacle cells; goals entry {counts[0]}, transit {counts[1]}, "
                        f"exit {counts[2]}; transit {plan.length:.1f} m, entry R {plan.entry_radius:.2f}, exit R "
                        f"{plan.exit_radius:.2f}; reds to starboard {plan.red_passed}, greens to port "
                        f"{plan.green_passed}; closest approach {plan.clearance:.2f} m"
                        + (f"; {plan.message}" if plan.message else ""))
                    if plan.goals or (start is not None and plan.status == "missing_exit"):  # the latter: all done
                        with self._lock:
                            self._route_buoys = inputs
                            self._route_level = max(plan.level, 0)
                            self._route_planner = box["planner"]
                            self._circle = [(np.array(g[:2]), g[2]) for g in plan.goals]
                        return plan
            self._wait(goal_handle, lambda: self._check(goal_handle) or False, self.plan_retry_period)

    def _planner_inputs(self, buoys: list, grid: geo.CostGrid) -> tuple[list, np.ndarray]:
        """What the planner works on: every drone buoy where it is now believed to be (on its blob: confirmed; else
        at its shifted drone position: unconfirmed, so it is obeyed but earns nothing), the buoy-sized blobs no
        drone buoy sits on (kept clear of, colourless), and the lethal cells of everything larger."""
        with self._lock:
            aligner = self._aligner
            claimed = []
            for b in buoys:
                fresh = aligner.estimate(b.index, claimed)
                held = b.estimate
                # A costmap that misses the blob a buoy was last seen as does not move it
                if (fresh.source == ba.PREDICTED and held is not None and held.source == ba.BLOB
                        and np.linalg.norm(fresh.predicted - held.xy) <= fresh.radius):
                    fresh = held
                b.estimate = fresh
                if fresh.source == ba.BLOB:
                    claimed.append(fresh.xy)
            blobs, join_gap = aligner.buoy_blobs.copy(), aligner.p.join_gap
        out = [sr.Buoy(float(b.estimate.xy[0]), float(b.estimate.xy[1]), b.state, b.estimate.source == ba.BLOB,
                       self.blob_sigma if b.estimate.source == ba.BLOB else self.predicted_sigma,
                       f"{b.index}" + ("" if b.estimate.source == ba.BLOB else "*")) for b in buoys]
        placed = np.array([[b.x, b.y] for b in out])
        low, high = placed.min(axis=0) - self.unreported_margin, placed.max(axis=0) + self.unreported_margin
        for k, blob in enumerate(blobs):
            if (np.linalg.norm(placed - blob, axis=1).min() > join_gap and np.all(blob > low)
                    and np.all(blob < high)):
                out.append(sr.Buoy(float(blob[0]), float(blob[1]), sr.UNKNOWN, True, self.blob_sigma, f"u{k}"))
        cells = grid.lethal_points()
        if len(cells) and len(blobs):  # a buoy-sized blob's own cells are that buoy, not an obstacle
            cells = cells[cKDTree(blobs).query(cells)[0] > self.blob_cell_radius]
        if len(cells):
            cells = cells[cKDTree(placed).query(cells)[0] < self.obstacle_range]
        return out, cells

    # Route

    def _drive_route(self, goal_handle, route: sr.RoutePlan, progress: Progress, total: int,
                     judged: bool = False) -> bool:
        """ENTRY, TRANSIT, EXIT: the plan's goals in order, one TrackPath goal each. False when a goal failed
        every try: the caller plans again from where the boat is. Raises Assistance or Canceled.

        Passes are followed on the boat's own pose from where the entry loop closes to where the boat joins the
        exit circle: what is in hand is not planned for again. A plan without an entry loop is followed from its
        first point, which the boat reaches facing the first leg (as the planner assumed), or, with judged (the
        boat stopped part way along the judged path), from where the boat is.
        """
        goals = route.goals
        phases = [g[3] for g in goals]
        last = {phase: max((i for i, p in enumerate(phases) if p == phase), default=-1)
                for phase in ("entry", "transit", "exit")}
        circled = {"entry": sr.ENTRY, "exit": sr.EXIT}
        tries = 1 + max(0, int(self.point_retries))
        if last["entry"] < 0 and judged:
            self._track(True)
        try:
            for i, (x, y, yaw, phase) in enumerate(goals):
                k, of = phases[:i + 1].count(phase), phases.count(phase)
                self._feedback(goal_handle, phase.upper(), total, k, of, circled.get(phase, ""))
                boat = self._boat_xy()
                timeout = self.goal_timeout + (float(np.hypot(x - boat[0], y - boat[1])) / self.goal_speed
                                               if boat is not None else 0.0)
                for attempt in range(1, tries + 1):
                    if self._drive(goal_handle, (x, y), yaw, f"{phase.upper()} goal {k}/{of} try {attempt}/{tries}",
                                   timeout=timeout):
                        break
                else:
                    self.get_logger().warning(f"{phase.upper()} goal {k}/{of} failed {tries} times: planning again "
                                              "from where the boat is")
                    progress.failed_goals += 1
                    return False
                if i == last["entry"]:
                    progress.entry_done = True
                    self._track(True)
                    self.get_logger().info("Entry loop closed: passes count from here")
                elif i == 0 and last["entry"] < 0:
                    self._track(True)
                if i == last["transit"] and last["exit"] >= 0:
                    self._track(False, progress)
                    self.get_logger().info("Exit circle joined: passes no longer count")
            progress.exit_done = last["exit"] >= 0
            return True
        finally:
            self._track(False, progress)

    def _track(self, on: bool, progress: Optional[Progress] = None):
        """Start following passes on the boat's pose, or stop and keep what was earned."""
        with self._lock:
            tracker, self._tracker = self._tracker, None
            if on:
                # Judged with the margins the route was planned with: stricter ones would call out what it allowed
                clearance, shield, credit, near = self.route_params.levels[self._route_level]
                self._tracker = tracker or sr.PassTracker(self._route_buoys, replace(
                    self.route_params, clearance=clearance, shield_margin=shield, credit_margin=credit,
                    near_range=near))
                return
        if tracker is None or progress is None:
            return
        result = tracker.result()
        progress.red |= bool(result.red_credits)
        progress.green |= bool(result.green_credits)
        progress.passes += [str(e) for e in result.events if e.verdict in ("credit", "violation")]

    def _track_passes(self):
        """Timer: one pose through the pass tracker while a judged stretch is being driven."""
        if self._tracker is None:
            return
        pose = self._boat_pose(quiet=True)
        if pose is None:
            return
        with self._lock:
            events = self._tracker.update(pose) if self._tracker is not None else []
        for e in events:
            if e.verdict == "violation":
                self.get_logger().error(f"Pass: {e}")
            elif e.verdict == "credit":
                self.get_logger().info(f"Pass: {e}")

    def _feedback(self, goal_handle, phase: str, total: int, point: int = 0, point_total: int = 0, state: str = ""):
        with self._lock:
            matched = self._aligner.matched if self._aligner is not None else 0
            source = next((b.estimate.source for b in self._buoys if b.state == state and b.estimate is not None), "")
        goal_handle.publish_feedback(SafePassage.Feedback(
            phase=phase, buoy_total=total, buoy_state=state, point=point, point_total=point_total,
            estimate_source=source, shift_inliers=matched))


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

    def _answer_assistance(self, goal_handle, total: int, back=None):
        """Answer every waiting request, newest last, go back to `back` (where the boat was interrupted) while the
        task is still the incident's, then switch the task back. Raises Canceled."""
        while True:
            answered = False
            while True:
                with self._lock:
                    incident, self._pending = self._pending, None
                    self._incident = incident
                if incident is None:
                    break
                age = time.monotonic() - incident.received
                if self.assistance_max_age > 0.0 and age > self.assistance_max_age:
                    with self._lock:  # too old to start on: no ack, no task switch, the boat stays on its course
                        self._end(incident, "expired")
                        self._incident = None
                    self.get_logger().warning(f"Assistance request {incident.seq}: received {age:.0f} s ago, more "
                                              f"than {self.assistance_max_age:.0f} s: discarded")
                    continue
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
            if back is not None and self.incident_return:
                try:
                    self._return(goal_handle, back, total)
                except Assistance:  # one more came in on the way back: answer it, then come back again
                    continue
            # Waited for, so it cannot land after a task the mission tree sets once this goal returns
            self._send_task(self.resume_task, goal_handle)
            if self._pending is None:  # else one more came in during that switch: answer it before going on
                self.get_logger().info("Back to Safe Passage")
                return

    def _return(self, goal_handle, back, total: int):
        """Back to where the boat was when the request interrupted it: the plan made there is the one that was
        checked to work, and the way back is not Safe Passage's to be judged on. Raises Assistance or Canceled."""
        goal_handle.publish_feedback(SafePassage.Feedback(phase="INCIDENT_RETURN", buoy_total=total))
        tries = 1 + max(0, int(self.incident_goal_retries))
        for attempt in range(1, tries + 1):
            boat = self._boat_xy()
            yaw = math.atan2(back[1] - boat[1], back[0] - boat[0]) if boat is not None else 0.0
            if self._drive(goal_handle, back, yaw, f"Return to ({back[0]:.2f}, {back[1]:.2f}) {attempt}/{tries}",
                           self.incident_sample_radius, self.incident_tolerance, self.incident_goal_timeout):
                return
        self.get_logger().warning("Could not get back to where the boat was interrupted: going on from here")

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
            def cancel(sent):  # at once when the goal is already accepted, else as soon as it is
                if sent.result() is not None and sent.result().accepted:
                    sent.result().cancel_goal_async()

            send.add_done_callback(cancel)
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

    def _boat_pose(self, quiet: bool = False) -> Optional[tuple]:
        """(x, y, yaw) of the boat in the global frame."""
        try:
            t = self.tf_buffer.lookup_transform(self.global_frame_id, self.base_link_frame_id, RosTime(),
                                                Duration(seconds=0.0 if quiet else 0.5)).transform
        except tf2_ros.TransformException as e:
            if not quiet:
                self.get_logger().error(f"Boat pose lookup failed: {e}")
            return None
        q = t.rotation
        return (t.translation.x, t.translation.y,
                math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z)))

    def _boat_xy(self) -> Optional[np.ndarray]:
        pose = self._boat_pose()
        return None if pose is None else np.array(pose[:2])

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
