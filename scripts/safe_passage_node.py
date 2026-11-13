#!/usr/bin/env python3
"""RobotX 2026 Task 1 (Safe Passage) on the ASV, from the drone's classified buoys.

Serves bb_robotx_msgs/SafePassage. A goal carries the drone's buoys (lat/lon + state). They are converted once to
the global frame with fromLL, ordered (entry, reds and greens along the course, exit; blacks are never circled), and
circled one at a time with stop-and-go TrackPath goals:

  TRANSIT  TrackPath to p0, where the boat joins the circle tangentially in the buoy's direction
  CIRCLE   TrackPath to p1 .. pN+overshoot, N points per 360 deg; each goal waits for the previous one's result

Entry and red are circled clockwise (red stays on starboard), green and exit counterclockwise.

The drone's positions can be metres off as one layout, so before driving the node lines the layout up with the
costmap (frames.utils.buoy_align): ALIGN holds until the shift locks or align_timeout passes, and the buoys are
ordered only then. Each circle is centred on the buoy estimate E_i, read again before every goal: the small
costmap blob nearest the shifted drone position, else that position. Without a lock the shift is zero.

Retries: a failed point is re-placed and tried again point_retries times, then skipped; max_consecutive_skips in a
row, or a start point that cannot be reached, fails the attempt. A buoy gets buoy_attempts attempts. Failed reds and
greens are tried once more before the exit. A buoy that still fails is logged and skipped.

Task 4: with pause_on_incident, a set incident flag cancels the current goal and holds; once it clears, the current
buoy starts over from TRANSIT. A goal that fails without us canceling waits preempt_grace for the flag, since the
incident handler may preempt us before its flag arrives.

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
from bb_robotx_msgs.action import SafePassage
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
from std_msgs.msg import Bool, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from frames.utils import buoy_align as ba
from frames.utils import safe_passage_geometry as geo

# nav2 costmaps publish reliable transient local; the incident flag is latched the same way
LATCHED_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST, depth=1,
    reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL,
)
POLL_S = 0.05
MARKER_PERIOD_S = 0.5

STATE_COLOURS = {
    geo.ENTRY: (0.1, 0.3, 1.0), geo.EXIT: (0.0, 0.9, 0.9), geo.RED: (1.0, 0.1, 0.1),
    geo.GREEN: (0.1, 0.9, 0.1), geo.BLACK: (0.3, 0.3, 0.3),
}


class Canceled(Exception):
    """The SafePassage goal was canceled."""


class Paused(Exception):
    """The Task 4 incident flag is set: stop driving and hold."""


@dataclass
class Buoy:
    index: int
    state: str
    xy: np.ndarray                      # C_i
    sigma: int = 0                      # +1 clockwise, -1 counterclockwise, 0 never circled
    status: str = "pending"             # pending | circled | failed
    attempts: int = 0
    estimate: Optional[ba.Estimate] = None  # the one the last circle used
    radius: float = 0.0


class SafePassageNode(Node):
    def __init__(self):
        super().__init__("safe_passage_node")
        param = lambda name, default: self.declare_parameter(name, default).value  # noqa: E731

        # Wiring
        self.costmap_topic = param("costmap_topic", "/asv5/global_costmap/costmap")
        self.incident_topic = param("incident_topic", "/asv5/task4/incident_active")
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
        self.circle_radius = param("circle_radius", 2.5)
        self.circle_radius_min = param("circle_radius_min", 1.6)
        self.circle_radius_step = param("circle_radius_step", 0.25)
        self.neighbour_clearance = param("neighbour_clearance", 1.25)
        self.circle_points = param("circle_points", 8)
        self.circle_overshoot_points = param("circle_overshoot_points", 1)
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

        # Task 4
        self.preempt_grace = param("preempt_grace", 1.0)
        self.resume_settle = param("resume_settle", 1.0)

        if self.goal_sample_radius > 0.0 and self.goal_sample_radius < 0.25:
            self.get_logger().warning("goal_sample_radius under the tracker's sample_spacing (0.25 m) samples nothing")

        self.cb_group = ReentrantCallbackGroup()
        self.costmap_group = MutuallyExclusiveCallbackGroup()  # a solve takes a while: never two at once
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self._grid: Optional[geo.CostGrid] = None
        self.create_subscription(OccupancyGrid, self.costmap_topic, self._on_costmap, LATCHED_QOS,
                                 callback_group=self.costmap_group)
        self._incident = False
        self.create_subscription(Bool, self.incident_topic, self._on_incident, LATCHED_QOS,
                                 callback_group=self.cb_group)
        self.marker_pub = self.create_publisher(MarkerArray, param("marker_topic", "/asv5/safe_passage/markers"), 1)
        self.create_timer(MARKER_PERIOD_S, self._publish_markers, callback_group=self.cb_group)

        # Goal state shared with the costmap thread: the aligner, the buoys, the circle being driven
        self._lock = threading.Lock()
        self._aligner: Optional[ba.Aligner] = None
        self._buoys: list[Buoy] = []
        self._circle: list = []      # [(xy, yaw)] being driven, for markers
        self._target: Optional[np.ndarray] = None
        self._logged_t = np.zeros(2)
        self._pause_enabled = False

        self.track_path_name = param("track_path_action", "/asv5/track_path")
        self.track_path = ActionClient(self, TrackPath, self.track_path_name, callback_group=self.cb_group)
        self.from_ll = self.create_client(FromLL, param("from_ll_service", "/asv5/fromLL"),
                                          callback_group=self.cb_group)

        self._busy = threading.Lock()
        self.server = ActionServer(
            self, SafePassage, param("action_name", "/asv5/safe_passage"),
            execute_callback=self._execute, goal_callback=self._on_goal,
            cancel_callback=lambda _: CancelResponse.ACCEPT, callback_group=self.cb_group,
        )
        self.get_logger().info(f"Safe Passage ready: costmap {self.costmap_topic}, incidents on {self.incident_topic}")

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
        self._pause_enabled = bool(request.pause_on_incident)
        buoys = []
        for i, b in enumerate(request.buoys):
            xy = self._from_ll(b.position.latitude, b.position.longitude)
            if xy is None:
                self.get_logger().error(f"Buoy {i} ({b.state}): fromLL failed, dropped")
                continue
            buoys.append(Buoy(len(buoys), b.state, xy, geo.SIGMA.get(b.state, 0)))
        if not any(b.sigma for b in buoys):
            return None
        aligner = ba.Aligner([b.xy for b in buoys], self.align_params)
        with self._lock:
            self._aligner, self._buoys, self._logged_t = aligner, buoys, np.zeros(2)
        self.get_logger().info(
            "Safe Passage buoys: " + ", ".join(f"{b.index}:{b.state}({b.xy[0]:.1f},{b.xy[1]:.1f})" for b in buoys)
            + f"; tightest spacing {float(aligner.nn.min()):.2f} m; pause_on_incident {self._pause_enabled}")
        self._align(goal_handle, len(buoys))

        # Ordered where the buoys are now believed to be: a shift of metres changes which is nearest
        boat = self._boat_xy()
        with self._lock:
            placed = aligner.predicted()
        order = geo.order_buoys(list(placed), [b.state for b in buoys],
                                boat if boat is not None else placed.mean(axis=0))
        self.get_logger().info(f"Order {order}")

        total = len(order)
        exit_ = [i for i in order if buoys[i].state == geo.EXIT]
        for position, i in enumerate(i for i in order if buoys[i].state != geo.EXIT):
            self._do_buoy(goal_handle, buoys[i], position, total)
        if self.requeue_failed:
            for i in order:
                b = buoys[i]
                if b.status == "failed" and b.state in (geo.RED, geo.GREEN):
                    self.get_logger().info(f"Buoy {i} ({b.state}): requeued for one more attempt before the exit")
                    b.attempts = max(0, int(self.buoy_attempts) - 1)
                    self._do_buoy(goal_handle, b, order.index(i), total)
        for i in exit_:
            self._do_buoy(goal_handle, buoys[i], order.index(i), total)

        circled = [b for b in buoys if b.status == "circled"]
        failed = [b for b in buoys if b.status == "failed"]
        lines = []
        for i in order:
            b = buoys[i]
            e = b.estimate
            detail = (f", {e.source} {np.linalg.norm(e.xy - b.xy):.2f} m from the drone, R {b.radius:.2f}"
                      if e is not None else "")
            lines.append(f"{i}:{b.state} {b.status} ({b.attempts} attempt(s){detail})")
        return not failed, len(circled), len(failed), "; ".join(lines)

    # Alignment

    def _align(self, goal_handle, total: int):
        """Hold until the drone layout locks onto the costmap or align_timeout passes. Raises Canceled."""
        self._align_step(self._snapshot())  # the costmap already here; later ones arrive through _on_costmap
        start = time.monotonic()

        def locked():
            with self._lock:
                matched, done = self._aligner.matched, self._aligner.locked
            goal_handle.publish_feedback(SafePassage.Feedback(phase="ALIGN", buoy_total=total, shift_inliers=matched))
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
        if aligner is None:
            return
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

    # One buoy

    def _do_buoy(self, goal_handle, buoy: Buoy, position: int, total: int):
        while buoy.attempts < self.buoy_attempts:
            buoy.attempts += 1
            label = f"Buoy {buoy.index} ({buoy.state}) attempt {buoy.attempts}/{self.buoy_attempts}"
            try:
                ok = self._attempt(goal_handle, buoy, label, position, total)
            except Paused:
                buoy.attempts -= 1  # an incident is not a failed attempt
                self._hold(goal_handle, buoy, position, total)
                continue
            if ok:
                buoy.status = "circled"
                self.get_logger().info(f"{label}: circled")
                return
            self.get_logger().warning(f"{label}: failed")
        buoy.status = "failed"

    def _attempt(self, goal_handle, buoy: Buoy, label: str, position: int, total: int) -> bool:
        estimate = self._estimate(buoy)
        boat = self._boat_xy()
        if boat is None:
            return False
        radius, why = geo.choose_radius(self._snapshot(), estimate.xy, self._neighbours(buoy), self.circle_radius,
                                        self.circle_radius_min, self.circle_radius_step, self.neighbour_clearance,
                                        self.cost_threshold)
        phi0 = geo.start_angle(estimate.xy, radius, boat, buoy.sigma)
        centre = estimate.xy

        def place(c):
            return geo.circle_points(c, radius, phi0, buoy.sigma, self.circle_points, self.circle_overshoot_points)

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
                    if np.linalg.norm(estimate.xy - centre) > self.estimate_move_threshold or retry > 0:
                        if np.linalg.norm(estimate.xy - centre) > 1e-6:
                            self.get_logger().info(
                                f"{label}: centre moved {np.linalg.norm(estimate.xy - centre):.2f} m "
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

    def _estimate(self, buoy: Buoy) -> ba.Estimate:
        with self._lock:
            claimed = [b.estimate.xy for b in self._buoys if b.status == "circled" and b.estimate is not None]
            return self._aligner.estimate(buoy.index, claimed)

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

    def _drive(self, goal_handle, xy, yaw: float, label: str) -> bool:
        """TrackPath to xy, ending at yaw (ENU). Raises Paused or Canceled."""
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
        goal.sample_radius = float(self.goal_sample_radius)
        goal.specified_heading = False
        goal.ignore_depth = True
        goal.forward_tolerance = goal.sidemove_tolerance = float(self.goal_tolerance)
        goal.heading_tolerance = math.pi  # heading unspecified
        goal.depth_tolerance = 1.0  # must stay positive, controls compares strictly
        goal.stabilize_duration = int(self.goal_stabilize)
        self.get_logger().info(f"{label}: TrackPath to ({xy[0]:.2f}, {xy[1]:.2f}) heading {math.degrees(yaw):.0f}")
        ok = self._run_action(goal_handle, self.track_path, self.track_path_name, goal, self.goal_timeout,
                              lambda r: r.status == TrackPath.Result.STATUS_SUCCESS)
        if not ok and self._pause_enabled:
            # The incident handler may have preempted us before its flag got here
            self._wait(goal_handle, lambda: self._incident, self.preempt_grace)
            self._check(goal_handle)
        return ok

    def _hold(self, goal_handle, buoy: Buoy, position: int, total: int):
        self.get_logger().warning(f"Incident: holding at buoy {buoy.index} ({buoy.state})")
        self._feedback(goal_handle, "PAUSED", buoy, position, total)
        with self._lock:
            self._target = None
        start = time.monotonic()
        event = threading.Event()
        while True:
            while self._incident and self._pause_enabled:
                if goal_handle.is_cancel_requested:
                    raise Canceled()
                event.wait(POLL_S)
            settle_end = time.monotonic() + self.resume_settle
            while time.monotonic() < settle_end and not self._incident:
                if goal_handle.is_cancel_requested:
                    raise Canceled()
                event.wait(POLL_S)
            if not self._incident:
                break
        self.get_logger().info(f"Incident cleared after {time.monotonic() - start:.0f} s: "
                               f"buoy {buoy.index} ({buoy.state}) starts over")

    # Plumbing

    def _check(self, goal_handle):
        if goal_handle.is_cancel_requested:
            raise Canceled()
        if self._pause_enabled and self._incident:
            raise Paused()

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
        except (Canceled, Paused):
            if send.done() and send.result().accepted:
                send.result().cancel_goal_async()
            raise
        succeeded = ok(result.result().result)
        if not succeeded:
            self.get_logger().warning(f"{name} did not succeed (status {result.result().result.status})")
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

    def _on_incident(self, msg: Bool):
        if msg.data != self._incident:
            self.get_logger().warning(f"Incident flag {'set' if msg.data else 'cleared'}")
        self._incident = bool(msg.data)

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
