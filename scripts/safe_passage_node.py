#!/usr/bin/env python3
"""RobotX 2026 Task 1 (Safe Passage) on the ASV, from the drone's path.

Serves bb_robotx_msgs/SafePassage. A goal waits for a new geographic_msgs/GeoPath (poses[0] entry buoy, poses[-1]
exit buoy, the rest pass-through points in order), converts it once into the global frame with fromLL, then runs
the legs in order:

  ENTRY_APPROACH  TrackPath to a free point outside the entry circle
  ENTRY_CIRCLE    Encircle clockwise (seen from above), one round plus overshoot
  WAYPOINT i      TrackPath through before -> point -> after, along the line from the previous point to the next
  EXIT_APPROACH   TrackPath to a free point outside the exit circle
  EXIT_CIRCLE     Encircle counterclockwise, one round plus overshoot

Each leg samples against its own snapshot of the global costmap (frames.utils.safe_passage_geometry), taken when the
leg starts. A failed or timed out leg is logged and skipped: only a missing path aborts the goal.

Conventions: the path is WGS84; everything is planned in global_frame_id (asv5/map, ENU) and TrackPath goals are
GOAL_POSE there with specified_heading false. Encircle takes its centre in the controls frame (odom NED, x north,
y east, via convert_to_controls_pose) and turn_angle in degrees, positive clockwise seen from above.
"""

import math
import threading
from typing import Callable, Optional

import numpy as np
import rclpy
import tf2_ros
from bb_controls_msgs.action import Encircle, TrackPath
from bb_planner_msgs.srv import GetPoseToControlsFrame
from bb_robotx_msgs.action import SafePassage
from builtin_interfaces.msg import Time
from geographic_msgs.msg import GeoPath
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time as RosTime
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from robot_localization.srv import FromLL

from frames.utils import safe_passage_geometry as geo

# Best effort: the path crosses a lossy link from the drone. Volatile: only live paths count.
PATH_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST, depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE,
)
# nav2 costmaps publish reliable transient local
COSTMAP_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST, depth=1,
    reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL,
)
POLL_S = 0.05


class Canceled(Exception):
    """The SafePassage goal was canceled."""


class SafePassageNode(Node):
    def __init__(self):
        super().__init__("safe_passage_node")
        param = lambda name, default: self.declare_parameter(name, default).value  # noqa: E731

        self.path_topic = param("path_topic", "/uav/safe_passage/path")
        self.path_wait_timeout = param("path_wait_timeout", 30.0)
        self.costmap_topic = param("costmap_topic", "/asv5/global_costmap/costmap")
        self.global_frame_id = param("global_frame_id", "asv5/map")
        self.base_link_frame_id = param("base_link_frame_id", "asv5/base_link")
        self.service_timeout = param("service_timeout", 2.0)

        # Circles
        self.circle_radius = param("circle_radius", 2.5)
        self.circle_radius_tol = param("circle_radius_tol", 0.5)
        self.circle_overshoot_deg = param("circle_overshoot_deg", 20.0)
        self.snap_radius = param("snap_radius", 1.0)
        self.approach_offset = param("approach_offset", 1.5)
        self.approach_step_deg = param("approach_step_deg", 30.0)
        self.circle_tolerance = param("circle_tolerance", 1.0)
        self.circle_heading_tolerance = param("circle_heading_tolerance", math.radians(30.0))
        self.circle_max_correction_time = param("circle_max_correction_time", 5)
        self.circle_timeout = param("circle_timeout", 180.0)
        self.circle_max_xy_vel = param("circle_max_xy_vel", -1.0)

        # Pass-through points. Keep sample_radius below the manager's standoff distance.
        self.sample_radius = param("sample_radius", 1.0)
        self.sample_rings = param("sample_rings", 2)
        self.sample_per_ring = param("sample_per_ring", 8)
        self.lead_distance = param("lead_distance", 1.5)
        self.cost_threshold = param("cost_threshold", 50)
        self.check_spacing = param("check_spacing", 0.2)

        # TrackPath
        self.waypoint_tolerance = param("waypoint_tolerance", 1.0)
        self.waypoint_stabilize = param("waypoint_stabilize", 0)
        self.approach_tolerance = param("approach_tolerance", 1.0)  # below approach_offset: stop outside the circle
        self.approach_stabilize = param("approach_stabilize", 2)
        self.max_replans = param("max_replans", 10)
        self.leg_timeout = param("leg_timeout", 120.0)

        if self.sample_radius >= self.lead_distance > 0:
            self.get_logger().warning("sample_radius >= lead_distance: before/after samples may swap order")
        if self.approach_tolerance >= self.approach_offset:
            self.get_logger().warning("approach_tolerance >= approach_offset: the approach may end inside the circle")

        self.cb_group = ReentrantCallbackGroup()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self._costmap: Optional[OccupancyGrid] = None
        self.create_subscription(OccupancyGrid, self.costmap_topic, self._on_costmap, COSTMAP_QOS,
                                 callback_group=self.cb_group)

        self.track_path_name = param("track_path_action", "/asv5/track_path")
        self.encircle_name = param("encircle_action", "/asv5/in/locomotion/encircle")
        self.track_path = ActionClient(self, TrackPath, self.track_path_name, callback_group=self.cb_group)
        self.encircle = ActionClient(self, Encircle, self.encircle_name, callback_group=self.cb_group)
        self.from_ll = self.create_client(FromLL, param("from_ll_service", "/asv5/fromLL"), callback_group=self.cb_group)
        self.to_controls = self.create_client(
            GetPoseToControlsFrame, param("convert_to_controls_pose_service", "/asv5/convert_to_controls_pose"),
            callback_group=self.cb_group)

        self._busy = threading.Lock()
        self.server = ActionServer(
            self, SafePassage, param("action_name", "/asv5/safe_passage"),
            execute_callback=self._execute, goal_callback=self._on_goal, cancel_callback=lambda _: CancelResponse.ACCEPT,
            callback_group=self.cb_group,
        )
        self.get_logger().info(f"Safe Passage ready, default path topic '{self.path_topic}'")

    # Goal handling

    def _on_goal(self, _goal) -> GoalResponse:
        if self._busy.locked():
            self.get_logger().error("Rejected Safe Passage goal: one is already running")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _execute(self, goal_handle):
        with self._busy:
            result = SafePassage.Result()
            try:
                result.success, result.message = self._run(goal_handle)
            except Canceled:
                result.success, result.message = False, "canceled"
                goal_handle.canceled()
                return result
            if result.message.startswith("no path"):
                goal_handle.abort()
            else:
                goal_handle.succeed()
            self.get_logger().info(f"Safe Passage done: {result.message}")
            return result

    def _run(self, goal_handle) -> tuple[bool, str]:
        failed: list[str] = []
        topic = goal_handle.request.path_topic or self.path_topic

        def feedback(phase: str, index: int = 0, total: int = 0):
            self._check_cancel(goal_handle)
            goal_handle.publish_feedback(SafePassage.Feedback(phase=phase, index=index, total=total))
            self.get_logger().info(f"Safe Passage {phase}" + (f" {index + 1}/{total}" if phase == "WAYPOINT" else ""))

        feedback("WAIT_PATH")
        path = self._wait_for_path(goal_handle, topic)
        if path is None:
            return False, f"no path on {topic} within {self.path_wait_timeout:.0f} s"
        points = self._from_ll(path)
        if points is None:
            return False, "no path: fromLL conversion failed"
        entry, waypoints, exit_ = points[0], points[1:-1], points[-1]
        total = len(waypoints)
        self.get_logger().info(f"Path: entry {entry.round(1)}, {total} pass-through point(s), exit {exit_.round(1)}")

        # Entry
        feedback("ENTRY_APPROACH", 0, total)
        grid = self._snapshot()
        centre = geo.snap_centre(grid, entry, self.snap_radius)
        radius = geo.choose_radius(grid, centre, self.circle_radius, self.circle_radius_tol, spacing=self.check_spacing)
        boat = self._boat_xy()
        if boat is None or not self._approach(goal_handle, grid, centre, radius, boat, clockwise=True):
            failed.append("ENTRY_APPROACH")
        feedback("ENTRY_CIRCLE", 0, total)
        if not self._circle(goal_handle, centre, radius, geo.circle_turn_deg(True, self.circle_overshoot_deg)):
            failed.append("ENTRY_CIRCLE")

        # Pass-through points, nominal neighbours give the crossing direction
        nominal = [entry, *waypoints, exit_]
        for i, point in enumerate(waypoints):
            feedback("WAYPOINT", i, total)
            grid = self._snapshot()
            middle = geo.sample_point(grid, point, self.sample_radius, self.sample_rings, self.sample_per_ring)
            d = geo.crossing_direction(nominal[i], nominal[i + 2])
            yaw = math.atan2(d[1], d[0])
            legs = [middle]
            if self.lead_distance > 0:
                legs = [geo.sample_point(grid, middle + s * self.lead_distance * d, self.sample_radius,
                                         self.sample_rings, self.sample_per_ring) for s in (-1.0, 1.0)]
                legs.insert(1, middle)
            for j, leg in enumerate(legs):
                if not self._track(goal_handle, leg, yaw, self.waypoint_tolerance, self.waypoint_stabilize):
                    failed.append(f"WAYPOINT {i + 1} leg {j + 1}")

        # Exit
        feedback("EXIT_APPROACH", total, total)
        grid = self._snapshot()
        centre = geo.snap_centre(grid, exit_, self.snap_radius)
        radius = geo.choose_radius(grid, centre, self.circle_radius, self.circle_radius_tol, spacing=self.check_spacing)
        boat = self._boat_xy()
        if boat is None or not self._approach(goal_handle, grid, centre, radius, boat, clockwise=False):
            failed.append("EXIT_APPROACH")
        feedback("EXIT_CIRCLE", total, total)
        if not self._circle(goal_handle, centre, radius, geo.circle_turn_deg(False, self.circle_overshoot_deg)):
            failed.append("EXIT_CIRCLE")

        return not failed, ("all legs succeeded" if not failed else "failed: " + ", ".join(failed))

    # Legs

    def _approach(self, goal_handle, grid, centre, radius, boat, clockwise: bool) -> bool:
        point = geo.choose_approach(grid, centre, radius, self.approach_offset, boat, clockwise, self.cost_threshold,
                                    self.approach_step_deg, self.check_spacing)
        # End facing the tangent point Encircle joins at, so its first turn is small
        run = geo.tangent_point(centre, radius, point, clockwise) - point
        return self._track(goal_handle, point, math.atan2(run[1], run[0]), self.approach_tolerance,
                           self.approach_stabilize)

    def _track(self, goal_handle, xy, yaw: float, tolerance: float, stabilize: int) -> bool:
        """TrackPath to xy. yaw (ENU) is where it ends facing: with specified_heading false the vehicle still turns
        to the last heading setpoint, so this is the direction of travel, not a constraint along the way."""
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
        goal.specified_heading = False
        goal.ignore_depth = True
        goal.forward_tolerance = goal.sidemove_tolerance = float(tolerance)
        goal.heading_tolerance = math.pi  # heading unspecified
        goal.depth_tolerance = 1.0  # must stay positive, controls compares strictly
        goal.stabilize_duration = int(stabilize)
        self.get_logger().info(f"TrackPath to ({xy[0]:.1f}, {xy[1]:.1f})")
        return self._run_action(goal_handle, self.track_path, self.track_path_name, goal, self.leg_timeout,
                                lambda r: r.status == TrackPath.Result.STATUS_SUCCESS)

    def _circle(self, goal_handle, centre, radius: float, turn_deg: float) -> bool:
        ned = self._to_controls(centre)
        if ned is None:
            return False
        goal = Encircle.Goal()
        goal.turn_angle = float(turn_deg)
        goal.radius = float(radius)
        goal.centre_x, goal.centre_y = float(ned[0]), float(ned[1])  # north, east
        goal.tangent_offset = 0.0
        goal.spiral_proportion = 0.0
        goal.specified_heading = False  # turn to the circle, join it tangentially from outside, then sweep
        goal.forward_tolerance = goal.sidemove_tolerance = float(self.circle_tolerance)
        goal.heading_tolerance = float(self.circle_heading_tolerance)
        goal.depth_tolerance = 1.0
        goal.max_correction_time = int(self.circle_max_correction_time)
        goal.max_xy_vel = float(self.circle_max_xy_vel)
        self.get_logger().info(
            f"Encircle {'clockwise' if turn_deg > 0 else 'counterclockwise'} {abs(turn_deg):.0f} deg, "
            f"R {radius:.2f} m, centre map ({centre[0]:.1f}, {centre[1]:.1f}) = NED ({ned[0]:.1f}, {ned[1]:.1f})")
        return self._run_action(goal_handle, self.encircle, self.encircle_name, goal, self.circle_timeout,
                                lambda r: r.status == Encircle.Result.STATUS_SUCCESS)

    # Plumbing

    def _check_cancel(self, goal_handle):
        if goal_handle.is_cancel_requested:
            raise Canceled()

    def _wait(self, goal_handle, done: Callable[[], bool], timeout: float) -> bool:
        """Poll done until it holds or timeout passes; raises Canceled. Never spins: the executor runs callbacks."""
        deadline = self.get_clock().now() + Duration(seconds=timeout)
        event = threading.Event()
        while not done():
            self._check_cancel(goal_handle)
            if self.get_clock().now() > deadline:
                return False
            event.wait(POLL_S)
        return True

    def _run_action(self, goal_handle, client: ActionClient, name: str, goal, timeout: float, ok: Callable) -> bool:
        if not client.wait_for_server(timeout_sec=self.service_timeout):
            self.get_logger().error(f"{name} unavailable")
            return False
        send = client.send_goal_async(goal)
        try:
            if not self._wait(goal_handle, send.done, self.service_timeout) or not send.result().accepted:
                self.get_logger().error(f"{name} did not accept the goal")
                return False
            child = send.result()
            result = child.get_result_async()
            if not self._wait(goal_handle, result.done, timeout):
                self.get_logger().error(f"{name} timed out after {timeout:.0f} s, canceling")
                child.cancel_goal_async()
                return False
        except Canceled:
            if send.done() and send.result().accepted:
                send.result().cancel_goal_async()
            raise
        succeeded = ok(result.result().result)
        if not succeeded:
            self.get_logger().error(f"{name} failed (status {result.result().result.status})")
        return succeeded

    def _wait_for_path(self, goal_handle, topic: str) -> Optional[GeoPath]:
        """First GeoPath with at least two poses published after the goal started."""
        received: list[GeoPath] = []

        def on_path(msg: GeoPath):
            if len(msg.poses) >= 2 and not received:
                received.append(msg)

        sub = self.create_subscription(GeoPath, topic, on_path, PATH_QOS, callback_group=self.cb_group)
        try:
            self._wait(goal_handle, lambda: bool(received), self.path_wait_timeout)
        finally:
            self.destroy_subscription(sub)
        return received[0] if received else None

    def _from_ll(self, path: GeoPath) -> Optional[list[np.ndarray]]:
        points = []
        for pose in path.poses:
            request = FromLL.Request()
            request.ll_point = pose.pose.position
            response = self._call(self.from_ll, request)
            if response is None:
                return None
            points.append(np.array([response.map_point.x, response.map_point.y]))
        return points

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

    def _to_controls(self, xy) -> Optional[np.ndarray]:
        pose = PoseStamped()
        pose.header.frame_id = self.global_frame_id  # zero stamp: latest
        pose.pose.position.x, pose.pose.position.y = float(xy[0]), float(xy[1])
        pose.pose.orientation.w = 1.0
        request = GetPoseToControlsFrame.Request(
            input_poses=[pose], anchor_frame_name=self.base_link_frame_id, timeout=float(self.service_timeout))
        response = self._call(self.to_controls, request)
        if response is None or not response.tf_success or not response.output_poses:
            self.get_logger().error("convert_to_controls_pose failed for the circle centre")
            return None
        p = response.output_poses[0].pose.position
        return np.array([p.x, p.y])

    def _boat_xy(self) -> Optional[np.ndarray]:
        try:
            t = self.tf_buffer.lookup_transform(self.global_frame_id, self.base_link_frame_id, RosTime(),
                                                Duration(seconds=0.5)).transform.translation
        except tf2_ros.TransformException as e:
            self.get_logger().error(f"Boat pose lookup failed: {e}")
            return None
        return np.array([t.x, t.y])

    def _on_costmap(self, msg: OccupancyGrid):
        self._costmap = msg

    def _snapshot(self) -> geo.CostGrid:
        msg = self._costmap
        if msg is None:
            self.get_logger().warning(f"No costmap on {self.costmap_topic}, sampling on an empty one")
            return geo.CostGrid.empty()
        if msg.header.frame_id != self.global_frame_id:
            self.get_logger().warning(f"Costmap in '{msg.header.frame_id}', not '{self.global_frame_id}': ignored")
            return geo.CostGrid.empty()
        return geo.CostGrid.from_msg(msg)


def main(args=None):
    rclpy.init(args=args)
    node = SafePassageNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
