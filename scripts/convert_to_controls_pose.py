#!/usr/bin/env python3
from bisect import bisect_left
from collections import deque
from typing import Any, Optional

import rclpy
import tf2_ros
from bb_planner_msgs.srv import GetPoseToControlsFrame
from builtin_interfaces.msg import Time
from frames.utils.transform_ros_msgs import (
    compose_transforms,
    get_base_to_world_tf_stamped,
    get_world_to_base_tf_stamped,
)
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from rclpy.time import Time as RclpyTime
from tf2_geometry_msgs import do_transform_pose_stamped
from tf2_msgs.msg import TFMessage
from tf2_ros.buffer import Buffer


def header_time_to_seconds(header_time: Time) -> float:
    return header_time.sec + header_time.nanosec * 1e-9


class ConvertToControlsPose(Node):
    """
    Provides service to transform poses to a specific coordinate frame that controls accepts
    (usually 'world_ned'), which may include recalculating the pose based on an anchor frame if necessary.
    """

    def __init__(self, name: str = "convert_to_controls_pose_node") -> None:
        """
        Initialize the ConvertToControlsPose node.

        Args:
            name: The name of the ROS2 node
        """
        super().__init__(name)

        # ------ Declare parameters ------
        self.controls_frame: str = (
            self.declare_parameter("controls_frame", "world_ned")
            .get_parameter_value()
            .string_value
        )
        self.base_frame: str = (
            self.declare_parameter("base_frame", "auv4/base_link_ned")
            .get_parameter_value()
            .string_value
        )
        self.queue_size: int = (
            self.declare_parameter("queue_size", 100)
            .get_parameter_value()
            .integer_value
        )
        self.odom_topic: str = (
            self.declare_parameter("odom_topic", "/uav2/odom_ned")
            .get_parameter_value()
            .string_value
        )

        # ------ Initialize subs and srv ------
        static_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.tf_buffer: Buffer = Buffer()
        self.odom_buffer: deque[Odometry] = deque(maxlen=self.queue_size)
        self.tf_sub = self.create_subscription(
            TFMessage,
            "/tf_static",
            self.handle_tf_static,
            qos_profile=static_qos,
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.handle_odom,
            qos_profile=qos_profile_sensor_data,
        )
        self.conversion_service = self.create_service(
            GetPoseToControlsFrame,
            "convert_to_controls_pose",
            self.convert_callback,
        )

        self.get_logger().info("ConvertToControlsPose service started")
        self.get_logger().info(f"Using controls frame: '{self.controls_frame}'")
        self.get_logger().info(f"Using base frame: '{self.base_frame}'")
        self.get_logger().info(f"Using queue size: {self.queue_size}")
        self.get_logger().info(f"Using odom topic: {self.odom_topic}")

    def handle_tf_static(self, msg: TFMessage) -> None:
        """
        Callback to set static transforms in the TF buffer.
        """
        self.get_logger().info("Received static TF transforms")
        for tf in msg.transforms:
            self.tf_buffer.set_transform_static(tf, "default_authority")

    def handle_odom(self, msg: Odometry) -> None:
        self.odom_buffer.append(msg)

    def get_odom_at_time(self, time: float) -> Odometry:
        """
        Get the odometry message closest to the specified time.

        Args:
            time: The time to find the closest odometry for, in seconds

        Returns:
            The closest odometry message
        """
        idx = min(
            bisect_left(
                self.odom_buffer,
                key=lambda odom: (header_time_to_seconds(odom.header.stamp)),
                x=time,
            ),
            len(self.odom_buffer) - 1,
        )

        before = self.odom_buffer[max(idx - 1, 0)]
        after = self.odom_buffer[idx]

        before_time = header_time_to_seconds(before.header.stamp)
        after_time = header_time_to_seconds(after.header.stamp)

        before_time_diff = abs(before_time - time)
        after_time_diff = abs(after_time - time)

        self.get_logger().info(f"time diff (before): {before_time_diff}")
        self.get_logger().info(f"time diff (after): {after_time_diff}")

        if before_time_diff <= after_time_diff:
            return before
        return after

    def get_transform(
        self, target_frame: list[str], input_pose: PoseStamped, timeout: float
    ) -> Optional[tf2_ros.TransformStamped]:
        """
        Get the transform from the input pose frame to the target frame.
        """
        is_from_base_link = False
        target_to_input_transform = None

        try:
            target_to_input_transform = self.tf_buffer.lookup_transform(
                target_frame=target_frame[0],
                source_frame=input_pose.header.frame_id,
                time=RclpyTime(seconds=0),
                timeout=Duration(seconds=timeout),  # type: ignore
            )
            is_from_base_link = True
        except (
            tf2_ros.LookupException,  #  # type: ignore
            tf2_ros.ConnectivityException,  # type: ignore
            tf2_ros.ExtrapolationException,  # type: ignore
        ) as e:
            self.get_logger().error(f"Transform lookup failed: {e}")

        if not is_from_base_link:
            # tf is from world to input frame
            try:
                target_to_input_transform = self.tf_buffer.lookup_transform(
                    target_frame=target_frame[1],
                    source_frame=input_pose.header.frame_id,
                    time=RclpyTime(seconds=0),
                    timeout=Duration(seconds=timeout),  # type: ignore
                )
                odom = self.get_odom_at_time(
                    header_time_to_seconds(target_to_input_transform.header.stamp)
                )

                world_to_base_tf = get_world_to_base_tf_stamped(odom, self.base_frame)
                target_to_input_transform = compose_transforms(
                    target_to_input_transform.transform, world_to_base_tf.transform
                )
                target_to_input_transform = tf2_ros.TransformStamped(
                    header=world_to_base_tf.header,
                    child_frame_id=input_pose.header.frame_id,
                    transform=target_to_input_transform,
                )
            except (
                tf2_ros.LookupException,  # type: ignore
                tf2_ros.ConnectivityException,  # type: ignore
                tf2_ros.ExtrapolationException,  # type: ignore
            ) as e:
                self.get_logger().error(f"Transform lookup failed: {e}")
                return None

        return target_to_input_transform

    def transform_to_frame(
        self, input_pose: PoseStamped, target_frame: list[str], timeout: float
    ) -> Optional[PoseStamped]:
        """
        Transform a pose to the specified target_frame[0].

        Args:
            input_pose: The pose to transform
            target_frame: The target coordinate frame
            timeout: Maximum time to wait for the transform, in seconds

        Returns:
            The transformed pose

        Raises:
            Exception: If the transform fails
        """
        if input_pose.header.frame_id == target_frame[0]:
            return input_pose

        self.get_logger().debug(
            f"Transforming pose from '{input_pose.header.frame_id}' to '{target_frame[0]}'"
        )
        self.target_to_input_transform = self.get_transform(
            target_frame, input_pose, timeout
        )

        if self.target_to_input_transform is None:
            self.get_logger().error("Failed to obtain target_to_input_transform.")
            return None

        output_pose = do_transform_pose_stamped(
            input_pose, self.target_to_input_transform
        )
        self.get_logger().info(
            f"Transformed pose: position [{output_pose.pose.position.x}, "
            f"{output_pose.pose.position.y}, {output_pose.pose.position.z}], "
            f"orientation [{output_pose.pose.orientation.w}, {output_pose.pose.orientation.x}, "
            f"{output_pose.pose.orientation.y}, {output_pose.pose.orientation.z}]"
            f"into target frame: {target_frame}"
        )

        return output_pose

    def recalculate_target(
        self, original_target: PoseStamped, anchor_frame: str, timeout: float
    ) -> PoseStamped:
        """
        Recalculate a target pose based on an anchor frame.

        This function adjusts the position of the pose based on the translation
        between the anchor frame and the base frame, while keeping the orientation unchanged.

        Args:
            original_target: The original pose to recalculate
            anchor_frame: The anchor coordinate frame to use for recalculation
            timeout: Maximum time to wait for the transform, in seconds

        Returns:
            The recalculated pose

        Raises:
            Exception: If the transform lookup fails
        """
        if anchor_frame == self.base_frame:
            return original_target

        self.get_logger().debug(
            f"Getting translation from '{anchor_frame}' to '{self.base_frame}'"
        )

        transform = self.tf_buffer.lookup_transform(
            self.base_frame,
            anchor_frame,
            Duration(seconds=timeout),  # type: ignore
        )

        # Create output pose and copy original pose data
        output_pose = PoseStamped()
        output_pose.header.stamp = original_target.header.stamp
        output_pose.header.frame_id = self.base_frame

        # Copy the orientation directly
        output_pose.pose.orientation = original_target.pose.orientation

        # Only subtract the translation component from the transform
        output_pose.pose.position.x = (
            original_target.pose.position.x - transform.transform.translation.x
        )
        output_pose.pose.position.y = (
            original_target.pose.position.y - transform.transform.translation.y
        )
        output_pose.pose.position.z = (
            original_target.pose.position.z - transform.transform.translation.z
        )

        self.get_logger().info(
            f"Recalculated target pose: position [{output_pose.pose.position.x}, "
            f"{output_pose.pose.position.y}, {output_pose.pose.position.z}], "
            f"orientation [{output_pose.pose.orientation.w}, {output_pose.pose.orientation.x}, "
            f"{output_pose.pose.orientation.y}, {output_pose.pose.orientation.z}]"
            f"with anchor frame {anchor_frame}"
        )

        return output_pose

    def convert_callback(
        self,
        request: GetPoseToControlsFrame.Request,
        response: GetPoseToControlsFrame.Response,
    ) -> GetPoseToControlsFrame.Response:
        """
        Service callback to handle pose conversion requests for multiple poses.

        The service performs the following steps for each input pose:
        1. Transform the input pose to the base frame
        2. Recalculate the pose if an anchor frame is specified
        3. Transform the result to the controls frame

        Args:
            request: The service request containing input poses, anchor frame, and timeout
            response: The service response object to populate

        Returns:
            The populated service response with output poses and success status
        """
        input_poses = request.input_poses
        anchor_frame = request.anchor_frame_name
        timeout = request.timeout

        response.tf_success = False
        response.output_poses = input_poses

        output_poses = []

        self.get_logger().info(
            f"Received transform request for {len(input_poses)} poses to '{self.controls_frame}'"
        )

        if timeout <= 0.0:
            timeout = 1.0
            self.get_logger().debug(f"Using default timeout of {timeout} seconds")
        else:
            self.get_logger().debug(f"Using specified timeout of {timeout} seconds")

        try:
            for i, input_pose in enumerate(input_poses):
                self.get_logger().info(
                    f"Processing pose {i + 1}/{len(input_poses)} from frame '{input_pose.header.frame_id}'"
                )
                self.get_logger().info(
                    f"Input pose {i + 1}: position [{input_pose.pose.position.x}, {input_pose.pose.position.y}, {input_pose.pose.position.z}], "
                    f"orientation [{input_pose.pose.orientation.w}, {input_pose.pose.orientation.x}, "
                    f"{input_pose.pose.orientation.y}, {input_pose.pose.orientation.z}]"
                )

                # Step 1: Transform input pose to base frame
                pose_in_base_frame = self.transform_to_frame(
                    input_pose, [self.base_frame, self.controls_frame], timeout
                )

                if pose_in_base_frame is None:
                    self.get_logger().error(
                        f"Failed to transform pose {i + 1} to base frame '{self.base_frame}'"
                    )
                    break
                assert self.target_to_input_transform is not None, (
                    "Transform should not be None here"
                )

                # Step 2: Recalculate pose based on anchor frame
                recalculated_pose = self.recalculate_target(
                    pose_in_base_frame, anchor_frame, timeout
                )

                # Step 3: Transform to controls frame
                odom_transform = get_base_to_world_tf_stamped(
                    self.get_odom_at_time(
                        header_time_to_seconds(
                            self.target_to_input_transform.header.stamp
                        )
                    )
                )

                # Apply odom transform to get final pose in odom parent frame
                final_pose = do_transform_pose_stamped(
                    recalculated_pose, odom_transform
                )

                assert final_pose is not None, (
                    f"Final transform to '{self.controls_frame}' failed"
                )
                output_poses.append(final_pose)

            response.output_poses = output_poses
            response.tf_success = True
            self.get_logger().info(f"Successfully converted {len(output_poses)} poses")

        except Exception as e:
            self.get_logger().error(f"Failed to convert poses: {str(e)}")
            import traceback

            self.get_logger().debug(f"Exception traceback: {traceback.format_exc()}")

        return response


def main(args: Optional[Any] = None) -> None:
    """
    Main entry point for the node.

    Args:
        args: Command-line arguments to pass to rclpy.init
    """
    rclpy.init(args=args)
    node = ConvertToControlsPose()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
