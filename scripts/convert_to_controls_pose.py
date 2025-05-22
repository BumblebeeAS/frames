#!/usr/bin/env python3
from typing import Any, Optional

import rclpy
import tf2_geometry_msgs  # noqa: F401
from bb_planner_msgs.srv import GetPoseToControlsFrame
from geometry_msgs.msg import PoseStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener


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

        self.declare_parameter("controls_frame", "world_ned")
        self.controls_frame: str = (
            self.get_parameter("controls_frame").get_parameter_value().string_value
        )

        self.declare_parameter("base_frame", "auv4/base_link_ned")
        self.base_frame: str = (
            self.get_parameter("base_frame").get_parameter_value().string_value
        )

        self.tf_buffer: Buffer = Buffer()
        self.tf_listener: TransformListener = TransformListener(self.tf_buffer, self)

        self.conversion_service = self.create_service(
            GetPoseToControlsFrame,
            "convert_to_controls_pose",
            self.convert_callback,
        )

        self.get_logger().info("ConvertToControlsPose service started")
        self.get_logger().info(f"Using controls frame: '{self.controls_frame}'")
        self.get_logger().info(f"Using base frame: '{self.base_frame}'")

    def transform_to_frame(
        self, input_pose: PoseStamped, target_frame: str, timeout: float
    ) -> Optional[PoseStamped]:
        """
        Transform a pose to the specified target frame.

        Args:
            input_pose: The pose to transform
            target_frame: The target coordinate frame
            timeout: Maximum time to wait for the transform, in seconds

        Returns:
            The transformed pose if successful, None otherwise
        """
        if input_pose.header.frame_id == target_frame:
            return input_pose

        try:
            self.get_logger().debug(
                f"Transforming pose from '{input_pose.header.frame_id}' to '{target_frame}'"
            )

            output_pose = self.tf_buffer.transform(
                input_pose, target_frame, Duration(seconds=timeout)
            )

            self.get_logger().debug(
                f"Transformed pose: position [{output_pose.pose.position.x}, "
                f"{output_pose.pose.position.y}, {output_pose.pose.position.z}], "
                f"orientation [{output_pose.pose.orientation.w}, {output_pose.pose.orientation.x}, "
                f"{output_pose.pose.orientation.y}, {output_pose.pose.orientation.z}]"
            )

            return output_pose

        except Exception as e:
            self.get_logger().error(
                f"Failed to transform pose from '{input_pose.header.frame_id}' to '{target_frame}': {str(e)}"
            )
            import traceback

            self.get_logger().info(f"Exception traceback: {traceback.format_exc()}")
            return None

    def recalculate_target(
        self, original_target: PoseStamped, anchor_frame: str, timeout: float
    ) -> Optional[PoseStamped]:
        """
        Recalculate a target pose based on an anchor frame.

        This function adjusts the position of the pose based on the translation
        between the anchor frame and the base frame, while keeping the orientation unchanged.

        Args:
            original_target: The original pose to recalculate
            anchor_frame: The anchor coordinate frame to use for recalculation
            timeout: Maximum time to wait for the transform, in seconds

        Returns:
            The recalculated pose if successful, None otherwise
        """
        if anchor_frame == self.base_frame:
            return original_target

        try:
            self.get_logger().debug(
                f"Getting translation from '{anchor_frame}' to '{self.base_frame}'"
            )

            transform = self.tf_buffer.lookup_transform(
                self.base_frame,
                anchor_frame,
                Time.from_msg(original_target.header.stamp),
                Duration(seconds=timeout),
            )
        except Exception as e:
            self.get_logger().error(
                f"Failed to find transform from '{anchor_frame}' to '{self.base_frame}': {str(e)}"
            )
            import traceback

            self.get_logger().debug(f"Exception traceback: {traceback.format_exc()}")
            return None

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

        self.get_logger().debug(
            f"Recalculated target pose: position [{output_pose.pose.position.x}, "
            f"{output_pose.pose.position.y}, {output_pose.pose.position.z}], "
            f"orientation [{output_pose.pose.orientation.w}, {output_pose.pose.orientation.x}, "
            f"{output_pose.pose.orientation.y}, {output_pose.pose.orientation.z}]"
        )

        return output_pose

    def convert_callback(
        self,
        request: GetPoseToControlsFrame.Request,
        response: GetPoseToControlsFrame.Response,
    ) -> GetPoseToControlsFrame.Response:
        """
        Service callback to handle pose conversion requests.

        The service performs the following steps:
        1. Transform the input pose to the base frame
        2. Recalculate the pose if an anchor frame is specified
        3. Transform the result to the controls frame

        Args:
            request: The service request containing input pose, anchor frame, and timeout
            response: The service response object to populate

        Returns:
            The populated service response with output pose and success status
        """
        input_pose = request.input_pose
        anchor_frame = request.anchor_frame_name
        timeout = request.timeout

        self.get_logger().info(
            f"Received transform request from frame '{input_pose.header.frame_id}' to '{self.controls_frame}'"
        )
        self.get_logger().debug(
            f"Input pose: position [{input_pose.pose.position.x}, {input_pose.pose.position.y}, {input_pose.pose.position.z}], "
            f"orientation [{input_pose.pose.orientation.w}, {input_pose.pose.orientation.x}, "
            f"{input_pose.pose.orientation.y}, {input_pose.pose.orientation.z}]"
        )

        if timeout <= 0.0:
            timeout = 1.0
            self.get_logger().debug(f"Using default timeout of {timeout} seconds")
        else:
            self.get_logger().debug(f"Using specified timeout of {timeout} seconds")

        pose_in_base_frame = self.transform_to_frame(
            input_pose, self.base_frame, timeout
        )
        if pose_in_base_frame is None:
            response.output_pose = input_pose
            response.tf_success = False
            return response

        recalculated_pose = self.recalculate_target(
            pose_in_base_frame, anchor_frame, timeout
        )
        if recalculated_pose is None:
            response.output_pose = input_pose
            response.tf_success = False
            return response

        final_pose = self.transform_to_frame(
            recalculated_pose, self.controls_frame, timeout
        )
        if final_pose is None:
            response.output_pose = input_pose
            response.tf_success = False
            return response

        response.output_pose = final_pose
        response.tf_success = True
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
