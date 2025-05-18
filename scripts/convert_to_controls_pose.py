#!/usr/bin/env python3
import rclpy
from bb_planner_msgs.srv import GetPoseToControlsFrame
from geometry_msgs.msg import PoseStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from tf2_geometry_msgs import do_transform_pose
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener


class ConvertToControlsPose(Node):
    def __init__(self, name="convert_to_controls_pose_node"):
        super().__init__(name)

        # Declare parameter for controls frame
        self.declare_parameter("controls_frame", "world_ned")
        self.controls_frame = (
            self.get_parameter("controls_frame").get_parameter_value().string_value
        )

        self.declare_parameter("base_frame", "auv4/base_link_ned")
        self.base_frame = (
            self.get_parameter("base_frame").get_parameter_value().string_value
        )

        # Create TF2 buffer and listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Create service
        self.conversion_service = self.create_service(
            GetPoseToControlsFrame,
            "convert_to_controls_pose",
            self.convert_callback,
        )

        self.get_logger().info("ConvertToControlsPose service started")
        self.get_logger().info(f"Using controls frame: '{self.controls_frame}'")

    def get_target_in_controls_frame(self, input_pose, timeout):
        if input_pose.header.frame_id == self.controls_frame:
            return input_pose

        try:
            self.get_logger().debug(f"Transforming pose from '{input_pose.header.frame_id}' to '{self.controls_frame}'")

            output_pose = self.tf_buffer.transform(
                input_pose, self.controls_frame, Duration(seconds=timeout)
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
                f"Failed to transform pose from '{input_pose.header.frame_id}' to '{self.controls_frame}': {str(e)}"
            )
            # Log the traceback for detailed debugging
            import traceback

            self.get_logger().debug(f"Exception traceback: {traceback.format_exc()}")

            return None

    def recalculate_target(
        self, original_target: PoseStamped, anchor_frame: str, timeout
    ):
        if anchor_frame == self.base_frame:
            return original_target

        try:
            self.get_logger().debug(f"Transforming pose from '{anchor_frame}' to '{self.base_frame}'")

            # Grab latest transform from anchor frame to base frame
            transform = self.tf_buffer.lookup_transform(
                self.base_frame, anchor_frame, Time(), Duration(seconds=timeout)
            )
        except Exception as e:
            self.get_logger().error(
                f"Failed to find static transform from '{anchor_frame}' to '{self.base_frame}': {str(e)}"
            )
            import traceback

            self.get_logger().debug(f"Exception traceback: {traceback.format_exc()}")

            return None

        transformed_pose = do_transform_pose(original_target.pose, transform)

        output_pose = PoseStamped()
        output_pose.header.stamp = original_target.header.stamp
        output_pose.header.frame_id = original_target.header.frame_id

        output_pose.pose.position = transformed_pose.position

        output_pose.pose.orientation = original_target.pose.orientation

        self.get_logger().debug(
            f"Transformed target pose: position [{output_pose.pose.position.x}, "
            f"{output_pose.pose.position.y}, {output_pose.pose.position.z}], "
            f"orientation [{output_pose.pose.orientation.w}, {output_pose.pose.orientation.x}, "
            f"{output_pose.pose.orientation.y}, {output_pose.pose.orientation.z}]"
        )

        return output_pose

    def convert_callback(
        self,
        request: GetPoseToControlsFrame.Request,
        response: GetPoseToControlsFrame.Response,
    ):
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

        # Set default timeout if not specified
        if timeout <= 0.0:
            timeout = 1.0  # Default timeout of 1 second
            self.get_logger().debug(f"Using default timeout of {timeout} seconds")
        else:
            self.get_logger().debug(f"Using specified timeout of {timeout} seconds")

        original_target = self.get_target_in_controls_frame(input_pose, timeout)

        if original_target is None:
            response.output_pose = input_pose
            response.tf_success = False

            return response

        transformed_target = self.recalculate_target(
            original_target, anchor_frame, timeout
        )

        if transformed_target is None:
            response.output_pose = input_pose
            response.tf_success = False

            return response

        response.output_pose = transformed_target
        response.tf_success = True

        return response


def main(args=None):
    rclpy.init(args=args)
    node = ConvertToControlsPose()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
