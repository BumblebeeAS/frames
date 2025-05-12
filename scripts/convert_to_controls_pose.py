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

    def convert_callback(self, request, response):
        input_pose = request.input_pose
        timeout = request.timeout

        self.get_logger().info(
            f"Received transform request from frame '{input_pose.header.frame_id}' to '{self.controls_frame}'"
        )
        if input_pose.header.frame_id == self.controls_frame:
            response.output_pose = input_pose
            response.tf_success = True

            return response

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

        try:
            # Determine the timestamp to use
            when = (
                Time.from_msg(input_pose.header.stamp)
                if input_pose.header.stamp.sec != 0
                else self.get_clock().now()
            )

            self.get_logger().debug(
                f"Looking up transform at time: {when.nanoseconds / 1e9:.6f} seconds"
            )

            # Look up the transform
            transform = self.tf_buffer.lookup_transform(
                self.controls_frame,
                input_pose.header.frame_id,
                when,
                Duration(seconds=timeout),
            )

            self.get_logger().debug(
                f"Found transform: translation [{transform.transform.translation.x}, "
                f"{transform.transform.translation.y}, {transform.transform.translation.z}], "
                f"rotation [{transform.transform.rotation.w}, {transform.transform.rotation.x}, "
                f"{transform.transform.rotation.y}, {transform.transform.rotation.z}]"
            )

            # Use the standard TF2 function to transform the pose
            transformed_pose = do_transform_pose(input_pose.pose, transform)

            # Create output pose with the correct header
            output_pose = PoseStamped()
            output_pose.header.stamp = self.get_clock().now().to_msg()
            output_pose.header.frame_id = self.controls_frame
            output_pose.pose = transformed_pose.pose

            self.get_logger().debug(
                f"Transformed pose: position [{output_pose.pose.position.x}, "
                f"{output_pose.pose.position.y}, {output_pose.pose.position.z}], "
                f"orientation [{output_pose.pose.orientation.w}, {output_pose.pose.orientation.x}, "
                f"{output_pose.pose.orientation.y}, {output_pose.pose.orientation.z}]"
            )

            # Set response
            response.output_pose = output_pose
            response.tf_success = True

            self.get_logger().info(
                f"Successfully transformed pose from '{input_pose.header.frame_id}' to '{self.controls_frame}'"
            )
            return response

        except Exception as e:
            self.get_logger().error(
                f"Failed to transform pose from '{input_pose.header.frame_id}' to '{self.controls_frame}': {str(e)}"
            )
            # Log the traceback for detailed debugging
            import traceback

            self.get_logger().debug(f"Exception traceback: {traceback.format_exc()}")

            response.tf_success = False
            # Still set output_pose with the original pose to avoid null values
            response.output_pose = input_pose
            return response


def main(args=None):
    rclpy.init(args=args)
    node = ConvertToControlsPose()
    rclpy.spin(node)
    rclpy.shutdown()
    # executor = MultiThreadedExecutor()
    # executor.add_node(node)

    # node.get_logger().info("ConvertToControlsPose node is running")

    # try:
    #     executor.spin()
    # except KeyboardInterrupt:
    #     node.get_logger().info("Keyboard interrupt, shutting down")
    # except Exception as e:
    #     node.get_logger().error(f"Unexpected error: {str(e)}")
    #     import traceback
    #     node.get_logger().debug(f"Exception traceback: {traceback.format_exc()}")
    # finally:
    #     node.get_logger().info("Cleaning up and shutting down")
    #     node.destroy_node()
    #     rclpy.shutdown()


if __name__ == "__main__":
    main()
