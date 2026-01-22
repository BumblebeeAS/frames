#!/usr/bin/env python3
from operator import attrgetter
from typing import Any, Optional

import numpy as np
import rclpy
import tf2_ros
from bb_planner_msgs.srv import GetPoseToControlsFrame
from geometry_msgs.msg import PoseStamped, Quaternion, Transform, Vector3
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
from rclpy.time import Time
from tf2_geometry_msgs import do_transform_pose_stamped
from tf2_msgs.msg import TFMessage
from tf2_ros.buffer import Buffer
from tf_transformations import (
    concatenate_matrices,
    inverse_matrix,
    quaternion_from_matrix,
    quaternion_inverse,
    quaternion_matrix,
    quaternion_multiply,
    translation_from_matrix,
    translation_matrix,
    unit_vector,
)


def compose_transforms(f: Transform, g: Transform):
    """
    Given two transforms f and g, computes gf.
    """

    tf_to_quat = lambda tf: unit_vector(
        np.array(attrgetter("x", "y", "z", "w")(tf.rotation))
    )
    tf_to_trans = lambda tf: np.array([*attrgetter("x", "y", "z")(tf.translation), 0.0])

    g_quat = tf_to_quat(g)
    g_trans = tf_to_trans(g)
    f_quat = tf_to_quat(f)
    f_trans = tf_to_trans(f)

    # Note that we want gf = R_g*(R_f*x + t_f) + t_g = R_g*R_f*x + (R_g*t_f + t_g)
    # The computations below are just the quaternion equivalents. In particular, for
    # applying rotations to vectors, we set w = 0, and do q * v * q^-1.
    # WESLEY DONT REMOVE THIS COMMENT
    rot = quaternion_multiply(g_quat, f_quat)
    trans = (
        quaternion_multiply(
            g_quat,
            quaternion_multiply(f_trans, quaternion_inverse(g_quat)),
        )
        + g_trans
    )

    composed = Transform()
    composed.rotation = Quaternion(x=rot[0], y=rot[1], z=rot[2], w=rot[3])
    composed.translation = Vector3(x=trans[0], y=trans[1], z=trans[2])

    return composed


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

        static_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.tf_buffer: Buffer = Buffer()
        self.tf_sub = self.create_subscription(
            TFMessage,
            "/tf_static",
            self.handle_tf_static,
            qos_profile=static_qos,
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            "/uav2/odom_ned",
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

    def handle_tf_static(self, msg: TFMessage) -> None:
        """
        Callback to set static transforms in the TF buffer.
        """
        self.get_logger().info("Received static TF transforms")
        for tf in msg.transforms:
            self.tf_buffer.set_transform_static(tf, "default_authority")

    def handle_odom(self, msg: Odometry) -> None:
        self.odom = msg

    def transform_to_frame(
        self, input_pose: PoseStamped, target_frame: list[str], timeout: float
    ) -> PoseStamped | None:
        """
        Transform a pose to the specified target frame.

        Args:
            input_pose: The pose to transform
            target_frame: The target coordinate frame
            timeout: Maximum time to wait for the transform, in seconds

        Returns:
            The transformed pose

        Raises:
            Exception: If the transform fails
        """
        if input_pose.header.frame_id == target_frame:
            return input_pose

        is_from_base_link = False
        self.get_logger().debug(
            f"Transforming pose from '{input_pose.header.frame_id}' to '{target_frame}'"
        )
        try:
            target_to_input_transform = self.tf_buffer.lookup_transform(
                target_frame=target_frame[0],
                source_frame=input_pose.header.frame_id,
                time=Time.from_msg(input_pose.header.stamp),
                timeout=Duration(seconds=timeout),  # type: ignore
            )
            is_from_base_link = True
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            self.get_logger().error(f"Transform lookup failed: {e}")

        if not is_from_base_link:
            # tf is from world to input frame
            try:
                target_to_input_transform = self.tf_buffer.lookup_transform(
                    target_frame=target_frame[1],
                    source_frame=input_pose.header.frame_id,
                    time=Time.from_msg(input_pose.header.stamp),
                    timeout=Duration(seconds=timeout),  # type: ignore
                )
                world_to_base_tf = self.get_world_to_base_tf_stamped()
                target_to_input_transform = compose_transforms(
                    target_to_input_transform.transform, world_to_base_tf.transform
                )
                target_to_input_transform = tf2_ros.TransformStamped(
                    header=world_to_base_tf.header,
                    child_frame_id=input_pose.header.frame_id,
                    transform=target_to_input_transform,
                )
            except (
                tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException,
            ) as e:
                self.get_logger().error(f"Transform lookup failed: {e}")
                return None

        output_pose = do_transform_pose_stamped(input_pose, target_to_input_transform)

        # output_pose = self.tf_buffer.transform(
        #     input_pose,
        #     target_frame,
        #     Duration(seconds=timeout),  # type: ignore
        # )

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
            Time.from_msg(original_target.header.stamp),
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

        self.get_logger().info(
            f"Received transform request for {len(input_poses)} poses to '{self.controls_frame}'"
        )

        if timeout <= 0.0:
            timeout = 1.0
            self.get_logger().debug(f"Using default timeout of {timeout} seconds")
        else:
            self.get_logger().debug(f"Using specified timeout of {timeout} seconds")

        output_poses = []

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
                    response.output_poses = input_poses
                    response.tf_success = False
                    self.get_logger().error(
                        f"Failed to transform pose {i + 1} to base frame '{self.base_frame}'"
                    )
                    break

                # Step 2: Recalculate pose based on anchor frame
                recalculated_pose = self.recalculate_target(
                    pose_in_base_frame, anchor_frame, timeout
                )

                # Step 3: Transform to controls frame
                # final_pose = self.transform_to_frame(
                #     recalculated_pose, self.controls_frame, timeout
                # )
                odom_transform = self.get_base_to_world_tf_stamped()

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

            response.output_poses = input_poses
            response.tf_success = False

        return response

    def get_base_to_world_tf_stamped(self):
        odom_transform = tf2_ros.TransformStamped()
        odom_transform.header = self.odom.header
        odom_transform.transform.translation.x = self.odom.pose.pose.position.x
        odom_transform.transform.translation.y = self.odom.pose.pose.position.y
        odom_transform.transform.translation.z = self.odom.pose.pose.position.z
        odom_transform.transform.rotation = self.odom.pose.pose.orientation
        return odom_transform

    def get_world_to_base_tf_stamped(self):
        odom_transform = self.get_base_to_world_tf_stamped()
        translation = (
            odom_transform.transform.translation.x,
            odom_transform.transform.translation.y,
            odom_transform.transform.translation.z,
        )
        rotation = (
            odom_transform.transform.rotation.x,
            odom_transform.transform.rotation.y,
            odom_transform.transform.rotation.z,
            odom_transform.transform.rotation.w,
        )

        T = concatenate_matrices(
            translation_matrix(translation),
            quaternion_matrix(rotation),
        )
        T_inv = inverse_matrix(T)
        translation_inv = translation_from_matrix(T_inv)
        rotation_inv = quaternion_from_matrix(T_inv)
        world_to_base_tf = tf2_ros.TransformStamped()
        world_to_base_tf.header = self.odom.header
        world_to_base_tf.header.frame_id = self.base_frame

        world_to_base_tf.transform.translation.x = translation_inv[0]
        world_to_base_tf.transform.translation.y = translation_inv[1]
        world_to_base_tf.transform.translation.z = translation_inv[2]
        world_to_base_tf.transform.rotation.x = rotation_inv[0]
        world_to_base_tf.transform.rotation.y = rotation_inv[1]
        world_to_base_tf.transform.rotation.z = rotation_inv[2]
        world_to_base_tf.transform.rotation.w = rotation_inv[3]
        return world_to_base_tf


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
