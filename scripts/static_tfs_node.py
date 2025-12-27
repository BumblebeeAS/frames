#!/usr/bin/env python3

import math
import traceback
from typing import Any, Dict, List

import rclpy
import yaml
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from tf2_ros import StaticTransformBroadcaster
from tf_transformations import quaternion_from_euler


class StaticTfsNode(Node):
    """
    Publishes multiple static transforms in a single ROS2 node.
    This helps to reduce the number of nodes when many static transforms are needed
    compared to using multiple static_transform_publisher nodes.
    """

    def __init__(self):
        super().__init__("static_tfs_node")
        self.declare_parameter("single_tfs_file", "")
        self.declare_parameter("grouped_tfs_file", "")
        self.declare_parameter("default_suffix", "view")

        single_tfs_config_path = (
            self.get_parameter("single_tfs_file").get_parameter_value().string_value
        )
        grouped_tfs_config_path = (
            self.get_parameter("grouped_tfs_file").get_parameter_value().string_value
        )
        self.default_suffix = (
            self.get_parameter("default_suffix").get_parameter_value().string_value
        )

        if not single_tfs_config_path and not grouped_tfs_config_path:
            self.get_logger().error(
                "No config files provided! Use --ros-args -p single_tfs_file:=<path> "
                "and/or -p grouped_tfs_file:=<path>"
            )
            return

        self.tf_static_broadcaster = StaticTransformBroadcaster(self)

        all_transforms: List[TransformStamped] = []

        if single_tfs_config_path:
            single_tfs = self.load_single_tfs(single_tfs_config_path)
            all_transforms.extend(single_tfs)

        if grouped_tfs_config_path:
            grouped_tfs = self.load_grouped_tfs(grouped_tfs_config_path)
            all_transforms.extend(grouped_tfs)

        # Publish all transforms
        if all_transforms:
            self.tf_static_broadcaster.sendTransform(all_transforms)
            self.get_logger().info(
                f"Published {len(all_transforms)} mission transforms"
            )
            for tf in all_transforms:
                self.get_logger().info(
                    f"Published {tf.header.frame_id} -> {tf.child_frame_id}"
                )

    def load_config_file(self, config_file_path: str) -> Dict[str, Any]:
        """Load and parse YAML config file."""
        try:
            with open(config_file_path, "r") as file:
                config = yaml.safe_load(file)
            return config

        except FileNotFoundError:
            self.get_logger().error(
                f"Config file not found: {config_file_path}. "
                f"Traceback:\n{traceback.format_exc()}"
            )
            raise

        except yaml.YAMLError:
            self.get_logger().error(
                f"Error parsing YAML file: {config_file_path}. "
                f"Traceback:\n{traceback.format_exc()}"
            )
            raise

        except Exception:
            self.get_logger().error(
                f"Error loading config file: {config_file_path}. "
                f"Traceback:\n{traceback.format_exc()}"
            )
            raise

    def load_single_tfs(self, config_file_path: str) -> List[TransformStamped]:
        """Load transforms from single YAML config and return them."""
        try:
            config = self.load_config_file(config_file_path)
            single_transforms: List[TransformStamped] = []

            for _, transforms_list in config.items():
                for tf_config in transforms_list:
                    transform = self.create_single_transform(tf_config)
                    single_transforms.append(transform)

            self.get_logger().info(f"Loaded {len(single_transforms)} single transforms")
            return single_transforms

        except Exception:
            self.get_logger().error(
                f"Error loading single transforms. Traceback:\n{traceback.format_exc()}"
            )
            raise

    def load_grouped_tfs(self, config_file_path: str) -> List[TransformStamped]:
        """Load transforms from grouped YAML config and return them."""
        try:
            config = self.load_config_file(config_file_path)
            grouped_transforms: List[TransformStamped] = []

            for _, grouped_configs in config.items():
                for grouped_config in grouped_configs:
                    transforms = self.create_grouped_transforms(grouped_config)
                    grouped_transforms.extend(transforms)

            self.get_logger().info(
                f"Loaded {len(grouped_transforms)} grouped transforms"
            )
            return grouped_transforms

        except Exception:
            self.get_logger().error(
                f"Error loading grouped transforms. Traceback:\n{traceback.format_exc()}"
            )
            raise

    def create_single_transform(self, tf_config: Dict[str, Any]) -> TransformStamped:
        """Create a TransformStamped message from single tf configuration."""
        try:
            return self.create_transform(
                x=tf_config.get("x", 0.0),
                y=tf_config.get("y", 0.0),
                z=tf_config.get("z", 0.0),
                roll=tf_config.get("roll", 0.0),
                pitch=tf_config.get("pitch", 0.0),
                yaw=tf_config.get("yaw", 0.0),
                parent_frame=tf_config["parent_frame_id"],
                child_frame=tf_config["child_frame_id"],
            )

        except Exception:
            self.get_logger().error(
                f"Error creating single transform from config: {tf_config}. "
                f"Traceback:\n{traceback.format_exc()}"
            )
            raise

    def create_grouped_transforms(
        self, grouped_config: Dict[str, Any]
    ) -> List[TransformStamped]:
        """Create TransformStamped messages from grouped configuration."""
        try:
            transforms: List[TransformStamped] = []
            parents = grouped_config.get("parents", [])

            suffix = grouped_config.get("suffix", self.default_suffix)

            for parent_frame in parents:
                child_frame = f"{parent_frame}/{suffix}"
                transform = self.create_transform(
                    x=grouped_config.get("x", 0.0),
                    y=grouped_config.get("y", 0.0),
                    z=grouped_config.get("z", 0.0),
                    roll=grouped_config.get("roll", 0.0),
                    pitch=grouped_config.get("pitch", 0.0),
                    yaw=grouped_config.get("yaw", 0.0),
                    parent_frame=parent_frame,
                    child_frame=child_frame,
                )
                transforms.append(transform)

            return transforms

        except Exception:
            self.get_logger().error(
                f"Error creating grouped transforms from config: {grouped_config}. "
                f"Traceback:\n{traceback.format_exc()}"
            )
            raise

    def create_transform(
        self,
        x: float,
        y: float,
        z: float,
        roll: float,
        pitch: float,
        yaw: float,
        parent_frame: str,
        child_frame: str,
    ) -> TransformStamped:
        """Create a TransformStamped message from transform parameters."""
        try:
            roll_rad = math.radians(roll)
            pitch_rad = math.radians(pitch)
            yaw_rad = math.radians(yaw)

            transform = TransformStamped()
            transform.header.stamp = self.get_clock().now().to_msg()
            transform.header.frame_id = parent_frame
            transform.child_frame_id = child_frame

            transform.transform.translation.x = float(x)
            transform.transform.translation.y = float(y)
            transform.transform.translation.z = float(z)

            quaternion = quaternion_from_euler(roll_rad, pitch_rad, yaw_rad)
            transform.transform.rotation.x = quaternion[0]
            transform.transform.rotation.y = quaternion[1]
            transform.transform.rotation.z = quaternion[2]
            transform.transform.rotation.w = quaternion[3]

            return transform

        except Exception:
            self.get_logger().error(
                f"Error creating transform. " f"Traceback:\n{traceback.format_exc()}"
            )
            raise


def main(args=None):
    rclpy.init(args=args)
    static_tfs_node = StaticTfsNode()
    try:
        rclpy.spin(static_tfs_node)
    except KeyboardInterrupt:
        pass
    finally:
        static_tfs_node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
