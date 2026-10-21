import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config_file = os.path.join(
        get_package_share_directory("frames"),
        "cfg",
        "asv5_buoy_gate_manager.yaml",
    )

    return LaunchDescription(
        [
            Node(
                package="frames",
                executable="buoy_gate_manager_node.py",
                name="buoy_gate_manager_node",
                namespace="asv5",
                output="screen",
                parameters=[config_file],
            )
        ]
    )
