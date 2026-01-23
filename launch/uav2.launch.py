from launch import LaunchDescription
from launch_ros.actions import Node, PushRosNamespace


def generate_launch_description():
    launch_objects = [PushRosNamespace("/uav2")]

    nodes = [
        Node(
            package="frames",
            executable="convert_to_controls_pose.py",
            name="convert_to_controls_pose",
            parameters=[
                {
                    "controls_frame": "odom_ned",
                    "base_frame": "uav2/base_link_frd",
                    "queue_size": 100,
                    "odom_topic": "/uav2/odom_ned",
                }
            ],
            output="screen",
        )
    ]

    return LaunchDescription(launch_objects + nodes)
