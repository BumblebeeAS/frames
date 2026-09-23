import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    config_file = os.path.join(
        get_package_share_directory('frames'),
        'cfg',
        'path_tracker.yaml'
    )

    return LaunchDescription([
        Node(
            package='frames',
            executable='path_tracker',
            name='path_tracker_node',
            output='screen',
            parameters=[config_file]
        )
    ])
