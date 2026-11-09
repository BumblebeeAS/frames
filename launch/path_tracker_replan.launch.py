import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    cfg_dir = os.path.join(get_package_share_directory('frames'), 'cfg')

    return LaunchDescription([
        Node(
            package='frames',
            executable='path_tracker_replan',
            name='replan_tracker_node',
            output='screen',
            parameters=[os.path.join(cfg_dir, 'path_tracker_replan.yaml')]
        ),
        # Task 1 safe passage server (/asv5/safe_passage): a TrackPath client of the tracker above
        Node(
            package='frames',
            executable='safe_passage_node.py',
            name='safe_passage_node',
            output='screen',
            parameters=[os.path.join(cfg_dir, 'asv5_safe_passage.yaml')]
        ),
    ])
