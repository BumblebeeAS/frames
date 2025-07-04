from launch_ros.actions import Node

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    ld = LaunchDescription()
    ld.add_action(DeclareLaunchArgument("controls_frame", default_value="map_ned"))
    ld.add_action(
        DeclareLaunchArgument("base_frame", default_value="orca4_ned")
    )

    ld.add_action(
        DeclareLaunchArgument(
            "namespace",
            default_value="mini",
            description="Namespace of the vehicle",
        )
    )
    dest_pose_from_tf_node = Node(
        package="frames",
        namespace=LaunchConfiguration("namespace"),
        executable="dest_pose_from_tf",
        name="convert_frame",
    )
    ld.add_action(dest_pose_from_tf_node)

    convert_to_controls_pose = Node(
        package="frames",
        namespace=LaunchConfiguration("namespace"),
        executable="convert_to_controls_pose.py",
        name="convert_to_controls_pose",
        parameters=[
            {
                "controls_frame": LaunchConfiguration("controls_frame"),
                "base_frame": LaunchConfiguration("base_frame"),
            }
        ],
        output="screen",
    )
    ld.add_action(convert_to_controls_pose)

    return ld
