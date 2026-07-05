from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    ld = LaunchDescription()
    ld.add_action(
        DeclareLaunchArgument("controls_frame", default_value="asv5/odom_ned")
    )
    ld.add_action(DeclareLaunchArgument("base_frame", default_value="asv5/base_link"))
    ld.add_action(DeclareLaunchArgument("use_sim_time", default_value="true"))

    ld.add_action(
        DeclareLaunchArgument(
            "namespace",
            default_value="asv5",
            description="Namespace of the vehicle",
        )
    )
    dest_pose_from_tf_node = Node(
        package="frames",
        namespace=LaunchConfiguration("namespace"),
        executable="dest_pose_from_tf",
        name="convert_frame",
        parameters=[{"use_sim_time": LaunchConfiguration("use_sim_time")}],
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
                "queue_size": 100,
                "odom_topic": "/asv5/odometry/filtered/local/ned",
                "use_sim_time": LaunchConfiguration("use_sim_time"),
            }
        ],
        output="screen",
    )
    ld.add_action(convert_to_controls_pose)

    return ld
