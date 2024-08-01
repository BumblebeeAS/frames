from launch_ros.actions import Node
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    ld = LaunchDescription()
    ld.add_action(
        DeclareLaunchArgument(
            "namespace",
            default_value="uav2",
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
    return ld
