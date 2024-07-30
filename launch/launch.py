from launch_ros.actions import Node

from launch import LaunchDescription

package_name = "frames"
vehicle_name = "uav2"


def generate_launch_description():
    ld = LaunchDescription()
    dest_pose_from_tf_node = Node(
        package=package_name,
        executable="dest_pose_from_tf",
        name="dest_pose_from_tf_node",
    )
    ld.add_action(dest_pose_from_tf_node)
    return ld
