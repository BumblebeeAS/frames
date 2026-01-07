#!/usr/bin/env python3
import tf2_ros
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from tf2_geometry_msgs import do_transform_pose_stamped


def transform_pose_to_odom(
    odom_msg: Odometry,
    pose_stamped_msg: PoseStamped,
    camera_to_odom_transform: tf2_ros.TransformStamped,
) -> PoseStamped:
    """Transform a PoseStamped message from the pose frame to odom parent frame.

    Args:
        odom_msg: Odometry message
        pose_stamped_msg: PoseStamped message
        camera_to_odom_transform: Transform from camera to odom_child frame

    Returns:
        PoseStamped in odom parent frame
    """

    # Transform pose from camera frame to odom child frame
    transformed_pose = do_transform_pose_stamped(
        pose_stamped_msg, camera_to_odom_transform
    )

    # Create transform from odom child to odom parent using odometry
    odom_transform = tf2_ros.TransformStamped()
    odom_transform.transform.translation.x = odom_msg.pose.pose.position.x
    odom_transform.transform.translation.y = odom_msg.pose.pose.position.y
    odom_transform.transform.translation.z = odom_msg.pose.pose.position.z
    odom_transform.transform.rotation = odom_msg.pose.pose.orientation

    # Apply odom transform to get final pose in odom parent frame
    final_pose = do_transform_pose_stamped(transformed_pose, odom_transform)

    return final_pose
