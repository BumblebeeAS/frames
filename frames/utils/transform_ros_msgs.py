#!/usr/bin/env python3
from operator import attrgetter

import numpy as np
import tf2_ros
from geometry_msgs.msg import PoseStamped, Quaternion, Transform, Vector3
from nav_msgs.msg import Odometry
from tf2_geometry_msgs import do_transform_pose_stamped
from tf_transformations import (
    concatenate_matrices,
    inverse_matrix,
    quaternion_from_matrix,
    quaternion_inverse,
    quaternion_matrix,
    quaternion_multiply,
    translation_from_matrix,
    translation_matrix,
    unit_vector,
)


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
    odom_transform: tf2_ros.TransformStamped = get_base_to_world_tf_stamped(odom_msg)

    # Apply odom transform to get final pose in odom parent frame
    final_pose = do_transform_pose_stamped(transformed_pose, odom_transform)

    return final_pose


def compose_transforms(f: Transform, g: Transform):
    """
    Given two transforms f and g, computes gf.
    """

    tf_to_quat = lambda tf: unit_vector(
        np.array(attrgetter("x", "y", "z", "w")(tf.rotation))
    )
    tf_to_trans = lambda tf: np.array([*attrgetter("x", "y", "z")(tf.translation), 0.0])

    g_quat = tf_to_quat(g)
    g_trans = tf_to_trans(g)
    f_quat = tf_to_quat(f)
    f_trans = tf_to_trans(f)

    # Note that we want gf = R_g*(R_f*x + t_f) + t_g = R_g*R_f*x + (R_g*t_f + t_g)
    # The computations below are just the quaternion equivalents. In particular, for
    # applying rotations to vectors, we set w = 0, and do q * v * q^-1.
    # WESLEY DONT REMOVE THIS COMMENT
    rot = quaternion_multiply(g_quat, f_quat)
    trans = (
        quaternion_multiply(
            g_quat,
            quaternion_multiply(f_trans, quaternion_inverse(g_quat)),
        )
        + g_trans
    )

    composed = Transform()
    composed.rotation = Quaternion(x=rot[0], y=rot[1], z=rot[2], w=rot[3])
    composed.translation = Vector3(x=trans[0], y=trans[1], z=trans[2])

    return composed


def get_base_to_world_tf_stamped(odom_msg: Odometry) -> tf2_ros.TransformStamped:
    odom_transform = tf2_ros.TransformStamped()
    odom_transform.header = odom_msg.header
    odom_transform.transform.translation.x = odom_msg.pose.pose.position.x
    odom_transform.transform.translation.y = odom_msg.pose.pose.position.y
    odom_transform.transform.translation.z = odom_msg.pose.pose.position.z
    odom_transform.transform.rotation = odom_msg.pose.pose.orientation
    return odom_transform


def get_world_to_base_tf_stamped(
    odom_msg: Odometry, base_frame: str
) -> tf2_ros.TransformStamped:
    odom_transform = get_base_to_world_tf_stamped(odom_msg)
    translation = (
        odom_transform.transform.translation.x,
        odom_transform.transform.translation.y,
        odom_transform.transform.translation.z,
    )
    rotation = (
        odom_transform.transform.rotation.x,
        odom_transform.transform.rotation.y,
        odom_transform.transform.rotation.z,
        odom_transform.transform.rotation.w,
    )

    T = concatenate_matrices(
        translation_matrix(translation),
        quaternion_matrix(rotation),
    )
    T_inv = inverse_matrix(T)
    translation_inv = translation_from_matrix(T_inv)
    rotation_inv = quaternion_from_matrix(T_inv)
    world_to_base_tf = tf2_ros.TransformStamped()
    world_to_base_tf.header = odom_msg.header
    world_to_base_tf.header.frame_id = base_frame

    world_to_base_tf.transform.translation.x = translation_inv[0]
    world_to_base_tf.transform.translation.y = translation_inv[1]
    world_to_base_tf.transform.translation.z = translation_inv[2]
    world_to_base_tf.transform.rotation.x = rotation_inv[0]
    world_to_base_tf.transform.rotation.y = rotation_inv[1]
    world_to_base_tf.transform.rotation.z = rotation_inv[2]
    world_to_base_tf.transform.rotation.w = rotation_inv[3]
    return world_to_base_tf
