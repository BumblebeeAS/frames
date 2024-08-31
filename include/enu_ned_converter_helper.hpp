#ifndef ENU_NED_CONVERTER_HELPER_HPP
#define ENU_NED_CONVERTER_HELPER_HPP

#include <cmath>
#include <string>
#include <vector>
#include <unordered_map>
#include <functional>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <geometry_msgs/msg/transform.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <geometry_msgs/msg/twist_with_covariance.hpp>
#include <geometry_msgs/msg/twist_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <geometry_msgs/msg/vector3_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <Eigen/Dense>

inline geometry_msgs::msg::Quaternion operator*(const geometry_msgs::msg::Quaternion& q1, const geometry_msgs::msg::Quaternion& q2) {
    geometry_msgs::msg::Quaternion result;
    result.x = q1.x * q2.w + q1.y * q2.z - q1.z * q2.y + q1.w * q2.x;
    result.y = -q1.x * q2.z + q1.y * q2.w + q1.z * q2.x + q1.w * q2.y;
    result.z = q1.x * q2.y - q1.y * q2.x + q1.z * q2.w + q1.w * q2.z;
    result.w = -q1.x * q2.x - q1.y * q2.y - q1.z * q2.z + q1.w * q2.w;
    return result;
}

const geometry_msgs::msg::Quaternion Q_BODY = [] {
    geometry_msgs::msg::Quaternion q;
    q.x = 1.0;
    q.y = 0.0;
    q.z = 0.0;
    q.w = 0.0;
    return q;
}();

const geometry_msgs::msg::Quaternion Q_WORLD = [] {
    geometry_msgs::msg::Quaternion q;
    q.x = std::sqrt(2) / 2;
    q.y = std::sqrt(2) / 2;
    q.z = 0.0;
    q.w = 0.0;
    return q;
}();

template <typename T>
void transform_xyz(T& msg, bool body = false) {
    if (body) {
        msg.y = -msg.y;
        msg.z = -msg.z;
    } else {
        std::swap(msg.x, msg.y);
        msg.z = -msg.z;
    }
}

void transform_twist(geometry_msgs::msg::Twist& msg, bool body = false) {
    transform_xyz(msg.linear, body);
    transform_xyz(msg.angular, body);
}

void transform_twist_with_covariance(geometry_msgs::msg::TwistWithCovariance& msg, bool body = false) {
    transform_twist(msg.twist, body);
    // Covariance transformation is not included for simplicity
}

void transform_orientation(geometry_msgs::msg::Quaternion& orientation, bool body = false, bool transform_child = true) {
    geometry_msgs::msg::Quaternion q = (body ? Q_BODY : Q_WORLD) * orientation;
    if (transform_child) {
        q = q * Q_BODY;
    }
    orientation = q;
}

void transform_pose(geometry_msgs::msg::Pose& msg, bool body = false, bool transform_child = false) {
    transform_xyz(msg.position, body);
    transform_orientation(msg.orientation, body, transform_child);
}

void transform_pose_with_covariance(geometry_msgs::msg::PoseWithCovariance& msg, bool body = false, bool transform_child = false) {
    transform_pose(msg.pose, body, transform_child);
    // Covariance transformation is not included for simplicity
}

std::string transform_frame_name(const std::string& name) {
    return name.size() > 4 && name.substr(name.size() - 4) == "_ned" ? name.substr(0, name.size() - 4) : name + "_ned";
}

bool is_ned(const std::string& frame_name) {
    return frame_name.size() > 4 && frame_name.substr(frame_name.size() - 4) == "_ned";
}

void transform_imu(sensor_msgs::msg::Imu& msg, bool /*unused*/) {
    msg.header.frame_id = transform_frame_name(msg.header.frame_id);
    transform_orientation(msg.orientation, false, true);
    transform_xyz(msg.angular_velocity, true);
    transform_xyz(msg.linear_acceleration, true);
}

void transform_odometry(nav_msgs::msg::Odometry& msg, bool is_body = true) {
    msg.header.frame_id = transform_frame_name(msg.header.frame_id);
    msg.child_frame_id = transform_frame_name(msg.child_frame_id);
    transform_pose_with_covariance(msg.pose, is_body, true);
    transform_twist_with_covariance(msg.twist, true);
}

void transform_path(nav_msgs::msg::Path& msg, bool is_body = true, bool transform_child = true) {
    for (auto& pose : msg.poses) {
        pose.header.frame_id = transform_frame_name(pose.header.frame_id);
        transform_pose(pose.pose, is_body, transform_child);
    }
    msg.header.frame_id = transform_frame_name(msg.header.frame_id);
}

void transform_transform(geometry_msgs::msg::Transform& msg, bool is_body = true, bool transform_child = true) {
    transform_xyz(msg.translation, is_body);    
    transform_orientation(msg.rotation, is_body, transform_child);
}

template <typename T>
void transform_msg(T& msg, bool is_body = false, bool transform_child = false) {
    if constexpr (std::is_same_v<T, geometry_msgs::msg::PoseWithCovariance>) {
        transform_pose_with_covariance(msg, is_body, transform_child);
    } else if constexpr (std::is_same_v<T, geometry_msgs::msg::PoseWithCovarianceStamped>) {
        transform_pose_with_covariance(msg.pose, is_body, transform_child);
        msg.header.frame_id = transform_frame_name(msg.header.frame_id);
    } else if constexpr (std::is_same_v<T, geometry_msgs::msg::Pose>) {
        transform_pose(msg, is_body, transform_child);
    } else if constexpr (std::is_same_v<T, geometry_msgs::msg::PoseStamped>) {
        transform_pose(msg.pose, is_body, transform_child);
        msg.header.frame_id = transform_frame_name(msg.header.frame_id);
    } else if constexpr (std::is_same_v<T, geometry_msgs::msg::TransformStamped>) {
        transform_transform(msg.transform, is_body, transform_child);
        msg.header.frame_id = transform_frame_name(msg.header.frame_id);
        if (transform_child)
            msg.child_frame_id = transform_frame_name(msg.child_frame_id);
    } else if constexpr (std::is_same_v<T, geometry_msgs::msg::Transform>) {
        transform_transform(msg, is_body, transform_child);
    } else if constexpr (std::is_same_v<T, nav_msgs::msg::Path>) {
        transform_path(msg, is_body, transform_child);
    } else if constexpr (std::is_same_v<T, geometry_msgs::msg::Twist>) {
        transform_twist(msg, is_body);
    } else if constexpr (std::is_same_v<T, geometry_msgs::msg::TwistStamped>) {
        transform_twist(msg.twist, is_body);
        msg.header.frame_id = transform_frame_name(msg.header.frame_id);
    } else if constexpr (std::is_same_v<T, geometry_msgs::msg::TwistWithCovariance>) {
        transform_twist_with_covariance(msg, is_body);
    } else if constexpr (std::is_same_v<T, geometry_msgs::msg::TwistWithCovarianceStamped>) {
        transform_twist_with_covariance(msg.twist, is_body);
    } else if constexpr (std::is_same_v<T, geometry_msgs::msg::Vector3>) {
        transform_xyz(msg, is_body);
    } else if constexpr (std::is_same_v<T, geometry_msgs::msg::Vector3Stamped>) {
        transform_xyz(msg.vector, is_body);
    } else if constexpr (std::is_same_v<T, geometry_msgs::msg::Point>) {
        transform_xyz(msg, is_body);
    } else if constexpr (std::is_same_v<T, geometry_msgs::msg::PointStamped>) {
        transform_xyz(msg.point, is_body);
        msg.header.frame_id = transform_frame_name(msg.header.frame_id);
    } else if constexpr (std::is_same_v<T, sensor_msgs::msg::Imu>) {
        transform_imu(msg, is_body);
    } else if constexpr (std::is_same_v<T, nav_msgs::msg::Odometry>) {
        transform_odometry(msg, is_body);
    }
}

#endif // ENU_NED_CONVERTER_HELPER_HPP
