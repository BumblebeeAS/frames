#include "enu_ned_converter.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "geometry_msgs/msg/pose_with_covariance.hpp"
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/transform.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "nav_msgs/msg/path.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "geometry_msgs/msg/twist_with_covariance.hpp"
#include "geometry_msgs/msg/twist_with_covariance_stamped.hpp"
#include "geometry_msgs/msg/vector3.hpp"
#include "geometry_msgs/msg/vector3_stamped.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "geometry_msgs/msg/point_stamped.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "nav_msgs/msg/odometry.hpp"

// Define type aliases for each supported message type
// To uncomment after testing each one
// using PoseWithCovarianceEnuNed = EnuNedRepublisher<geometry_msgs::msg::PoseWithCovariance>;
using PoseWithCovarianceStampedEnuNed = EnuNedRepublisher<geometry_msgs::msg::PoseWithCovarianceStamped>;
// using PoseEnuNed = EnuNedRepublisher<geometry_msgs::msg::Pose>;
// using PoseStampedEnuNed = EnuNedRepublisher<geometry_msgs::msg::PoseStamped>;
// using TransformEnuNed = EnuNedRepublisher<geometry_msgs::msg::Transform>;
// using TransformStampedEnuNed = EnuNedRepublisher<geometry_msgs::msg::TransformStamped>;
using PathEnuNed = EnuNedRepublisher<nav_msgs::msg::Path>;
// using TwistEnuNed = EnuNedRepublisher<geometry_msgs::msg::Twist>;
// using TwistWithCovarianceEnuNed = EnuNedRepublisher<geometry_msgs::msg::TwistWithCovariance>;
using TwistWithCovarianceStampedEnuNed = EnuNedRepublisher<geometry_msgs::msg::TwistWithCovarianceStamped>;
using TwistStampedEnuNed = EnuNedRepublisher<geometry_msgs::msg::TwistStamped>;
// using Vector3EnuNed = EnuNedRepublisher<geometry_msgs::msg::Vector3>;
using Vector3StampedEnuNed = EnuNedRepublisher<geometry_msgs::msg::Vector3Stamped>;
// using PointEnuNed = EnuNedRepublisher<geometry_msgs::msg::Point>;
// using PointStampedEnuNed = EnuNedRepublisher<geometry_msgs::msg::PointStamped>;
using ImuEnuNed = EnuNedRepublisher<sensor_msgs::msg::Imu>;
using OdometryEnuNed = EnuNedRepublisher<nav_msgs::msg::Odometry>;

// Register each node
// RCLCPP_COMPONENTS_REGISTER_NODE(PoseWithCovarianceEnuNed)
RCLCPP_COMPONENTS_REGISTER_NODE(PoseWithCovarianceStampedEnuNed)
// RCLCPP_COMPONENTS_REGISTER_NODE(PoseEnuNed)
// RCLCPP_COMPONENTS_REGISTER_NODE(PoseStampedEnuNed)
// RCLCPP_COMPONENTS_REGISTER_NODE(TransformEnuNed)
// RCLCPP_COMPONENTS_REGISTER_NODE(TransformStampedEnuNed)
RCLCPP_COMPONENTS_REGISTER_NODE(PathEnuNed)
// RCLCPP_COMPONENTS_REGISTER_NODE(TwistEnuNed)
// RCLCPP_COMPONENTS_REGISTER_NODE(TwistWithCovarianceEnuNed)
RCLCPP_COMPONENTS_REGISTER_NODE(TwistWithCovarianceStampedEnuNed)
RCLCPP_COMPONENTS_REGISTER_NODE(TwistStampedEnuNed)
// RCLCPP_COMPONENTS_REGISTER_NODE(Vector3EnuNed)
RCLCPP_COMPONENTS_REGISTER_NODE(Vector3StampedEnuNed)
// RCLCPP_COMPONENTS_REGISTER_NODE(PointEnuNed)
// RCLCPP_COMPONENTS_REGISTER_NODE(PointStampedEnuNed)
RCLCPP_COMPONENTS_REGISTER_NODE(ImuEnuNed)
RCLCPP_COMPONENTS_REGISTER_NODE(OdometryEnuNed)
