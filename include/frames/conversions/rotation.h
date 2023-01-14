#pragma once

#include <limits>
#include <boost/outcome.hpp>
#include <boost/outcome/result.hpp>
#include <boost/outcome/success_failure.hpp>

#include "frames/geometry/rotation.h"
#include "frames/math.h"

#include "geometry_msgs/Quaternion.h"
#include "geometry_msgs/QuaternionStamped.h"
#include "tf2/LinearMath/Quaternion.h"

namespace frame_utils::geometry
{
struct YPR_Deg
{
	double yaw;
	double pitch;
	double roll;
};

struct YPR_Rad
{
	double yaw;
	double pitch;
	double roll;
};

/**
 * @brief Factory function for creating a rotation from a quaternion
 *
 */
template<typename FrameConvention1, typename FrameConvention2>
[[nodiscard]] constexpr auto make_rotation(double w, double x, double y, double z) noexcept
	-> boost::outcome_v2::result<Rotation<FrameConvention1, FrameConvention2>>
{
	const auto results = Rotation<FrameConvention1, FrameConvention2> { w, x, y, z };

	if (math::is_close(results.norm(), 0., std::numeric_limits<float>::epsilon())) {
		return error_code::RotationConstructionErrc::NonUnitQuaternion;
	}

	return results.normalised();
}

/**
 * @brief Factory function for creating a rotation from a quaternion
 *
 */
template<typename FrameConvention1, typename FrameConvention2, typename Quaternion>
[[nodiscard]] constexpr auto make_rotation(const Quaternion& quat) noexcept
	-> boost::outcome_v2::result<Rotation<FrameConvention1, FrameConvention2>>
{
	const auto results = Rotation<FrameConvention1, FrameConvention2> { quat.w, quat.x, quat.y, quat.z };

	if (math::is_close(results.norm(), 0., std::numeric_limits<float>::epsilon())) {
		return error_code::RotationConstructionErrc::NonUnitQuaternion;
	}

	return results.normalised();
}

/**
 * @brief Factory function for creating a rotation from yaw-pitch-roll angles (degrees)
 *
 */
template<typename FrameConvention1, typename FrameConvention2>
[[nodiscard]] constexpr auto make_rotation(double yaw, double pitch, double roll)
	-> boost::outcome_v2::result<Rotation<FrameConvention1, FrameConvention2>>
{
	const auto roll_rad = math::deg2rad(roll);
	const auto pitch_rad = math::deg2rad(pitch);
	const auto yaw_rad = math::deg2rad(yaw);

	const auto cr = math::cos(roll_rad * 0.5);
	const auto sr = math::sin(roll_rad * 0.5);
	const auto cp = math::cos(pitch_rad * 0.5);
	const auto sp = math::sin(pitch_rad * 0.5);
	const auto cy = math::cos(yaw_rad * 0.5);
	const auto sy = math::sin(yaw_rad * 0.5);

	const auto results =
		Rotation<FrameConvention1, FrameConvention2> { cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
		                                               cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy };

	if (math::is_close(results.norm(), 0., std::numeric_limits<float>::epsilon())) {
		return error_code::RotationConstructionErrc::NonUnitQuaternion;
	}

	return results.normalised();
}

/**
 * @brief Factory function for creating a rotation from yaw-pitch-roll angles (degrees)
 *
 */
template<typename FrameConvention1, typename FrameConvention2>
[[nodiscard]] constexpr auto make_rotation(const YPR_Deg& ypr)
	-> boost::outcome_v2::result<Rotation<FrameConvention1, FrameConvention2>>
{
	return make_rotation<FrameConvention1, FrameConvention2>(ypr.yaw, ypr.pitch, ypr.roll);
}

/**
 * @brief Factory function for creating a rotation from yaw-pitch-roll angles (radian)
 *
 */
template<typename FrameConvention1, typename FrameConvention2>
[[nodiscard]] constexpr auto make_rotation(const YPR_Rad& ypr)
	-> boost::outcome_v2::result<Rotation<FrameConvention1, FrameConvention2>>
{
	return make_rotation<FrameConvention1, FrameConvention2>(
		math::rad2deg(ypr.yaw), math::rad2deg(ypr.pitch), math::rad2deg(ypr.roll)
	);
}

/**
 * @brief Extract yaw pitch roll angles (degrees)
 *
 */
template<typename FrameConvention1, typename FrameConvention2>
[[nodiscard]] constexpr auto to_ypr_deg(const Rotation<FrameConvention1, FrameConvention2>& rotation) -> YPR_Deg
{
	// yaw (z-axis rotation)
	const auto sinr_cosp = 2 * (rotation.getW() * rotation.getX() + rotation.getY() * rotation.getZ());
	const auto cosr_cosp = 1 - 2 * (rotation.getX() * rotation.getX() + rotation.getY() * rotation.getY());
	const auto yaw = math::atan2(sinr_cosp, cosr_cosp);

	// pitch (y-axis rotation)
	const auto sinp = 2 * (rotation.getW() * rotation.getY() - rotation.getZ() * rotation.getX());
	const auto pitch = math::abs(sinp) >= 1
	                       // use 90 degrees if out of range
	                       ? (sinp > 0 ? math::detail::pi<double>() / 2 : -math::detail::pi<double>() / 2)
	                       : math::asin(sinp);

	// roll (x-axis rotation)
	const auto siny_cosp = 2 * (rotation.getW() * rotation.getZ() + rotation.getX() * rotation.getY());
	const auto cosy_cosp = 1 - 2 * (rotation.getY() * rotation.getY() + rotation.getZ() * rotation.getZ());
	const auto roll = math::atan2(siny_cosp, cosy_cosp);
	return { math::rad2deg(roll), math::rad2deg(pitch), math::rad2deg(yaw) };
}

/**
 * @brief Extract yaw pitch roll angles (radian)
 *
 */
template<typename FrameConvention1, typename FrameConvention2>
[[nodiscard]] constexpr auto to_ypr_rad(const Rotation<FrameConvention1, FrameConvention2>& rotation) -> YPR_Rad
{
	const auto in_deg = to_ypr_deg(rotation);
	return { math::deg2rad(in_deg.roll), math::deg2rad(in_deg.pitch), math::deg2rad(in_deg.yaw) };
}
}