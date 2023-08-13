#pragma once

#include "conversions/rotation.h"
#include "coordinate_system/frame.h"
#include "coordinate_system/frame_convention.h"
#include "geometry/rotation.h"

namespace frame_utils::conversions::unsafe {
/**
 * @brief (Unsafe) convenience functions for orientation-related conversions.
 *
 * No compile-time frames checking are done in these functions.
 *
 */
template <typename Quaternion>
constexpr void convertOrientationToRPYDeg(const Quaternion& orientation, double& roll, double& pitch, double& yaw) {
    using namespace frame_utils::coordinates;
    using namespace frame_utils::geometry;
    const auto ypr = to_ypr_deg(make_rotation<__ArbitraryConvention, __ArbitraryConvention>(orientation).value());
    roll = ypr.roll;
    pitch = ypr.pitch;
    yaw = ypr.yaw;
}

template <typename Quaternion>
constexpr void convertRPYDegToOrientation(double roll, double pitch, double yaw, Quaternion& orientation) {
    using namespace frame_utils::coordinates;
    using namespace frame_utils::geometry;

    const auto _quat =
        make_rotation<__ArbitraryConvention, __ArbitraryConvention>(YPR_Deg{yaw, pitch, roll}).value();
    orientation.w = _quat.getW();
    orientation.x = _quat.getX();
    orientation.y = _quat.getY();
    orientation.z = _quat.getZ();
}

template <typename Quaternion>
constexpr void convertOrientationToRPYRad(const Quaternion& orientation, double& roll, double& pitch, double& yaw) {
    using namespace frame_utils::coordinates;
    using namespace frame_utils::geometry;

    const auto ypr = to_ypr_rad(make_rotation<__ArbitraryConvention, __ArbitraryConvention>(orientation).value());
    roll = ypr.roll;
    pitch = ypr.pitch;
    yaw = ypr.yaw;
}

template <typename Quaternion>
constexpr void convertRPYRadToOrientation(double roll, double pitch, double yaw, Quaternion& orientation) {
    using namespace frame_utils::coordinates;
    using namespace frame_utils::geometry;

    const auto _quat =
        make_rotation<__ArbitraryConvention, __ArbitraryConvention>(YPR_Rad{yaw, pitch, roll}).value();
    orientation.w = _quat.getW();
    orientation.x = _quat.getX();
    orientation.y = _quat.getY();
    orientation.z = _quat.getZ();
}

}  // namespace frame_utils::conversions::unsafe
